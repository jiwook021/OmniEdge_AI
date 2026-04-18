#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "stt/stt_node.hpp"
#include "stt_inferencer.hpp"
#include "stt/hallucination_filter.hpp"
#include "common/runtime_defaults.hpp"
#include "zmq/audio_constants.hpp"
#include "common/constants/whisper_constants.hpp"
#include "tests/mocks/mock_stt_inferencer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

// ---------------------------------------------------------------------------
// STTNode tests — CPU-only.
// No GPU, no TRT engine required.  The test creates a real POSIX SHM segment
// to feed audio to the node and verifies observable ZMQ output.
//
// Port offset: +20000 (avoids collisions with production and LLM tests)
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// SHM layout constants — must mirror stt_node.cpp exactly
// ---------------------------------------------------------------------------

constexpr std::size_t kAudioDataOffset = 192;  // ShmCircularBuffer: header(64) + control(128, alignas(64))

constexpr std::size_t kSlotSize =
    ((sizeof(ShmAudioHeader)
      + static_cast<std::size_t>(kVadChunkSampleCount) * sizeof(float))
     + 7) & ~std::size_t{7};

constexpr std::size_t kShmSize =
    kAudioDataOffset + kRingBufferSlotCount * kSlotSize;

// ---------------------------------------------------------------------------
// STT-specific mock that extends the shared MockSTTInferencer with call tracking.
// Includes "stt/stt_inferencer.hpp" via the shared mock header (DRY).
// ---------------------------------------------------------------------------

class TrackingMockSTTInferencer : public MockSTTInferencer {
public:
    TrackingMockSTTInferencer()
    {
        // Pre-configure base class canned values
        cannedTranscription = "test transcription";
        cannedLanguage      = "en";
        cannedNoSpeechProb  = 0.05f;
        cannedAvgLogprob    = -0.3f;
        // Ensure base considers model loaded (loadModel is a no-op here)
        MockSTTInferencer::loadModel("", "", "");
    }

    [[nodiscard]] tl::expected<TranscribeResult, std::string>
    transcribe(std::span<const float> melSpectrogram, uint32_t /*numFrames*/) override
    {
        ++transcribeCallCount;
        if (forceTranscribeError) {
            return tl::unexpected(std::string("mock inference error"));
        }
        melInputWasNonEmpty = !melSpectrogram.empty();

        TranscribeResult result;
        result.text         = transcriptionResult.text;
        result.language     = transcriptionResult.language;
        result.noSpeechProb = transcriptionResult.noSpeechProb;
        result.avgLogprob   = transcriptionResult.avgLogprob;
        return result;
    }

    /// The transcription result returned by transcribe().
    TranscribeResult transcriptionResult{
        .text         = "test transcription",
        .language     = "en",
        .noSpeechProb = 0.05f,
        .avgLogprob   = -0.3f,
    };
    /// When true, transcribe() returns an error.
    bool forceTranscribeError     = false;
    /// Set to true after transcribe() receives non-empty mel data.
    bool melInputWasNonEmpty      = false;
    /// Number of times transcribe() was called.
    int  transcribeCallCount      = 0;
};

// ---------------------------------------------------------------------------
// POSIX SHM helper — RAII for test SHM segments
// ---------------------------------------------------------------------------

struct TestShm {
    explicit TestShm(const char* segmentName) : segmentName_(segmentName)
    {
        // Unlink any stale segment from a previous failed test run.
        shm_unlink(segmentName_);

        const int fileDescriptor = shm_open(segmentName_, O_CREAT | O_RDWR | O_TRUNC, 0600);
        if (fileDescriptor < 0) throw std::runtime_error(
            std::string("shm_open failed: ") + strerror(errno));
        if (ftruncate(fileDescriptor, static_cast<off_t>(kShmSize)) != 0) {
            close(fileDescriptor);
            throw std::runtime_error("ftruncate failed");
        }
        mappedRegion_ = static_cast<uint8_t*>(
            mmap(nullptr, kShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0));
        close(fileDescriptor);
        if (mappedRegion_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        std::memset(mappedRegion_, 0, kShmSize);
    }

    ~TestShm()
    {
        if (mappedRegion_ && mappedRegion_ != MAP_FAILED) {
            munmap(mappedRegion_, kShmSize);
        }
        shm_unlink(segmentName_);
    }

    /// Write silence samples into slot 0 and set a valid sequence number.
    void writeSilenceSlot(int /*nSamples*/)
    {
        uint8_t* slotBase = mappedRegion_ + kAudioDataOffset;
        auto* audioSlotHeader = reinterpret_cast<ShmAudioHeader*>(slotBase);
        audioSlotHeader->seqNumber = 1;
        // PCM samples remain zero (silence) — ftruncate already zeroed them.
    }

    uint8_t*    mappedRegion_{nullptr};
    const char* segmentName_;
};

// ZMQ helper — uses shared test_zmq_helpers.hpp (DRY).
// Alias for brevity within this file.
[[nodiscard]] static nlohmann::json recvPubFrame(zmq::socket_t& subscriberSocket,
                                                  int timeoutMs = 1000)
{
    return receiveTestMessage(subscriberSocket, timeoutMs);
}

// ---------------------------------------------------------------------------
// Whisper compile-time constants
// ---------------------------------------------------------------------------

TEST(WhisperConstantsTest, HopLengthIs160)
{
    // 10 ms × 16 kHz = 160 samples
    EXPECT_EQ(kHopLengthSamples, 160u);
}

TEST(WhisperConstantsTest, NFFTIs400)
{
    // 25 ms × 16 kHz = 400 samples
    EXPECT_EQ(kFftWindowSizeSamples, 400u);
}

TEST(WhisperConstantsTest, NFramesIs3000)
{
    // 30 s × 16 000 / 160 = 3 000
    EXPECT_EQ(kFramesPerChunk, 3'000u);
}

TEST(WhisperConstantsTest, EncoderFramesIs1500)
{
    // After two Conv1D stride-2 layers: 3000 / 2 = 1500
    EXPECT_EQ(kEncoderOutputFrameCount, 1'500u);
}

TEST(WhisperConstantsTest, MaxTokensIs448)
{
    EXPECT_EQ(kMaxDecoderTokens, 448u);
}

// ---------------------------------------------------------------------------
// stop() idempotency — safe to call before initialize() or multiple times
// ---------------------------------------------------------------------------

TEST(STTNodeTest, StopBeforeInitializeDoesNotCrash)
{
    STTNode::Config cfg;
    cfg.pubPort      = 25563;  // +20000 offset — avoids clash with running system
    cfg.audioSubPort = 25556;

    STTNode node(cfg);
    // No initialize() — stop() must be safe to call regardless
    EXPECT_NO_THROW(node.stop());
}

TEST(STTNodeTest, StopIsIdempotent)
{
    STTNode::Config cfg;
    cfg.pubPort      = 25563;
    cfg.audioSubPort = 25556;

    STTNode node(cfg);
    EXPECT_NO_THROW(node.stop());
    EXPECT_NO_THROW(node.stop());  // second call must not crash
    EXPECT_NO_THROW(node.stop());  // third call too
}

// ---------------------------------------------------------------------------
// TranscribePCMProducesNonEmptyTranscript
//
// Feed a full 30 s window of silence (kVadChunkSamples) to the node via
// POSIX SHM + ZMQ audio_chunk message.  Verify that a "transcription" message
// with the MockSTTInferencer's canned text is published on the PUB socket.
// ---------------------------------------------------------------------------

TEST(STTNodeTest, TranscribePCMProducesNonEmptyTranscript)
{
    static constexpr int kPubPort  = 25600;
    static constexpr int kAudPort  = 25601;
    static const char*   kShmName  = "/oe_test_stt_transcribe";

    // 1 — create POSIX SHM and write silence into slot 0
    TestShm shm(kShmName);
    const int nSamples = static_cast<int>(kVadChunkSampleCount);
    shm.writeSilenceSlot(nSamples);

    // 2 — subscribe to "transcription" on the node's PUB port BEFORE init
    zmq::context_t testCtx(1);
    zmq::socket_t sub(testCtx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("transcription"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

    // 3 — bind a PUB socket that impersonates AudioIngestNode
    zmq::socket_t audioPub(testCtx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
    audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

    // 4 — configure and initialise the node
    auto mockInferencer = std::make_unique<TrackingMockSTTInferencer>();
    TrackingMockSTTInferencer* mockInferencerPtr = mockInferencer.get();

    STTNode::Config nodeConfig;
    nodeConfig.pubPort        = kPubPort;
    nodeConfig.audioSubPort   = kAudPort;
    nodeConfig.inputShmName   = kShmName;
    nodeConfig.pollTimeout    = std::chrono::milliseconds(20);
    nodeConfig.chunkSamples   = static_cast<std::size_t>(nSamples);  // trigger after one chunk
    nodeConfig.overlapSamples = 0;

    STTNode node(nodeConfig);
    node.setInferencer(std::move(mockInferencer));
    node.initialize();

    // 5 — run() in a background thread
    std::thread runThread([&node]{ node.run(); });

    // 6 — send audio_chunk with retry until a transcription is received
    const nlohmann::json audioChunkMetadata = {
        {"slot",    0},
        {"samples", nSamples},
    };
    const std::string audioChunkFrame = "audio_chunk " + audioChunkMetadata.dump();

    nlohmann::json receivedTranscription;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);

    while (std::chrono::steady_clock::now() < deadline) {
        audioPub.send(zmq::buffer(audioChunkFrame), zmq::send_flags::none);
        receivedTranscription = recvPubFrame(sub, 200);
        if (!receivedTranscription.is_null()) { break; }
    }

    node.stop();
    runThread.join();

    // 7 — verify the published transcription
    ASSERT_FALSE(receivedTranscription.is_null()) << "No transcription message received";
    EXPECT_EQ(receivedTranscription.value("type", std::string{}), std::string("transcription"));
    EXPECT_EQ(receivedTranscription.value("text", std::string{}), std::string("test transcription"));
    EXPECT_TRUE(mockInferencerPtr->melInputWasNonEmpty)
        << "Inferencer was never called with non-empty mel frames";
}

// ---------------------------------------------------------------------------
// HallucinationFilterRejectsHighNoSpeechProb
//
// Same SHM+ZMQ setup but MockSTTInferencer returns noSpeechProb = 0.99 (above
// the default threshold of 0.6).  Verify that NO transcription is published.
// ---------------------------------------------------------------------------

TEST(STTNodeTest, HallucinationFilterRejectsHighNoSpeechProb)
{
    static constexpr int kPubPort  = 25610;
    static constexpr int kAudPort  = 25611;
    static const char*   kShmName  = "/oe_test_stt_hallucination";

    // 1 — create POSIX SHM
    TestShm shm(kShmName);
    const int nSamples = static_cast<int>(kVadChunkSampleCount);
    shm.writeSilenceSlot(nSamples);

    // 2 — subscribe to "transcription"
    zmq::context_t testCtx(1);
    zmq::socket_t sub(testCtx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("transcription"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

    // 3 — bind audio PUB
    zmq::socket_t audioPub(testCtx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
    audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

    // 4 — mock returns high noSpeechProb (should be filtered out)
    auto mockInferencer = std::make_unique<TrackingMockSTTInferencer>();
    mockInferencer->transcriptionResult.noSpeechProb = 0.99f;
    mockInferencer->transcriptionResult.text         = "should not be published";

    STTNode::Config nodeConfig;
    nodeConfig.pubPort               = kPubPort;
    nodeConfig.audioSubPort          = kAudPort;
    nodeConfig.inputShmName          = kShmName;
    nodeConfig.pollTimeout           = std::chrono::milliseconds(20);
    nodeConfig.chunkSamples          = static_cast<std::size_t>(nSamples);
    nodeConfig.overlapSamples        = 0;
    nodeConfig.noSpeechProbThreshold = 0.6f;  // default: reject if noSpeechProb >= 0.6

    STTNode node(nodeConfig);
    node.setInferencer(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]{ node.run(); });

    const nlohmann::json chunkMeta = {
        {"slot",    0},
        {"samples", nSamples},
    };
    const std::string chunkFrame = "audio_chunk " + chunkMeta.dump();

    // Send the chunk several times to ensure the node processes it.
    for (int i = 0; i < 5; ++i) {
        audioPub.send(zmq::buffer(chunkFrame), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Wait up to 500 ms for any transcription (should NOT arrive).
    const nlohmann::json received = recvPubFrame(sub, 500);

    node.stop();
    runThread.join();

    EXPECT_TRUE(received.is_null())
        << "Transcription was published but hallucination filter should have suppressed it";
}

// ---------------------------------------------------------------------------
// VadSilenceSkipsInference
//
// Send vad_status(speaking=false) with no prior audio chunks.  The inferencer
// should NOT be called because the audio buffer is empty — there is nothing
// to transcribe.  This catches the bug where VAD silence triggers inference
// on an empty buffer, wasting GPU cycles or producing garbage output.
// ---------------------------------------------------------------------------

TEST(STTNodeTest, VadSilenceWithEmptyBufferSkipsInferencer)
{
    static constexpr int kPubPort  = 25620;
    static constexpr int kAudPort  = 25621;
    static const char*   kShmName  = "/oe_test_stt_vad_silence";

    // 1 — create POSIX SHM (required for initialize())
    TestShm shm(kShmName);

    // 2 — subscribe to "transcription"
    zmq::context_t testCtx(1);
    zmq::socket_t sub(testCtx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("transcription"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

    // 3 — bind audio PUB (impersonates AudioIngestNode)
    zmq::socket_t audioPub(testCtx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
    audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

    // 4 — configure node with mock inferencer
    auto mockInferencer = std::make_unique<TrackingMockSTTInferencer>();
    TrackingMockSTTInferencer* mockInferencerPtr = mockInferencer.get();

    STTNode::Config nodeConfig;
    nodeConfig.pubPort        = kPubPort;
    nodeConfig.audioSubPort   = kAudPort;
    nodeConfig.inputShmName   = kShmName;
    nodeConfig.pollTimeout    = std::chrono::milliseconds(20);

    STTNode node(nodeConfig);
    node.setInferencer(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]{ node.run(); });

    // 5 — send vad_status(speaking=false) WITHOUT any prior audio_chunk
    const nlohmann::json vadStatusPayload = {{"speaking", false}, {"silence_duration_ms", 800}};
    const std::string vadStatusFrame = "vad_status " + vadStatusPayload.dump();

    // Brief delay so ZMQ subscriptions propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    for (int i = 0; i < 3; ++i) {
        audioPub.send(zmq::buffer(vadStatusFrame), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // 6 — wait briefly — no transcription should arrive
    const nlohmann::json receivedTranscription = recvPubFrame(sub, 300);

    node.stop();
    runThread.join();

    // 7 — verify: inferencer was never called, no transcription published
    EXPECT_EQ(mockInferencerPtr->transcribeCallCount, 0)
        << "Inferencer should not be called when audio buffer is empty on VAD silence";
    EXPECT_TRUE(receivedTranscription.is_null())
        << "No transcription should be published when no audio was accumulated";
}

// ---------------------------------------------------------------------------
// InferencerErrorDoesNotCrash
//
// Feed audio into the node but make the mock inferencer return an error.
// Verify the node does NOT crash, does NOT publish a transcription, and
// continues running (can be stopped cleanly).
// ---------------------------------------------------------------------------

TEST(STTNodeTest, InferencerErrorDoesNotCrash)
{
    static constexpr int kPubPort  = 25630;
    static constexpr int kAudPort  = 25631;
    static const char*   kShmName  = "/oe_test_stt_inferencer_err";

    TestShm shm(kShmName);
    const int nSamples = static_cast<int>(kVadChunkSampleCount);
    shm.writeSilenceSlot(nSamples);

    zmq::context_t testCtx(1);
    zmq::socket_t sub(testCtx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("transcription"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

    zmq::socket_t audioPub(testCtx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
    audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

    auto mockInferencer = std::make_unique<TrackingMockSTTInferencer>();
    mockInferencer->forceTranscribeError = true;  // inferencer will return error on every call

    STTNode::Config nodeConfig;
    nodeConfig.pubPort        = kPubPort;
    nodeConfig.audioSubPort   = kAudPort;
    nodeConfig.inputShmName   = kShmName;
    nodeConfig.pollTimeout    = std::chrono::milliseconds(20);
    nodeConfig.chunkSamples   = static_cast<std::size_t>(nSamples);
    nodeConfig.overlapSamples = 0;

    STTNode node(nodeConfig);
    node.setInferencer(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]{ node.run(); });

    // Send audio chunks — triggers inference which will fail
    const nlohmann::json chunkMeta = {{"slot", 0}, {"samples", nSamples}};
    const std::string chunkFrame = "audio_chunk " + chunkMeta.dump();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (int i = 0; i < 3; ++i) {
        audioPub.send(zmq::buffer(chunkFrame), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // No transcription should be published (inferencer errored)
    const nlohmann::json received = recvPubFrame(sub, 300);

    // Node should still be alive — stop cleanly
    node.stop();
    runThread.join();

    EXPECT_TRUE(received.is_null())
        << "No transcription should be published when inferencer returns error";
}

// ---------------------------------------------------------------------------
// MalformedAudioChunkDoesNotCrash
//
// Send a malformed ZMQ message (not valid JSON) on the audio_chunk topic.
// The node should log a warning and continue running — no crash.
// ---------------------------------------------------------------------------

TEST(STTNodeTest, MalformedAudioChunkDoesNotCrash)
{
    static constexpr int kPubPort  = 25640;
    static constexpr int kAudPort  = 25641;
    static const char*   kShmName  = "/oe_test_stt_malformed";

    TestShm shm(kShmName);

    zmq::context_t testCtx(1);
    zmq::socket_t audioPub(testCtx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
    audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

    auto mockInferencer = std::make_unique<TrackingMockSTTInferencer>();
    TrackingMockSTTInferencer* mockInferencerPtr = mockInferencer.get();

    STTNode::Config nodeConfig;
    nodeConfig.pubPort        = kPubPort;
    nodeConfig.audioSubPort   = kAudPort;
    nodeConfig.inputShmName   = kShmName;
    nodeConfig.pollTimeout    = std::chrono::milliseconds(20);

    STTNode node(nodeConfig);
    node.setInferencer(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]{ node.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Send malformed JSON on audio_chunk topic
    const std::string malformedMessage = "audio_chunk {{{not json at all";
    for (int i = 0; i < 3; ++i) {
        audioPub.send(zmq::buffer(malformedMessage), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Give the node time to process the malformed messages
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Node must still be alive
    node.stop();
    runThread.join();

    // Inferencer should never have been called — malformed input never reaches inference
    EXPECT_EQ(mockInferencerPtr->transcribeCallCount, 0)
        << "Inferencer should not be called for malformed input";
}

