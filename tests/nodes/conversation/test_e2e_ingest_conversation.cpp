// test_e2e_ingest_conversation.cpp — End-to-end integration test
//
// Runs the REAL VideoIngestNode and AudioIngestNode with file-based sources,
// feeding frames and audio through GStreamer → SHM → ZMQ into ConversationNode.
//
// Pipeline under test (same as production):
//
//   MP4 file → GStreamer → VideoIngestNode → SHM /oe.test.e2e.vid → ConversationNode
//   WAV file → GStreamer → AudioIngestNode → SileroVAD → SHM /oe.test.e2e.aud
//              → ZMQ audio_chunk/vad_status → ConversationNode → transcription
//
// Prerequisites:
//   - GStreamer 1.x with decodebin, videoconvert, wavparse, audioresample
//   - models/silero_vad.onnx (Silero VAD v5 ONNX model)
//   - tests/fixtures/video_talking/talking_test_3s.mp4  (3 s, 1920x1080, 30 fps)
//   - tests/fixtures/video_talking/talking_speech_3s.wav (3 s, 16 kHz, F32LE mono)

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "conversation/conversation_node.hpp"
#include "conversation_inferencer.hpp"
#include "ingest/video_ingest_node.hpp"
#include "ingest/audio_ingest_node.hpp"
#include "common/constants/conversation_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "common/runtime_defaults.hpp"
#include "shm/shm_frame_reader.hpp"
#include "zmq/audio_constants.hpp"

#include "tests/mocks/mock_conversation_inferencer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"


// ---------------------------------------------------------------------------
// Port allocation — fully isolated from production and other test suites.
// All three nodes (video ingest, audio ingest, conversation) use unique ports.
// ---------------------------------------------------------------------------
static constexpr int kE2eVideoIngestPub    = 27100;
static constexpr int kE2eAudioIngestPub    = 27101;
static constexpr int kE2eConversationPub   = 27102;
static constexpr int kE2eDaemonSub         = 27103;
static constexpr int kE2eUiCmdSub          = 27104;
static constexpr int kE2eWsBridgeSub       = 27105;

// SHM names — VideoIngestNode and AudioIngestNode hardcode these names.
// The test must use the same names so ConversationNode can find the segments.
static constexpr const char* kE2eVideoShmName = "/oe.vid.ingest";
static constexpr const char* kE2eAudioShmName = "/oe.aud.ingest";


// ---------------------------------------------------------------------------
// RAII scope guard — ensures node.stop() + thread.join() on early ASSERT exit.
// Without this, a failed ASSERT causes the test function to return while
// std::thread objects are still joinable, which calls std::terminate().
// ---------------------------------------------------------------------------

struct ScopedCleanup {
    std::function<void()> cleanup;
    explicit ScopedCleanup(std::function<void()> fn) : cleanup(std::move(fn)) {}
    ~ScopedCleanup() { if (cleanup) cleanup(); }
    ScopedCleanup(const ScopedCleanup&) = delete;
    ScopedCleanup& operator=(const ScopedCleanup&) = delete;
};


// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class E2eIngestConversationTest : public ::testing::Test {
protected:
    std::filesystem::path projectRoot_{CMAKE_SOURCE_DIR};
    zmq::context_t ctx_{1};
};


// ---------------------------------------------------------------------------
// E2E: VideoIngestNode (file) → SHM → ConversationNode → generateWithVideo
//
// Runs VideoIngestNode with a real MP4 as source.  GStreamer decodes the
// video, writes BGR24 frames to SHM at 30 FPS.  ConversationNode reads the
// latest frame from the same SHM and calls generateWithVideo().
// ---------------------------------------------------------------------------

TEST_F(E2eIngestConversationTest, VideoFileIngestToConversationNode)
{
    const auto videoPath = projectRoot_ / "tests/fixtures/video_talking/talking_test_3s.mp4";
    if (!std::filesystem::exists(videoPath)) {
        GTEST_SKIP() << "Video fixture not found: " << videoPath;
    }

    // ── Configure VideoIngestNode with file source ──
    VideoIngestNode::Config videoCfg;
    videoCfg.v4l2Device      = videoPath.string();
    videoCfg.frameWidth      = kMaxInputWidth;   // 1920
    videoCfg.frameHeight     = kMaxInputHeight;  // 1080
    videoCfg.pubPort         = kE2eVideoIngestPub;
    videoCfg.wsBridgeSubPort = kE2eWsBridgeSub;
    videoCfg.moduleName      = "video_ingest_e2e";

    // ── Configure ConversationNode to read video from the same SHM ──
    ConversationNode::Config convCfg;
    convCfg.pubPort          = kE2eConversationPub;
    convCfg.daemonSubPort    = kE2eDaemonSub;
    convCfg.uiCommandSubPort = kE2eUiCmdSub;
    convCfg.modelDir         = "/tmp/test_model";
    convCfg.modelVariant     = "gemma_e4b";
    convCfg.pollTimeout      = std::chrono::milliseconds(20);
    convCfg.videoShmName     = kE2eVideoShmName;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    auto* mockPtr = mockInferencer.get();
    mockInferencer->nativeVision = true;
    mockInferencer->nativeTts = false;
    mockInferencer->tokensToEmit = {"I", " see", " a", " film", "."};

    // ── Bind ZMQ control sockets BEFORE ConversationNode subscribes ──
    zmq::socket_t daemonPub(ctx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kE2eDaemonSub));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kE2eDaemonSub));

    zmq::socket_t uiPub(ctx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kE2eUiCmdSub));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kE2eUiCmdSub));

    // ── Start VideoIngestNode (creates SHM, starts GStreamer pipeline) ──
    VideoIngestNode videoNode(videoCfg);
    SPDLOG_INFO("E2E: initializing VideoIngestNode with file={}", videoPath.string());
    videoNode.initialize();

    std::thread videoThread([&videoNode]() { videoNode.run(); });

    // Pointers for nodes/threads created later — used by the cleanup guard.
    ConversationNode* convNodePtr = nullptr;
    std::thread* convThreadPtr = nullptr;

    // RAII guard: stop nodes and join threads even if an ASSERT fails mid-test.
    ScopedCleanup guard([&]() {
        if (convNodePtr) convNodePtr->stop();
        if (convThreadPtr && convThreadPtr->joinable()) convThreadPtr->join();
        videoNode.stop();
        if (videoThread.joinable()) videoThread.join();
    });

    // Poll SHM until GStreamer writes at least one frame (up to 5 s).
    // The fixed-sleep approach is flaky because GStreamer startup time varies.
    {
        const auto videoShmSize = ShmCircularBuffer<ShmVideoHeader>::segmentSize(
            kCircularBufferSlotCount, kMaxBgr24FrameBytes);
        ShmMapping videoShmCheck(kE2eVideoShmName, videoShmSize, /*create=*/false);
        const auto shmDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool frameReady = false;
        while (std::chrono::steady_clock::now() < shmDeadline) {
            auto frame = readLatestBgrFrame(videoShmCheck);
            if (frame.data != nullptr && frame.width > 0) {
                SPDLOG_INFO("E2E: first frame in SHM: {}x{}", frame.width, frame.height);
                frameReady = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_TRUE(frameReady)
            << "GStreamer did not write any frames to SHM within 5 seconds";
    }

    // ── Start ConversationNode (opens existing SHM, subscribes to ZMQ) ──
    ConversationNode convNode(convCfg, std::move(mockInferencer));
    convNode.initialize();
    convNodePtr = &convNode;

    std::thread convThread([&convNode]() { convNode.run(); });
    convThreadPtr = &convThread;

    // Subscribe AFTER node.initialize() + run — avoids ZMQ slow-joiner bug where
    // cleanStaleEndpoint() removes the IPC socket, causing the subscriber to
    // auto-reconnect and lose its SUB filter.
    zmq::socket_t responseSub(ctx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kE2eConversationPub));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Enable video conversation.
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send a prompt — ConversationNode should read the latest frame from
    // the SHM that VideoIngestNode is actively writing to.
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "What do you see?"},
        {"sequence_id", 700},
    };

    // Collect tokens with retry logic.
    std::vector<nlohmann::json> received;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds{5000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{200});
        if (rc > 0) {
            for (int attempt = 0; attempt < 30; ++attempt) {
                auto msg = receiveTestMessage(responseSub, 200);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    // ── Shutdown (explicit stop before assertions; guard handles join) ──
    convNode.stop();
    convThread.join();
    videoNode.stop();
    videoThread.join();

    // Clear guard — cleanup already done above.
    convThreadPtr = nullptr;
    convNodePtr = nullptr;

    // ── Assertions ──
    ASSERT_GE(received.size(), 1u) << "Expected tokens from ConversationNode";
    EXPECT_TRUE(received.back().value("finished", false));

    EXPECT_TRUE(mockPtr->lastGenerateHadVideo)
        << "ConversationNode must call generateWithVideo() with a real frame "
           "read from SHM written by VideoIngestNode (GStreamer → SHM → node)";
    EXPECT_EQ(mockPtr->lastFrameWidth, kMaxInputWidth);
    EXPECT_EQ(mockPtr->lastFrameHeight, kMaxInputHeight);
    EXPECT_EQ(mockPtr->lastVideoFrameSize, kMaxBgr24FrameBytes);

    std::string fullText;
    for (const auto& msg : received) {
        fullText += msg.value("token", "");
    }
    EXPECT_EQ(fullText, "I see a film.");

    SPDLOG_INFO("E2E VideoIngest→Conversation: frame={}x{} ({} bytes), "
                "generateWithVideo={}, text='{}'",
        mockPtr->lastFrameWidth, mockPtr->lastFrameHeight,
        mockPtr->lastVideoFrameSize, mockPtr->lastGenerateHadVideo, fullText);
}


// ---------------------------------------------------------------------------
// E2E: AudioIngestNode (WAV file) → SileroVAD → SHM + ZMQ → ConversationNode
//
// Runs AudioIngestNode with a real WAV speech file.  GStreamer decodes the
// audio, Silero VAD classifies each 30 ms chunk, speech chunks are written
// to SHM and published via ZMQ.  ConversationNode accumulates the audio in
// AudioAccumulator, then transcribes on vad_status silence.
// ---------------------------------------------------------------------------

TEST_F(E2eIngestConversationTest, AudioWavIngestToConversationTranscription)
{
    const auto wavPath = projectRoot_ / "tests/fixtures/video_talking/talking_speech_3s.wav";
    const auto vadModel = projectRoot_ / "models/silero_vad.onnx";

    if (!std::filesystem::exists(wavPath)) {
        GTEST_SKIP() << "WAV fixture not found: " << wavPath;
    }
    if (!std::filesystem::exists(vadModel)) {
        GTEST_SKIP() << "Silero VAD model not found: " << vadModel;
    }

    // ── Configure AudioIngestNode with WAV file source ──
    AudioIngestNode::Config audioCfg;
    audioCfg.audioSource       = wavPath.string();   // triggers file mode (filesrc)
    audioCfg.sampleRateHz      = kSttInputSampleRateHz;  // 16000
    audioCfg.chunkSamples      = kVadChunkSampleCount;   // 480
    audioCfg.vadModelPath      = vadModel.string();
    audioCfg.vadSpeechThreshold = 0.3f;  // lower threshold to catch more speech
    audioCfg.silenceDurationMs = 500;    // faster silence detection for test
    audioCfg.pubPort           = kE2eAudioIngestPub;
    audioCfg.wsBridgeSubPort   = kE2eWsBridgeSub;
    audioCfg.moduleName        = "audio_ingest_e2e";

    // ── Configure ConversationNode to consume audio from AudioIngestNode ──
    ConversationNode::Config convCfg;
    convCfg.pubPort          = kE2eConversationPub + 10;  // avoid port collision with video test
    convCfg.daemonSubPort    = kE2eDaemonSub + 10;
    convCfg.uiCommandSubPort = kE2eUiCmdSub + 10;
    convCfg.modelDir         = "/tmp/test_model";
    convCfg.modelVariant     = "gemma_e4b";
    convCfg.pollTimeout      = std::chrono::milliseconds(20);
    convCfg.audioShmName     = kE2eAudioShmName;
    convCfg.audioSubPort     = kE2eAudioIngestPub;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = false;
    mockInferencer->mockTranscription = "duck and cover";
    mockInferencer->tokensToEmit = {"OK"};

    // ── Start AudioIngestNode (loads Silero VAD, creates audio SHM) ──
    AudioIngestNode audioNode(audioCfg);
    SPDLOG_INFO("E2E: initializing AudioIngestNode with wav={}", wavPath.string());
    audioNode.initialize();

    // Give audio node time to create its SHM before conversation node opens it.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // ── Bind ZMQ control sockets BEFORE ConversationNode subscribes ──
    zmq::socket_t daemonPub(ctx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", convCfg.daemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", convCfg.daemonSubPort));

    zmq::socket_t uiPub(ctx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", convCfg.uiCommandSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", convCfg.uiCommandSubPort));

    // ── Start ConversationNode ──
    ConversationNode convNode(convCfg, std::move(mockInferencer));
    convNode.initialize();

    // Subscribe to transcription output AFTER ConversationNode's PUB is bound,
    // avoiding the ZMQ slow-joiner problem where subscriptions don't propagate
    // in time for one-shot messages.
    zmq::socket_t transcriptionSub(ctx_, ZMQ_SUB);
    transcriptionSub.set(zmq::sockopt::subscribe, std::string("transcription"));
    transcriptionSub.connect(
        std::format("ipc:///tmp/omniedge_{}", convCfg.pubPort));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::thread convThread([&convNode]() { convNode.run(); });

    // ── Run AudioIngestNode — GStreamer reads WAV, VAD filters, publishes ──
    std::thread audioThread([&audioNode]() { audioNode.run(); });

    // RAII guard: stop nodes and join threads even if an ASSERT fails mid-test.
    ScopedCleanup guard([&]() {
        convNode.stop();
        if (convThread.joinable()) convThread.join();
        audioNode.stop();
        if (audioThread.joinable()) audioThread.join();
    });

    // Wait for the audio file to be fully consumed + silence detection.
    // 3 s of audio + 0.5 s silence timeout + margin = ~5 s.
    nlohmann::json transcription;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds{8000};
    while (std::chrono::steady_clock::now() < deadline) {
        auto msg = receiveTestMessage(transcriptionSub, 300);
        if (!msg.is_null() && msg.value("type", "") == "transcription") {
            transcription = msg;
            break;
        }
    }

    // ── Shutdown (explicit; guard handles cleanup if assertions fail below) ──
    convNode.stop();
    convThread.join();
    audioNode.stop();
    audioThread.join();

    // ── Assertions ──
    ASSERT_FALSE(transcription.is_null())
        << "Expected transcription from ConversationNode after real speech "
           "flowed through AudioIngestNode → SileroVAD → SHM → AudioAccumulator → transcribe()";
    EXPECT_EQ(transcription.value("text", ""), "duck and cover");
    EXPECT_EQ(transcription.value("source", ""), "native_stt");

    SPDLOG_INFO("E2E AudioIngest→Conversation: transcription='{}', source='{}'",
        transcription.value("text", ""), transcription.value("source", ""));
}
