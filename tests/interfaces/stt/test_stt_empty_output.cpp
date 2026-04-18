#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "stt/stt_node.hpp"
#include "stt_inferencer.hpp"
#include "common/runtime_defaults.hpp"
#include "zmq/audio_constants.hpp"
#include "common/constants/whisper_constants.hpp"
#include "tests/mocks/mock_stt_inferencer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

// ---------------------------------------------------------------------------
// STTNode empty-output tests — CPU-only.
//
// Verifies that the STT node does NOT publish empty or whitespace-only
// transcriptions on the PUB socket.  The bug: STT publishes empty text,
// causing LLM to receive empty user content and produce garbage output.
//
// Port offset: +20000 (avoids collisions with production and other tests).
// Ports start at 25660 to avoid collisions with test_stt_node.cpp (25600-25641).
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
// Same pattern as test_stt_node.cpp — duplicated here because each test
// binary links independently and may configure the mock differently.
// ---------------------------------------------------------------------------

class TrackingMockSTTInferencer_EmptyOutput : public MockSTTInferencer {
public:
	TrackingMockSTTInferencer_EmptyOutput()
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
	/// Set to true after transcribe() receives non-empty mel data.
	bool melInputWasNonEmpty = false;
	/// Number of times transcribe() was called.
	int  transcribeCallCount = 0;
};

// ---------------------------------------------------------------------------
// POSIX SHM helper — RAII for test SHM segments
// ---------------------------------------------------------------------------

struct TestShm_EmptyOutput {
	explicit TestShm_EmptyOutput(const char* segmentName) : segmentName_(segmentName)
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

	~TestShm_EmptyOutput()
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

// ZMQ helper — alias for brevity within this file.
[[nodiscard]] static nlohmann::json recvPubFrame(zmq::socket_t& subscriberSocket,
                                                  int timeoutMs = 1000)
{
	return receiveTestMessage(subscriberSocket, timeoutMs);
}

// ---------------------------------------------------------------------------
// STTEmptyOutput_EmptyText_NotPublished
//
// Bug: STT node publishes empty transcription text, causing downstream LLM
// to receive an empty user message → garbage response.
//
// Set transcription text to "" with noSpeechProb below threshold (i.e. the
// inferencer claims speech was detected, but the decoded text is empty).
// Verify: no "transcription" message is published.
// ---------------------------------------------------------------------------

TEST(STTEmptyOutputTest, EmptyText_NotPublished)
{
	static constexpr int kPubPort  = 25660;
	static constexpr int kAudPort  = 25661;
	static const char*   kShmName  = "/oe_test_stt_empty_text";

	// 1 -- create POSIX SHM and write silence into slot 0
	TestShm_EmptyOutput shm(kShmName);
	const int nSamples = static_cast<int>(kVadChunkSampleCount);
	shm.writeSilenceSlot(nSamples);

	// 2 -- subscribe to "transcription" on the node's PUB port BEFORE init
	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("transcription"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

	// 3 -- bind a PUB socket that impersonates AudioIngestNode
	zmq::socket_t audioPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
	audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

	// 4 -- configure mock: empty text, low noSpeechProb (below threshold)
	auto mockInferencer = std::make_unique<TrackingMockSTTInferencer_EmptyOutput>();
	mockInferencer->transcriptionResult.text         = "";
	mockInferencer->transcriptionResult.noSpeechProb = 0.05f;  // below 0.6 threshold
	mockInferencer->transcriptionResult.avgLogprob   = -0.3f;

	STTNode::Config nodeConfig;
	nodeConfig.pubPort               = kPubPort;
	nodeConfig.audioSubPort          = kAudPort;
	nodeConfig.inputShmName          = kShmName;
	nodeConfig.pollTimeout           = std::chrono::milliseconds(20);
	nodeConfig.chunkSamples          = static_cast<std::size_t>(nSamples);
	nodeConfig.overlapSamples        = 0;
	nodeConfig.noSpeechProbThreshold = 0.6f;

	STTNode node(nodeConfig);
	node.setInferencer(std::move(mockInferencer));
	node.initialize();

	// 5 -- run() in a background thread
	std::thread runThread([&node]{ node.run(); });

	// 6 -- send audio_chunk repeatedly to trigger transcription
	const nlohmann::json audioChunkMetadata = {
		{"slot",    0},
		{"samples", nSamples},
	};
	const std::string audioChunkFrame = "audio_chunk " + audioChunkMetadata.dump();

	// Brief delay so ZMQ subscriptions propagate
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	for (int i = 0; i < 5; ++i) {
		audioPub.send(zmq::buffer(audioChunkFrame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	// 7 -- wait up to 500 ms for any transcription (should NOT arrive)
	const nlohmann::json received = recvPubFrame(sub, 500);

	node.stop();
	runThread.join();

	EXPECT_TRUE(received.is_null())
		<< "Empty transcription text was published — STT node should suppress "
		   "empty-text transcriptions to prevent LLM from receiving empty user content";
}

// ---------------------------------------------------------------------------
// STTEmptyOutput_WhitespaceOnlyText_NotPublished
//
// Same bug variant: inferencer returns whitespace-only text ("   ").
// This should also be suppressed — whitespace is semantically empty.
// ---------------------------------------------------------------------------

TEST(STTEmptyOutputTest, WhitespaceOnlyText_NotPublished)
{
	static constexpr int kPubPort  = 25662;
	static constexpr int kAudPort  = 25663;
	static const char*   kShmName  = "/oe_test_stt_whitespace_text";

	TestShm_EmptyOutput shm(kShmName);
	const int nSamples = static_cast<int>(kVadChunkSampleCount);
	shm.writeSilenceSlot(nSamples);

	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("transcription"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

	zmq::socket_t audioPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
	audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

	// Mock returns whitespace-only text, noSpeechProb below threshold
	auto mockInferencer = std::make_unique<TrackingMockSTTInferencer_EmptyOutput>();
	mockInferencer->transcriptionResult.text         = "   ";
	mockInferencer->transcriptionResult.noSpeechProb = 0.05f;
	mockInferencer->transcriptionResult.avgLogprob   = -0.3f;

	STTNode::Config nodeConfig;
	nodeConfig.pubPort               = kPubPort;
	nodeConfig.audioSubPort          = kAudPort;
	nodeConfig.inputShmName          = kShmName;
	nodeConfig.pollTimeout           = std::chrono::milliseconds(20);
	nodeConfig.chunkSamples          = static_cast<std::size_t>(nSamples);
	nodeConfig.overlapSamples        = 0;
	nodeConfig.noSpeechProbThreshold = 0.6f;

	STTNode node(nodeConfig);
	node.setInferencer(std::move(mockInferencer));
	node.initialize();

	std::thread runThread([&node]{ node.run(); });

	const nlohmann::json audioChunkMetadata = {
		{"slot",    0},
		{"samples", nSamples},
	};
	const std::string audioChunkFrame = "audio_chunk " + audioChunkMetadata.dump();

	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	for (int i = 0; i < 5; ++i) {
		audioPub.send(zmq::buffer(audioChunkFrame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	const nlohmann::json received = recvPubFrame(sub, 500);

	node.stop();
	runThread.join();

	EXPECT_TRUE(received.is_null())
		<< "Whitespace-only transcription was published — STT node should trim "
		   "and suppress whitespace-only text to prevent empty LLM user content";
}

// ---------------------------------------------------------------------------
// STTEmptyOutput_NonEmptyText_IsPublished
//
// Control test: verify that a non-empty transcription IS published.
// If this fails, the test infrastructure is broken, not the empty-text guard.
// ---------------------------------------------------------------------------

TEST(STTEmptyOutputTest, NonEmptyText_IsPublished)
{
	static constexpr int kPubPort  = 25664;
	static constexpr int kAudPort  = 25665;
	static const char*   kShmName  = "/oe_test_stt_nonempty_text";

	TestShm_EmptyOutput shm(kShmName);
	const int nSamples = static_cast<int>(kVadChunkSampleCount);
	shm.writeSilenceSlot(nSamples);

	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("transcription"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

	zmq::socket_t audioPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kAudPort));
	audioPub.bind(std::format("ipc:///tmp/omniedge_{}", kAudPort));

	// Mock returns valid non-empty text
	auto mockInferencer = std::make_unique<TrackingMockSTTInferencer_EmptyOutput>();
	mockInferencer->transcriptionResult.text         = "Hello world";
	mockInferencer->transcriptionResult.noSpeechProb = 0.05f;
	mockInferencer->transcriptionResult.avgLogprob   = -0.3f;

	STTNode::Config nodeConfig;
	nodeConfig.pubPort               = kPubPort;
	nodeConfig.audioSubPort          = kAudPort;
	nodeConfig.inputShmName          = kShmName;
	nodeConfig.pollTimeout           = std::chrono::milliseconds(20);
	nodeConfig.chunkSamples          = static_cast<std::size_t>(nSamples);
	nodeConfig.overlapSamples        = 0;
	nodeConfig.noSpeechProbThreshold = 0.6f;

	STTNode node(nodeConfig);
	node.setInferencer(std::move(mockInferencer));
	node.initialize();

	std::thread runThread([&node]{ node.run(); });

	// Send audio_chunk with retry until a transcription is received
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

	// Verify the published transcription
	ASSERT_FALSE(receivedTranscription.is_null())
		<< "No transcription message received — control test failed; "
		   "test infrastructure may be broken";
	EXPECT_EQ(receivedTranscription.value("type", std::string{}),
	          std::string("transcription"));
	EXPECT_EQ(receivedTranscription.value("text", std::string{}),
	          std::string("Hello world"));
}

