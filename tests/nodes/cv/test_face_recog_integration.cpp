#include <gtest/gtest.h>

#include "cv/face_recog_node.hpp"
#include "face_recog_inferencer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/constants/video_constants.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// FaceRecognitionNode Integration Tests — CPU-only.
//
// Purpose: Verify the full detect -> embed -> match -> publish cycle through
// FaceRecognitionNode with a stub inferencer and real ZMQ transport.
//
// What these tests catch:
//   - Node processes video_frame messages but never publishes identity
//   - Frame subsampling logic has off-by-one errors
//   - Inferencer errors crash the node instead of being handled gracefully
//   - No-face-detected scenario still publishes spurious identity messages
//
// Port offset: +20000 (avoids collisions with production and other tests)
//
// SHM layout mirrors the production /oe.vid.ingest circular buffer:
//   [ShmVideoHeader (64 B)] [ShmCircularControl (128 B)] [slot0 BGR] ... [slotN-1 BGR]
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Test-scoped frame geometry — small to keep SHM segments tiny
// ---------------------------------------------------------------------------

static constexpr uint32_t    kTestFrameW      = 64;
static constexpr uint32_t    kTestFrameH      = 48;
static constexpr std::size_t kTestBgrBytes     =
	static_cast<std::size_t>(kTestFrameW) * kTestFrameH * kBgr24BytesPerPixel;

// SHM layout sizes — must match what FaceRecognitionNode expects.
// Node always allocates for max resolution (1920x1080), not test frame size.
static const std::size_t kTestShmSize =
	ShmCircularBuffer<ShmVideoHeader>::segmentSize(
		kCircularBufferSlotCount, kMaxBgr24FrameBytes);

// ---------------------------------------------------------------------------
// StubFaceRecogInferencer — deterministic embeddings for CPU-only testing.
//
// Returns configurable face detections. The embedding direction is derived
// from the first pixel's RGB channels, making the output deterministic and
// input-dependent.
// ---------------------------------------------------------------------------
class StubFaceRecogInferencer : public FaceRecogInferencer {
public:
	void loadModel(const std::string& /*modelPackPath*/) override {
		modelLoaded_ = true;
	}

	[[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
	detect(const uint8_t* bgrInputFrame,
	       uint32_t       /*frameWidth*/,
	       uint32_t       /*frameHeight*/) override
	{
		++detectCallCount_;

		if (forceError_) {
			return tl::unexpected(
				std::string("simulated face detection failure"));
		}
		if (!shouldDetectFace_) {
			return std::vector<FaceDetection>{};
		}

		FaceDetection face;
		face.bbox = {10, 10, 40, 50};
		face.landmarks = {};
		face.embedding.assign(512, 0.0f);

		// Derive embedding from the first pixel's BGR channels.
		// This ensures different input images produce different embeddings.
		if (bgrInputFrame != nullptr) {
			face.embedding[0] = static_cast<float>(bgrInputFrame[0]) / 255.0f;  // B
			face.embedding[1] = static_cast<float>(bgrInputFrame[1]) / 255.0f;  // G
			face.embedding[2] = static_cast<float>(bgrInputFrame[2]) / 255.0f;  // R
		} else {
			face.embedding[0] = 1.0f;
		}

		return std::vector<FaceDetection>{face};
	}

	void unload() noexcept override {}

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
		return 0;
	}

	// --- Configuration ---
	bool shouldDetectFace_{true};
	bool forceError_{false};
	int  detectCallCount_{0};
	bool modelLoaded_{false};
};

// ---------------------------------------------------------------------------
// POSIX SHM helper — RAII for test SHM segments
// ---------------------------------------------------------------------------
struct TestVideoShm {
	explicit TestVideoShm(const char* segmentName) : segmentName_(segmentName)
	{
		// Unlink any stale segment from a previous failed test run.
		shm_unlink(segmentName_);

		const int fd = shm_open(segmentName_, O_CREAT | O_RDWR | O_TRUNC, 0600);
		if (fd < 0) throw std::runtime_error(
			std::string("shm_open failed: ") + strerror(errno));
		if (ftruncate(fd, static_cast<off_t>(kTestShmSize)) != 0) {
			close(fd);
			throw std::runtime_error("ftruncate failed");
		}
		mappedRegion_ = static_cast<uint8_t*>(
			mmap(nullptr, kTestShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
		close(fd);
		if (mappedRegion_ == MAP_FAILED) throw std::runtime_error("mmap failed");
		std::memset(mappedRegion_, 0, kTestShmSize);

		// Initialize the ShmVideoHeader with test frame dimensions
		auto* header = reinterpret_cast<ShmVideoHeader*>(mappedRegion_);
		header->width         = kTestFrameW;
		header->height        = kTestFrameH;
		header->bytesPerPixel = kBgr24BytesPerPixel;
	}

	~TestVideoShm()
	{
		if (mappedRegion_ && mappedRegion_ != MAP_FAILED) {
			munmap(mappedRegion_, kTestShmSize);
		}
		shm_unlink(segmentName_);
	}

	/// Write a solid-colour BGR frame into slot 0 and set a valid seq number.
	void writeBgrSlot0(uint8_t b, uint8_t g, uint8_t r)
	{
		auto* header = reinterpret_cast<ShmVideoHeader*>(mappedRegion_);
		header->seqNumber   = ++writeSeq_;
		header->timestampNs = static_cast<uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());

		// Set writePos in ShmCircularControl — commit one write so consumer sees slot 0.
		auto* ctrl = reinterpret_cast<ShmCircularControl*>(
			mappedRegion_ + ShmCircularBuffer<ShmVideoHeader>::kControlOffset);
		ctrl->slotCount = kCircularBufferSlotCount;
		ctrl->slotByteSize = static_cast<uint32_t>(kMaxBgr24FrameBytes);
		ctrl->writePos.store(writeSeq_, std::memory_order_release);

		// Fill slot 0 with solid BGR
		uint8_t* slotBase = mappedRegion_
			+ ShmCircularBuffer<ShmVideoHeader>::kSlotsOffset;
		for (std::size_t i = 0; i < kTestBgrBytes; i += 3) {
			slotBase[i]     = b;
			slotBase[i + 1] = g;
			slotBase[i + 2] = r;
		}
	}

	uint8_t*    mappedRegion_{nullptr};
	const char* segmentName_;
	uint64_t    writeSeq_{0};
};

// ---------------------------------------------------------------------------
// ZMQ helper — receive a published JSON message on a SUB socket.
// ---------------------------------------------------------------------------
[[nodiscard]] static nlohmann::json recvPubFrame(zmq::socket_t& sub,
                                                  int timeoutMs = 1000)
{
	return receiveTestMessage(sub, timeoutMs);
}

// ===========================================================================
// Test 1: Detect and Publish
//
// Configure stub to detect one face with a known embedding.  Run the node,
// send a video_frame trigger, verify that an "identity" ZMQ message is
// published with face_detected=true.
// ===========================================================================

TEST(FaceRecogIntegration, DetectAndPublish)
{
	static constexpr int  kPubPort   = 25700;
	static constexpr int  kVidPort   = 25701;
	static constexpr int  kWsPort    = 25702;
	static const char*    kShmName   = "/oe_test_face_recog_integ";

	// 1 -- Create SHM and write a known BGR frame into slot 0
	TestVideoShm shm(kShmName);
	shm.writeBgrSlot0(100, 150, 200);

	// 2 -- Subscribe to "identity" on the node's PUB port BEFORE init
	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("identity"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPubPort));

	// 3 -- Bind a PUB socket that impersonates VideoIngestNode
	zmq::socket_t videoPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVidPort));
	videoPub.bind(std::format("ipc:///tmp/omniedge_{}",kVidPort));

	// 4 -- Configure and initialise the node with a stub inferencer
	auto stubInferencer = std::make_unique<StubFaceRecogInferencer>();
	StubFaceRecogInferencer* stubPtr = stubInferencer.get();

	// Pre-register a known face so the node can match against it.
	// We will register via the node API after initialize().

	FaceRecognitionNode::Config cfg;
	cfg.pubPort         = kPubPort;
	cfg.videoSubPort    = kVidPort;
	cfg.wsBridgeSubPort = kWsPort;
	cfg.inputShmName    = kShmName;
	cfg.knownFacesDb    = ":memory:";
	cfg.pollTimeout     = std::chrono::milliseconds(20);
	cfg.frameSubsample  = 1;  // process every frame for this test
	cfg.recognitionThreshold = 0.4f;

	FaceRecognitionNode node(cfg);
	node.setInferencer(std::move(stubInferencer));
	node.initialize();

	// Register a face using the same pixel values we will detect.
	// The stub produces an embedding derived from pixel (100, 150, 200).
	std::vector<uint8_t> regFrame(kTestBgrBytes);
	for (std::size_t i = 0; i < kTestBgrBytes; i += 3) {
		regFrame[i]     = 100;
		regFrame[i + 1] = 150;
		regFrame[i + 2] = 200;
	}
	bool registered = node.registerFace("TestUser", regFrame.data(),
	                                     kTestFrameW, kTestFrameH);
	ASSERT_TRUE(registered) << "registerFace must succeed with a valid frame";

	// 5 -- Run the node in a background thread
	std::thread runThread([&node]{ node.run(); });

	// 6 -- Send video_frame with retry until an identity message arrives
	const nlohmann::json videoMeta = {
		{"slot", 0},
		{"width", kTestFrameW},
		{"height", kTestFrameH},
	};
	const std::string videoFrame = "video_frame " + videoMeta.dump();

	nlohmann::json receivedIdentity;
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(5);

	while (std::chrono::steady_clock::now() < deadline) {
		videoPub.send(zmq::buffer(videoFrame), zmq::send_flags::none);
		receivedIdentity = recvPubFrame(sub, 200);
		if (!receivedIdentity.is_null()) { break; }
	}

	node.stop();
	runThread.join();

	// 7 -- Verify the published identity message
	ASSERT_FALSE(receivedIdentity.is_null())
		<< "No identity message received — node may not be processing frames";
	EXPECT_EQ(receivedIdentity.value("type", std::string{}), std::string("identity"))
		<< "Published message must have type=identity";
	EXPECT_EQ(receivedIdentity.value("name", std::string{}), std::string("TestUser"))
		<< "Identified face name must match the registered face";
	EXPECT_GE(receivedIdentity.value("confidence", 0.0f), 0.4f)
		<< "Confidence must meet the recognition threshold";
	EXPECT_GT(stubPtr->detectCallCount_, 0)
		<< "Inferencer detect() must have been called at least once";
}

// ===========================================================================
// Test 2: No Face Detected -> No Identity Published
//
// Configure stub to return empty detections.  Verify that no identity
// message is published.
// ===========================================================================

TEST(FaceRecogIntegration, NoFaceDetected_NoIdentityPublished)
{
	static constexpr int  kPubPort   = 25710;
	static constexpr int  kVidPort   = 25711;
	static constexpr int  kWsPort    = 25712;
	static const char*    kShmName   = "/oe_test_face_recog_noface";

	TestVideoShm shm(kShmName);
	shm.writeBgrSlot0(50, 50, 50);

	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("identity"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPubPort));

	zmq::socket_t videoPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVidPort));
	videoPub.bind(std::format("ipc:///tmp/omniedge_{}",kVidPort));

	// Stub configured to return NO faces
	auto stubInferencer = std::make_unique<StubFaceRecogInferencer>();
	stubInferencer->shouldDetectFace_ = false;

	FaceRecognitionNode::Config cfg;
	cfg.pubPort         = kPubPort;
	cfg.videoSubPort    = kVidPort;
	cfg.wsBridgeSubPort = kWsPort;
	cfg.inputShmName    = kShmName;
	cfg.knownFacesDb    = ":memory:";
	cfg.pollTimeout     = std::chrono::milliseconds(20);
	cfg.frameSubsample  = 1;

	FaceRecognitionNode node(cfg);
	node.setInferencer(std::move(stubInferencer));
	node.initialize();

	std::thread runThread([&node]{ node.run(); });

	const nlohmann::json videoMeta = {
		{"slot", 0},
		{"width", kTestFrameW},
		{"height", kTestFrameH},
	};
	const std::string videoFrame = "video_frame " + videoMeta.dump();

	// Send video_frame messages several times to ensure the node processes them
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	for (int i = 0; i < 5; ++i) {
		videoPub.send(zmq::buffer(videoFrame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	// Wait up to 500 ms for any identity message (should NOT arrive)
	const nlohmann::json received = recvPubFrame(sub, 500);

	node.stop();
	runThread.join();

	EXPECT_TRUE(received.is_null())
		<< "No identity message should be published when no face is detected";
}

// ===========================================================================
// Test 3: Inferencer Error -> No Publish, No Crash
//
// Configure stub to return an error from detect().  Verify the node does
// not crash, does not publish, and can be stopped cleanly.
// ===========================================================================

TEST(FaceRecogIntegration, InferencerError_NoPublish)
{
	static constexpr int  kPubPort   = 25720;
	static constexpr int  kVidPort   = 25721;
	static constexpr int  kWsPort    = 25722;
	static const char*    kShmName   = "/oe_test_face_recog_err";

	TestVideoShm shm(kShmName);
	shm.writeBgrSlot0(80, 80, 80);

	zmq::context_t testCtx(1);
	zmq::socket_t sub(testCtx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("identity"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPubPort));

	zmq::socket_t videoPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVidPort));
	videoPub.bind(std::format("ipc:///tmp/omniedge_{}",kVidPort));

	// Stub configured to FAIL on detect()
	auto stubInferencer = std::make_unique<StubFaceRecogInferencer>();
	stubInferencer->forceError_ = true;

	FaceRecognitionNode::Config cfg;
	cfg.pubPort         = kPubPort;
	cfg.videoSubPort    = kVidPort;
	cfg.wsBridgeSubPort = kWsPort;
	cfg.inputShmName    = kShmName;
	cfg.knownFacesDb    = ":memory:";
	cfg.pollTimeout     = std::chrono::milliseconds(20);
	cfg.frameSubsample  = 1;

	FaceRecognitionNode node(cfg);
	node.setInferencer(std::move(stubInferencer));
	node.initialize();

	std::thread runThread([&node]{ node.run(); });

	const nlohmann::json videoMeta = {
		{"slot", 0},
		{"width", kTestFrameW},
		{"height", kTestFrameH},
	};
	const std::string videoFrame = "video_frame " + videoMeta.dump();

	// Send video_frame messages — triggers detect() which will error
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	for (int i = 0; i < 5; ++i) {
		videoPub.send(zmq::buffer(videoFrame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	// No identity should be published (inferencer errored)
	const nlohmann::json received = recvPubFrame(sub, 500);

	// Node must still be alive — stop cleanly
	node.stop();
	runThread.join();

	EXPECT_TRUE(received.is_null())
		<< "No identity should be published when inferencer returns error";
}

// ===========================================================================
// Test 4: Frame Subsampling
//
// Set frameSubsample=3, send 4 video_frame messages.  Verify detect() is
// called only once (on the frame where frameCounter % subsample == 0).
// ===========================================================================

TEST(FaceRecogIntegration, FrameSubsampling)
{
	static constexpr int  kPubPort   = 25730;
	static constexpr int  kVidPort   = 25731;
	static constexpr int  kWsPort    = 25732;
	static const char*    kShmName   = "/oe_test_face_recog_sub";

	TestVideoShm shm(kShmName);
	shm.writeBgrSlot0(120, 120, 120);

	zmq::context_t testCtx(1);

	zmq::socket_t videoPub(testCtx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVidPort));
	videoPub.bind(std::format("ipc:///tmp/omniedge_{}",kVidPort));

	// Stub that detects faces but we only care about call count
	auto stubInferencer = std::make_unique<StubFaceRecogInferencer>();
	StubFaceRecogInferencer* stubPtr = stubInferencer.get();

	FaceRecognitionNode::Config cfg;
	cfg.pubPort         = kPubPort;
	cfg.videoSubPort    = kVidPort;
	cfg.wsBridgeSubPort = kWsPort;
	cfg.inputShmName    = kShmName;
	cfg.knownFacesDb    = ":memory:";
	cfg.pollTimeout     = std::chrono::milliseconds(20);
	cfg.frameSubsample  = 3;  // detect every 3rd frame

	FaceRecognitionNode node(cfg);
	node.setInferencer(std::move(stubInferencer));
	node.initialize();

	std::thread runThread([&node]{ node.run(); });

	const nlohmann::json videoMeta = {
		{"slot", 0},
		{"width", kTestFrameW},
		{"height", kTestFrameH},
	};
	const std::string videoFrame = "video_frame " + videoMeta.dump();

	// Brief delay so ZMQ subscriptions propagate
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// Send exactly 4 video_frame messages with enough delay between them
	// to ensure the node processes each one individually.
	for (int i = 0; i < 4; ++i) {
		videoPub.send(zmq::buffer(videoFrame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
	}

	// Allow the node to finish processing the last message
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	node.stop();
	runThread.join();

	// With frameSubsample=3 and counter starting at 0:
	//   frame 0: counter=1, 1%3!=0 -> skip
	//   frame 1: counter=2, 2%3!=0 -> skip
	//   frame 2: counter=3, 3%3==0 -> DETECT
	//   frame 3: counter=4, 4%3!=0 -> skip
	// => detect() called exactly once
	EXPECT_EQ(stubPtr->detectCallCount_, 1)
		<< "With frameSubsample=3, sending 4 frames should trigger detect() "
		   "exactly once (on the 3rd frame)";
}

