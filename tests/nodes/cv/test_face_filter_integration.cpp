#include <gtest/gtest.h>

#include "cv/face_filter_node.hpp"
#include "face_filter_inferencer.hpp"
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
#include <format>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// FaceFilterNode Integration Tests — CPU-only.
//
// What these tests catch:
//   - Node processes video_frame messages but never publishes filtered_frame
//   - Toggle enable/disable doesn't change the filtered flag
//   - select_filter command doesn't update the active filter ID
//   - Inferencer errors crash the node instead of being handled gracefully
//
// Port offset: +20000 (avoids collisions with production and other tests)
// ---------------------------------------------------------------------------


static constexpr uint32_t    kTestFrameW  = 64;
static constexpr uint32_t    kTestFrameH  = 48;
static constexpr std::size_t kTestBgrBytes =
	static_cast<std::size_t>(kTestFrameW) * kTestFrameH * kBgr24BytesPerPixel;

static const std::size_t kTestShmSize =
	ShmCircularBuffer<ShmVideoHeader>::segmentSize(
		kCircularBufferSlotCount, kMaxBgr24FrameBytes);

// ---------------------------------------------------------------------------
// StubFaceFilterInferencer — deterministic passthrough for CPU-only testing.
// ---------------------------------------------------------------------------
class StubFaceFilterInferencer : public FaceFilterInferencer {
public:
	void loadModel(const std::string& /*onnxPath*/) override {
		modelLoaded_ = true;
	}

	void loadFilterAssets(const std::string& /*manifestPath*/) override {}

	void setActiveFilter(const std::string& filterId) override {
		activeFilterId_ = filterId;
		++setFilterCallCount_;
	}

	[[nodiscard]] tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* /*bgrFrame*/, uint32_t /*width*/,
	             uint32_t /*height*/, uint8_t* outBuf,
	             std::size_t maxJpegBytes) override
	{
		++processFrameCallCount_;
		if (forceError_) {
			return tl::unexpected(std::string("simulated failure"));
		}
		constexpr std::size_t kStubSize = 256;
		if (maxJpegBytes < kStubSize) {
			return tl::unexpected(std::string("buffer too small"));
		}
		std::memset(outBuf, 0xAB, kStubSize);
		return kStubSize;
	}

	[[nodiscard]] FaceMeshResult lastLandmarks() const noexcept override {
		return {};
	}

	[[nodiscard]] std::vector<FilterInfo> availableFilters() const override {
		return {{"dog", "Dog Filter", true}};
	}

	void unload() noexcept override { modelLoaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
		return 0;
	}

	bool forceError_{false};
	int  processFrameCallCount_{0};
	int  setFilterCallCount_{0};
	bool modelLoaded_{false};
	std::string activeFilterId_;
};

// ---------------------------------------------------------------------------
// RAII POSIX SHM helper
// ---------------------------------------------------------------------------
struct TestVideoShm {
	explicit TestVideoShm(const char* name) : name_(name)
	{
		shm_unlink(name_);
		const int fd = shm_open(name_, O_CREAT | O_RDWR | O_TRUNC, 0600);
		if (fd < 0) throw std::runtime_error(
			std::string("shm_open: ") + strerror(errno));
		if (ftruncate(fd, static_cast<off_t>(kTestShmSize)) != 0) {
			close(fd); throw std::runtime_error("ftruncate");
		}
		ptr_ = static_cast<uint8_t*>(
			mmap(nullptr, kTestShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
		close(fd);
		if (ptr_ == MAP_FAILED) throw std::runtime_error("mmap");
		std::memset(ptr_, 0, kTestShmSize);

		auto* hdr = reinterpret_cast<ShmVideoHeader*>(ptr_);
		hdr->width         = kTestFrameW;
		hdr->height        = kTestFrameH;
		hdr->bytesPerPixel = kBgr24BytesPerPixel;
	}

	~TestVideoShm()
	{
		if (ptr_ && ptr_ != MAP_FAILED) munmap(ptr_, kTestShmSize);
		shm_unlink(name_);
	}

	void writeBgrSlot0(uint8_t b, uint8_t g, uint8_t r)
	{
		auto* hdr = reinterpret_cast<ShmVideoHeader*>(ptr_);
		hdr->seqNumber   = ++seq_;
		hdr->timestampNs = static_cast<uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());

		auto* ctrl = reinterpret_cast<ShmCircularControl*>(
			ptr_ + ShmCircularBuffer<ShmVideoHeader>::kControlOffset);
		ctrl->slotCount    = kCircularBufferSlotCount;
		ctrl->slotByteSize = static_cast<uint32_t>(kMaxBgr24FrameBytes);
		ctrl->writePos.store(seq_, std::memory_order_release);

		uint8_t* slot = ptr_ + ShmCircularBuffer<ShmVideoHeader>::kSlotsOffset;
		for (std::size_t i = 0; i < kTestBgrBytes; i += 3) {
			slot[i] = b; slot[i+1] = g; slot[i+2] = r;
		}
	}

	uint8_t*    ptr_{nullptr};
	const char* name_;
	uint64_t    seq_{0};
};

[[nodiscard]] static nlohmann::json recvPub(zmq::socket_t& sub, int ms = 1000)
{
	return receiveTestMessage(sub, ms);
}

// ===========================================================================
// Test 1: Passthrough — filter disabled, frame still published
// Bug caught: node silently drops frames when filter is off
// ===========================================================================
TEST(FaceFilterIntegration, PassthroughWhenDisabled)
{
	static constexpr int kPub = 25800, kVid = 25801, kWs = 25802;
	static const char* kShm = "/oe_test_ff_pass";

	TestVideoShm shm(kShm);
	shm.writeBgrSlot0(100, 150, 200);

	zmq::context_t ctx(1);
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("filtered_frame"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

	zmq::socket_t vidPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
	vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

	auto stub = std::make_unique<StubFaceFilterInferencer>();
	StubFaceFilterInferencer* sp = stub.get();

	FaceFilterNode::Config cfg;
	cfg.pubPort = kPub; cfg.videoSubPort = kVid; cfg.wsBridgeSubPort = kWs;
	cfg.inputShmName = kShm;
	cfg.pollTimeout = std::chrono::milliseconds(20);
	cfg.enabledAtStartup = false;

	FaceFilterNode node(cfg);
	node.setInferencer(std::move(stub));
	node.initialize();

	std::thread t([&node]{ node.run(); });

	const std::string msg = "video_frame " + nlohmann::json{
		{"slot",0},{"width",kTestFrameW},{"height",kTestFrameH}}.dump();

	nlohmann::json rx;
	auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (std::chrono::steady_clock::now() < dl) {
		vidPub.send(zmq::buffer(msg), zmq::send_flags::none);
		rx = recvPub(sub, 200);
		if (!rx.is_null()) break;
	}

	node.stop(); t.join();

	ASSERT_FALSE(rx.is_null()) << "No filtered_frame published";
	EXPECT_EQ(rx.value("type", std::string{}), std::string("filtered_frame"));
	EXPECT_FALSE(rx.value("filtered", true))
		<< "filtered must be false when disabled";
	EXPECT_GT(rx.value("size", int64_t{0}), int64_t{0});
	EXPECT_GT(sp->processFrameCallCount_, 0);
}

// ===========================================================================
// Test 2: Toggle — verify filtered flag changes after toggle command
// Bug caught: toggle_face_filter command ignored by node
// ===========================================================================
TEST(FaceFilterIntegration, ToggleEnable)
{
	static constexpr int kPub = 25810, kVid = 25811, kWs = 25812;
	static const char* kShm = "/oe_test_ff_toggle";

	TestVideoShm shm(kShm);
	shm.writeBgrSlot0(80, 80, 80);

	zmq::context_t ctx(1);
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("filtered_frame"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

	zmq::socket_t vidPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
	vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

	zmq::socket_t wsPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
	wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

	FaceFilterNode::Config cfg;
	cfg.pubPort = kPub; cfg.videoSubPort = kVid; cfg.wsBridgeSubPort = kWs;
	cfg.inputShmName = kShm;
	cfg.pollTimeout = std::chrono::milliseconds(20);
	cfg.enabledAtStartup = false;

	FaceFilterNode node(cfg);
	node.setInferencer(std::make_unique<StubFaceFilterInferencer>());
	node.initialize();
	EXPECT_FALSE(node.isFilterEnabled());

	std::thread t([&node]{ node.run(); });

	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Toggle ON
	const std::string toggle = "ui_command " +
		nlohmann::json{{"action","toggle_face_filter"}}.dump();
	wsPub.send(zmq::buffer(toggle), zmq::send_flags::none);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	const std::string frame = "video_frame " + nlohmann::json{
		{"slot",0},{"width",kTestFrameW},{"height",kTestFrameH}}.dump();

	nlohmann::json rx;
	auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (std::chrono::steady_clock::now() < dl) {
		vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
		rx = recvPub(sub, 200);
		if (!rx.is_null()) break;
	}

	node.stop(); t.join();

	ASSERT_FALSE(rx.is_null()) << "No message after toggle";
	EXPECT_TRUE(rx.value("filtered", false))
		<< "filtered must be true after toggle_face_filter";
}

// ===========================================================================
// Test 3: Select filter — verify filter_id propagated to inferencer + output
// Bug caught: select_filter command doesn't reach inferencer or output JSON
// ===========================================================================
TEST(FaceFilterIntegration, SelectFilter)
{
	static constexpr int kPub = 25820, kVid = 25821, kWs = 25822;
	static const char* kShm = "/oe_test_ff_select";

	TestVideoShm shm(kShm);
	shm.writeBgrSlot0(120, 120, 120);

	zmq::context_t ctx(1);
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("filtered_frame"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

	zmq::socket_t vidPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
	vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

	zmq::socket_t wsPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
	wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

	auto stub = std::make_unique<StubFaceFilterInferencer>();
	StubFaceFilterInferencer* sp = stub.get();

	FaceFilterNode::Config cfg;
	cfg.pubPort = kPub; cfg.videoSubPort = kVid; cfg.wsBridgeSubPort = kWs;
	cfg.inputShmName = kShm;
	cfg.pollTimeout = std::chrono::milliseconds(20);
	cfg.enabledAtStartup = true;

	FaceFilterNode node(cfg);
	node.setInferencer(std::move(stub));
	node.initialize();

	std::thread t([&node]{ node.run(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Select "dog" filter
	const std::string sel = "ui_command " +
		nlohmann::json{{"action","select_filter"},{"filter_id","dog"}}.dump();
	wsPub.send(zmq::buffer(sel), zmq::send_flags::none);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	const std::string frame = "video_frame " + nlohmann::json{
		{"slot",0},{"width",kTestFrameW},{"height",kTestFrameH}}.dump();

	nlohmann::json rx;
	auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (std::chrono::steady_clock::now() < dl) {
		vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
		rx = recvPub(sub, 200);
		if (!rx.is_null()) break;
	}

	node.stop(); t.join();

	ASSERT_FALSE(rx.is_null()) << "No message after select_filter";
	EXPECT_EQ(rx.value("filter_id", std::string{}), std::string("dog"));
	EXPECT_EQ(sp->activeFilterId_, std::string("dog"));
	EXPECT_GE(sp->setFilterCallCount_, 1);
}

// ===========================================================================
// Test 4: Inferencer error — no crash, no publish
// Bug caught: inferencer exception propagates and kills the node
// ===========================================================================
TEST(FaceFilterIntegration, InferencerError_NoCrash)
{
	static constexpr int kPub = 25830, kVid = 25831, kWs = 25832;
	static const char* kShm = "/oe_test_ff_err";

	TestVideoShm shm(kShm);
	shm.writeBgrSlot0(60, 60, 60);

	zmq::context_t ctx(1);
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("filtered_frame"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

	zmq::socket_t vidPub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
	vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

	auto stub = std::make_unique<StubFaceFilterInferencer>();
	stub->forceError_ = true;

	FaceFilterNode::Config cfg;
	cfg.pubPort = kPub; cfg.videoSubPort = kVid; cfg.wsBridgeSubPort = kWs;
	cfg.inputShmName = kShm;
	cfg.pollTimeout = std::chrono::milliseconds(20);

	FaceFilterNode node(cfg);
	node.setInferencer(std::move(stub));
	node.initialize();

	std::thread t([&node]{ node.run(); });

	const std::string frame = "video_frame " + nlohmann::json{
		{"slot",0},{"width",kTestFrameW},{"height",kTestFrameH}}.dump();

	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	for (int i = 0; i < 5; ++i) {
		vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	nlohmann::json rx = recvPub(sub, 500);
	node.stop(); t.join();

	EXPECT_TRUE(rx.is_null())
		<< "No filtered_frame when inferencer errors";
}

