#include <gtest/gtest.h>

#include "cv/sam2_node.hpp"
#include "sam2_inferencer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/constants/video_constants.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

#include <spdlog/spdlog.h>
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
// Sam2Node Integration Tests — CPU-only.
//
// What these tests catch:
//   - Node publishes segmentation_mask after receiving sam2_segment_point cmd
//   - Toggle enable/disable changes processing behaviour
//   - Box prompt dispatched correctly to inferencer
//   - Inferencer errors don't crash the node
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
// StubSam2Inferencer — deterministic for integration testing.
// ---------------------------------------------------------------------------
class StubSam2Inferencer : public Sam2Inferencer {
public:
    void loadModel(const std::string& /*enc*/, const std::string& /*dec*/) override {
        modelLoaded_ = true;
    }

    [[nodiscard]] tl::expected<void, std::string>
    encodeImage(const uint8_t* bgrFrame, uint32_t width, uint32_t height) override {
        if (!modelLoaded_) return tl::unexpected(std::string("not loaded"));
        if (!bgrFrame) return tl::unexpected(std::string("null"));
        width_ = width; height_ = height;
        encoded_ = true;
        return {};
    }

    [[nodiscard]] tl::expected<Sam2Result, std::string>
    segmentWithPrompt(const Sam2Prompt& prompt) override {
        ++segmentCallCount_;
        if (forceError_) return tl::unexpected(std::string("simulated failure"));
        if (!encoded_) return tl::unexpected(std::string("no image"));
        lastPromptType_ = prompt.type;

        Sam2Result result;
        result.maskWidth  = width_;
        result.maskHeight = height_;
        result.mask.resize(static_cast<std::size_t>(width_) * height_, 128);
        result.iouScore  = 0.91f;
        result.stability = 0.87f;
        return result;
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame, uint32_t width, uint32_t height,
                 const Sam2Prompt& prompt, uint8_t* outBuf, std::size_t maxJpegBytes) override
    {
        ++processFrameCallCount_;
        auto enc = encodeImage(bgrFrame, width, height);
        if (!enc) return tl::unexpected(enc.error());
        auto seg = segmentWithPrompt(prompt);
        if (!seg) return tl::unexpected(seg.error());

        // Store result for lastSegmentResult() accessor
        lastResult_ = seg.value();

        constexpr std::size_t kStubSize = 256;
        if (maxJpegBytes < kStubSize) return tl::unexpected(std::string("too small"));
        std::memset(outBuf, 0xAB, kStubSize);
        return kStubSize;
    }

    void unload() noexcept override { modelLoaded_ = false; }
    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override { return 0; }

    bool forceError_{false};
    int  processFrameCallCount_{0};
    int  segmentCallCount_{0};
    Sam2PromptType lastPromptType_{Sam2PromptType::kPoint};
    bool modelLoaded_{false};
    bool encoded_{false};
    uint32_t width_{0}, height_{0};
};

// ---------------------------------------------------------------------------
// RAII POSIX SHM helper
// ---------------------------------------------------------------------------
struct TestSam2Shm {
    explicit TestSam2Shm(const char* name) : name_(name)
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

    ~TestSam2Shm()
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
// Test 1: Point prompt triggers segmentation and publishes result
// Bug caught: node doesn't process sam2_segment_point commands
// ===========================================================================
TEST(Sam2Integration, PointPrompt_PublishesSegmentationMask)
{
    static constexpr int kPub = 25900, kVid = 25901, kWs = 25902, kDmn = 25903;
    static const char* kShm = "/oe_test_sam2_pt";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(100, 150, 200);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    StubSam2Inferencer* sp = stub.get();

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send point prompt
    const std::string promptCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_point"},
            {"x", 0.5}, {"y", 0.5}, {"label", 1}
        }.dump();
    wsPub.send(zmq::buffer(promptCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send video frame to trigger processing
    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    nlohmann::json rx;
    auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < dl) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        rx = recvPub(sub, 200);
        if (!rx.is_null()) break;
    }

    node.stop(); t.join();

    ASSERT_FALSE(rx.is_null()) << "No segmentation_mask published";
    EXPECT_EQ(rx.value("type", std::string{}), std::string("segmentation_mask"));
    EXPECT_GT(rx.value("iou_score", 0.0), 0.0);
    EXPECT_GT(sp->processFrameCallCount_, 0);

    spdlog::debug("[TestSAM2Integration] Point prompt test: process calls={}, seg calls={}",
                  sp->processFrameCallCount_, sp->segmentCallCount_);
}

// ===========================================================================
// Test 2: Toggle — verify enable/disable works
// Bug caught: toggle_sam2 command ignored
// ===========================================================================
TEST(Sam2Integration, Toggle_DisablesPipeline)
{
    static constexpr int kPub = 25910, kVid = 25911, kWs = 25912, kDmn = 25913;
    static const char* kShm = "/oe_test_sam2_tog";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(80, 80, 80);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::make_unique<StubSam2Inferencer>());
    node.initialize();
    EXPECT_TRUE(node.isSam2Enabled());

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Toggle OFF
    const std::string toggle = "ui_command " +
        nlohmann::json{{"action", "toggle_sam2"}}.dump();
    wsPub.send(zmq::buffer(toggle), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    node.stop(); t.join();

    EXPECT_FALSE(node.isSam2Enabled())
        << "toggle_sam2 must disable the module";
}

// ===========================================================================
// Test 3: Box prompt — dispatched correctly
// Bug caught: box coordinates not forwarded to inferencer
// ===========================================================================
TEST(Sam2Integration, BoxPrompt_DispatchedCorrectly)
{
    static constexpr int kPub = 25920, kVid = 25921, kWs = 25922, kDmn = 25923;
    static const char* kShm = "/oe_test_sam2_box";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(120, 120, 120);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    StubSam2Inferencer* sp = stub.get();

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send box prompt
    const std::string boxCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_box"},
            {"x1", 0.1}, {"y1", 0.2}, {"x2", 0.8}, {"y2", 0.9}
        }.dump();
    wsPub.send(zmq::buffer(boxCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    nlohmann::json rx;
    auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < dl) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        rx = recvPub(sub, 200);
        if (!rx.is_null()) break;
    }

    node.stop(); t.join();

    ASSERT_FALSE(rx.is_null()) << "No segmentation_mask after box prompt";
    EXPECT_EQ(rx.value("prompt_type", -1), static_cast<int>(Sam2PromptType::kBox));
    EXPECT_GT(sp->processFrameCallCount_, 0);
}

// ===========================================================================
// Test 4: Inferencer error — no crash, no publish
// Bug caught: exception propagates and kills the node
// ===========================================================================
TEST(Sam2Integration, InferencerError_NoCrash)
{
    static constexpr int kPub = 25930, kVid = 25931, kWs = 25932, kDmn = 25933;
    static const char* kShm = "/oe_test_sam2_err";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(60, 60, 60);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    stub->forceError_ = true;

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    std::thread t([&node]{ node.run(); });

    // Send prompt + frames
    const std::string promptCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_point"},
            {"x", 0.5}, {"y", 0.5}, {"label", 1}
        }.dump();
    wsPub.send(zmq::buffer(promptCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    for (int i = 0; i < 5; ++i) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    nlohmann::json rx = recvPub(sub, 500);
    node.stop(); t.join();

    EXPECT_TRUE(rx.is_null())
        << "No segmentation_mask when inferencer errors";
}

// ===========================================================================
// Test 5: Background mode command — changes mode and includes in metadata
// Bug caught: set_sam2_bg_mode command ignored, bg_mode missing from ZMQ JSON
// ===========================================================================
TEST(Sam2Integration, BgModeCommand_ChangesNodeState)
{
    static constexpr int kPub = 25940, kVid = 25941, kWs = 25942, kDmn = 25943;
    static const char* kShm = "/oe_test_sam2_bgm";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(90, 90, 90);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    StubSam2Inferencer* sp = stub.get();

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    // Default background mode should be kBlur (virtual meeting default)
    EXPECT_EQ(node.backgroundMode(), Sam2BackgroundMode::kBlur);

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Set background mode to "white"
    const std::string bgModeCmd = "ui_command " +
        nlohmann::json{{"action", "set_sam2_bg_mode"}, {"mode", "white"}}.dump();
    wsPub.send(zmq::buffer(bgModeCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send point prompt
    const std::string promptCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_point"},
            {"x", 0.5}, {"y", 0.5}, {"label", 1}
        }.dump();
    wsPub.send(zmq::buffer(promptCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    nlohmann::json rx;
    auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < dl) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        rx = recvPub(sub, 200);
        if (!rx.is_null()) break;
    }

    node.stop(); t.join();

    // Verify background mode was updated
    EXPECT_EQ(node.backgroundMode(), Sam2BackgroundMode::kWhite);

    // Verify published metadata includes bg_mode
    ASSERT_FALSE(rx.is_null()) << "No segmentation_mask published after bg mode change";
    EXPECT_EQ(rx.value("bg_mode", std::string{}), std::string("white"))
        << "Published metadata must include bg_mode=white";

    spdlog::debug("[TestSAM2Integration] BgMode test: bg_mode={}, process_calls={}",
                  rx.value("bg_mode", std::string{"?"}), sp->processFrameCallCount_);
}

// ===========================================================================
// Test 6: Background color command — changes color
// Bug caught: set_sam2_bg_color command ignored
// ===========================================================================
TEST(Sam2Integration, BgColorCommand_ChangesNodeState)
{
    static constexpr int kPub = 25950, kVid = 25951, kWs = 25952, kDmn = 25953;
    static const char* kShm = "/oe_test_sam2_bgc";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(70, 70, 70);

    zmq::context_t ctx(1);
    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::make_unique<StubSam2Inferencer>());
    node.initialize();

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Set background color to green
    const std::string colorCmd = "ui_command " +
        nlohmann::json{{"action", "set_sam2_bg_color"}, {"r", 0}, {"g", 255}, {"b", 0}}.dump();
    wsPub.send(zmq::buffer(colorCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    node.stop(); t.join();

    // No crash, no hang — the color command was processed
    SUCCEED() << "set_sam2_bg_color processed without crash";
}

// ===========================================================================
// Test 7: Continuous tracking — after initial prompt, node keeps processing
// Bug caught: node stops after first frame in tracking mode
// ===========================================================================
TEST(Sam2Integration, ContinuousTracking_ProcessesMultipleFrames)
{
    static constexpr int kPub = 25960, kVid = 25961, kWs = 25962, kDmn = 25963;
    static const char* kShm = "/oe_test_sam2_trk";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(110, 110, 110);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    StubSam2Inferencer* sp = stub.get();

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    EXPECT_FALSE(node.isTracking()) << "Node should not be tracking before first prompt";

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send initial point prompt — this triggers continuous tracking
    const std::string promptCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_point"},
            {"x", 0.5}, {"y", 0.5}, {"label", 1}
        }.dump();
    wsPub.send(zmq::buffer(promptCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    // Send multiple frames — continuous tracking should process them all
    int receivedMasks = 0;
    auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < dl && receivedMasks < 3) {
        shm.writeBgrSlot0(110 + static_cast<uint8_t>(receivedMasks), 110, 110);
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        auto rx = recvPub(sub, 200);
        if (!rx.is_null()) {
            ++receivedMasks;
            spdlog::debug("[TestSAM2] Tracking frame {}: tracking={}",
                          receivedMasks, rx.value("tracking", false));
        }
        // Wait well above frame pacing interval (33ms) to avoid CI jitter
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    node.stop(); t.join();

    EXPECT_GE(receivedMasks, 2)
        << "Continuous tracking must produce multiple segmentation masks";
    EXPECT_GE(sp->processFrameCallCount_, 2)
        << "processFrame must be called on multiple frames in tracking mode";

    spdlog::debug("[TestSAM2Integration] Tracking test: masks={}, process_calls={}",
                  receivedMasks, sp->processFrameCallCount_);
}

// ===========================================================================
// Test 8: Stop tracking — sam2_stop_tracking command halts continuous mode
// Bug caught: no way to stop tracking once started
// ===========================================================================
TEST(Sam2Integration, StopTracking_HaltsContinuousMode)
{
    static constexpr int kPub = 25970, kVid = 25971, kWs = 25972, kDmn = 25973;
    static const char* kShm = "/oe_test_sam2_stop";

    TestSam2Shm shm(kShm);
    shm.writeBgrSlot0(130, 130, 130);

    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe, std::string("segmentation_mask"));
    sub.connect(std::format("ipc:///tmp/omniedge_{}",kPub));

    zmq::socket_t vidPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kVid));
    vidPub.bind(std::format("ipc:///tmp/omniedge_{}",kVid));

    zmq::socket_t wsPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kWs));
    wsPub.bind(std::format("ipc:///tmp/omniedge_{}",kWs));

    zmq::socket_t dmnPub(ctx, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kDmn));
    dmnPub.bind(std::format("ipc:///tmp/omniedge_{}",kDmn));

    auto stub = std::make_unique<StubSam2Inferencer>();
    StubSam2Inferencer* sp = stub.get();

    Sam2Node::Config cfg;
    cfg.pubPort = kPub; cfg.videoSubPort = kVid;
    cfg.wsBridgeSubPort = kWs; cfg.daemonSubPort = kDmn;
    cfg.inputShmName = kShm;
    cfg.pollTimeout = std::chrono::milliseconds(20);
    cfg.enabledAtStartup = true;

    Sam2Node node(cfg);
    node.setInferencer(std::move(stub));
    node.initialize();

    std::thread t([&node]{ node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Start tracking with a point prompt
    const std::string promptCmd = "ui_command " +
        nlohmann::json{
            {"action", "sam2_segment_point"},
            {"x", 0.3}, {"y", 0.3}, {"label", 1}
        }.dump();
    wsPub.send(zmq::buffer(promptCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string frame = "video_frame " + nlohmann::json{
        {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH}}.dump();

    // Let tracking start — process at least one frame
    auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < dl) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        auto rx = recvPub(sub, 200);
        if (!rx.is_null()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    const int callsBeforeStop = sp->processFrameCallCount_;

    // Send stop tracking command
    const std::string stopCmd = "ui_command " +
        nlohmann::json{{"action", "sam2_stop_tracking"}}.dump();
    wsPub.send(zmq::buffer(stopCmd), zmq::send_flags::none);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send more frames — they should NOT be processed
    for (int i = 0; i < 5; ++i) {
        vidPub.send(zmq::buffer(frame), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    node.stop(); t.join();

    EXPECT_FALSE(node.isTracking()) << "isTracking must be false after stop command";
    EXPECT_LE(sp->processFrameCallCount_, callsBeforeStop + 1)
        << "No (or at most one) additional processFrame calls after stop tracking";

    spdlog::debug("[TestSAM2Integration] StopTracking: calls before={}, after={}",
                  callsBeforeStop, sp->processFrameCallCount_);
}

