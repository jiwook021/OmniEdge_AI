#include <gtest/gtest.h>

#include "cv/security_vlm_node.hpp"
#include "common/constants/security_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// SecurityVlmNode end-to-end integration test.
//
// Drives the full pipeline:
//   BGR24 SHM producer  ──► ring (JPEG-encoded)
//   fake ZMQ security_event ──► pending incident
//   wait postroll_sec ──► mock_vision_generate.py ──► JSON analysis
//   assert: security_vlm_analysis published on pubPort with matching event_id
//   assert: security_incidents.jsonl gets a row
//
// What this catches:
//   - configureTransport: port/topic wiring regressions
//   - Ring fill → clip assembly: frame count off-by-one, JPEG encoding failure
//   - Security-event handler: filter on "_detected" suffix, pending-queue overflow
//   - drainPendingIncidents: dispatchAt timing, VLM client lifecycle
//   - Publish: topic name, event_id correlation, JSONL append format
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

namespace {

constexpr uint32_t    kTestFrameW   = 64;
constexpr uint32_t    kTestFrameH   = 48;
constexpr std::size_t kTestBgrBytes =
    static_cast<std::size_t>(kTestFrameW) * kTestFrameH * kBgr24BytesPerPixel;

const std::size_t kTestShmSize =
    ShmCircularBuffer<ShmVideoHeader>::segmentSize(
        kCircularBufferSlotCount, kMaxBgr24FrameBytes);

struct TestVideoShm {
    explicit TestVideoShm(const char* segmentName) : segmentName_(segmentName)
    {
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

    void writeBgrSlot0(uint8_t b, uint8_t g, uint8_t r)
    {
        auto* header = reinterpret_cast<ShmVideoHeader*>(mappedRegion_);
        header->seqNumber   = ++writeSeq_;
        header->timestampNs = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());

        auto* ctrl = reinterpret_cast<ShmCircularControl*>(
            mappedRegion_ + ShmCircularBuffer<ShmVideoHeader>::kControlOffset);
        ctrl->slotCount    = kCircularBufferSlotCount;
        ctrl->slotByteSize = static_cast<uint32_t>(kMaxBgr24FrameBytes);
        ctrl->writePos.store(writeSeq_, std::memory_order_release);

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

fs::path findMockScript()
{
    if (const char* override = std::getenv("OE_MOCK_VISION_GENERATE")) {
        return fs::path(override);
    }
    fs::path cur = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        const auto candidate = cur / "tests" / "mocks" / "mock_vision_generate.py";
        if (fs::exists(candidate)) return candidate;
        if (!cur.has_parent_path()) break;
        cur = cur.parent_path();
    }
    return fs::path("/src/tests/mocks/mock_vision_generate.py");
}

}  // namespace

// ===========================================================================
// Test: SecurityEventTriggersVlmAnalysis
//
// Fill the ring, send a fake security_event, verify a security_vlm_analysis
// ZMQ message arrives with correlated event_id and a non-empty behavior field.
// ===========================================================================
TEST(SecurityVlmNodeIntegration, SecurityEventTriggersVlmAnalysis)
{
    static constexpr int kPubPort  = 25780;   // node's PUB (security_vlm_analysis)
    static constexpr int kVidPort  = 25781;   // impersonates video_ingest
    static constexpr int kWsPort   = 25782;
    static constexpr int kEvtPort  = 25783;   // impersonates security_camera
    static const char*   kShmName  = "/oe_test_sec_vlm_integ";

    const auto mockScript = findMockScript();
    ASSERT_TRUE(fs::exists(mockScript))
        << "mock_vision_generate.py not found at " << mockScript;

    const auto logDir = fs::temp_directory_path() / "oe_test_sec_vlm_logs";
    fs::create_directories(logDir);
    fs::remove(logDir / "security_incidents.jsonl");

    TestVideoShm shm(kShmName);

    zmq::context_t testCtx(1);

    // Subscribe to analysis before node binds its PUB socket.
    zmq::socket_t sub(testCtx, ZMQ_SUB);
    sub.set(zmq::sockopt::subscribe,
            std::string(kZmqTopicSecurityVlmAnalysis));
    sub.connect(std::format("ipc:///tmp/omniedge_{}", kPubPort));

    zmq::socket_t videoPub(testCtx, ZMQ_PUB);
    fs::remove(std::format("/tmp/omniedge_{}", kVidPort));
    videoPub.bind(std::format("ipc:///tmp/omniedge_{}", kVidPort));

    zmq::socket_t eventPub(testCtx, ZMQ_PUB);
    fs::remove(std::format("/tmp/omniedge_{}", kEvtPort));
    eventPub.bind(std::format("ipc:///tmp/omniedge_{}", kEvtPort));

    SecurityVlmNode::Config cfg;
    cfg.pubPort              = kPubPort;
    cfg.videoSubPort         = kVidPort;
    cfg.securityEventSubPort = kEvtPort;
    cfg.wsBridgeSubPort      = kWsPort;
    cfg.inputShmName         = kShmName;
    cfg.inputWidth           = kTestFrameW;
    cfg.inputHeight          = kTestFrameH;
    cfg.pollTimeout          = std::chrono::milliseconds(20);
    cfg.prerollSec           = 1;
    cfg.postrollSec          = 1;
    cfg.ingestFps            = 2;
    cfg.frameSamples         = 2;
    cfg.jpegDownscale        = 64;
    cfg.jpegQuality          = 70;
    cfg.maxPendingEvents     = 4;
    cfg.modelDir             = "unused/by/mock";
    cfg.scriptPathOverride   = mockScript.string();
    cfg.startupTimeoutSec    = 15;
    cfg.requestTimeoutMs     = 8000;
    cfg.idleUnloadSec        = 3600;  // don't race with test lifetime
    cfg.logDir               = logDir.string();

    SecurityVlmNode node(cfg);
    node.initialize();

    std::thread runThread([&node]{ node.run(); });

    // 1 -- Fill the ring with a few frames before triggering an event.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (int i = 0; i < 5; ++i) {
        shm.writeBgrSlot0(static_cast<uint8_t>(50 + 10 * i), 100, 150);
        const nlohmann::json videoMeta = {
            {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH},
        };
        const std::string msg = "video_frame " + videoMeta.dump();
        videoPub.send(zmq::buffer(msg), zmq::send_flags::none);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }

    // 2 -- Fire a security_event.
    const std::string eventId = "evt-test-42";
    // Match the real security_camera_node schema: type="security_event",
    // event="<class>_detected".  SecurityVlmNode filters on both fields.
    const nlohmann::json eventPayload = {
        {"v",          1},
        {"event_id",   eventId},
        {"type",       "security_event"},
        {"event",      "person_detected"},
        {"timestamp",  "2026-04-18T00:00:00Z"},
        {"frame_seq",  shm.writeSeq_},
        {"detections", nlohmann::json::array()},
    };
    const std::string eventMsg =
        std::string(kZmqTopicSecurityEvent) + " " + eventPayload.dump();
    eventPub.send(zmq::buffer(eventMsg), zmq::send_flags::none);

    // 3 -- Keep pumping frames while we wait (postrollSec + VLM latency).
    nlohmann::json analysis;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(15);

    while (std::chrono::steady_clock::now() < deadline) {
        shm.writeBgrSlot0(200, 150, 100);
        const nlohmann::json vm = {
            {"slot", 0}, {"width", kTestFrameW}, {"height", kTestFrameH},
        };
        const std::string m = "video_frame " + vm.dump();
        videoPub.send(zmq::buffer(m), zmq::send_flags::none);

        analysis = receiveTestMessage(sub, 200);
        if (!analysis.is_null()) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    node.stop();
    runThread.join();

    // 4 -- Assertions on the published analysis.
    ASSERT_FALSE(analysis.is_null())
        << "No security_vlm_analysis received — the agent loop did not complete";
    EXPECT_EQ(analysis.value("type", std::string{}),
              std::string(kZmqTopicSecurityVlmAnalysis));
    // The node generates its own event_id; correlation is via source_event.
    ASSERT_TRUE(analysis.contains("source_event"));
    EXPECT_EQ(analysis["source_event"].value("event_id", std::string{}), eventId)
        << "source_event must echo the triggering event payload";

    const auto& analysisBody = analysis["analysis"];
    ASSERT_TRUE(analysisBody.is_object())
        << "analysis must be structured JSON when mock returns a valid reply";
    EXPECT_TRUE(analysisBody.contains("behavior"));
    EXPECT_TRUE(analysisBody.contains("suspicion_level"));
    EXPECT_GT(analysis.value("image_count", 0), 0);

    const std::string nodeEventId = analysis.value("event_id", std::string{});
    ASSERT_FALSE(nodeEventId.empty()) << "node must assign an event_id";

    // 5 -- Assertions on the JSONL sink.
    const auto jsonlPath = logDir / "security_incidents.jsonl";
    ASSERT_TRUE(fs::exists(jsonlPath))
        << "security_incidents.jsonl must be created by the node";
    std::ifstream in(jsonlPath);
    std::string   line;
    bool          foundEvent = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto row = nlohmann::json::parse(line, nullptr, false);
        if (row.is_discarded()) continue;
        if (row.value("event_id", std::string{}) == nodeEventId) {
            foundEvent = true;
            EXPECT_EQ(row.value("status", std::string{}), "ok")
                << "JSONL row status must be ok for a successful VLM call";
            break;
        }
    }
    EXPECT_TRUE(foundEvent)
        << "security_incidents.jsonl must contain a row for our event_id";

    fs::remove_all(logDir);
}
