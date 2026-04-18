// test_snapshot_command.cpp — ScreenIngestNode snapshot round-trip
//
// Validates the `{action:"snapshot", source:"screen"}` ui_command contract:
//
//   1. Seed the node's JPEG cache via the friend-class back door
//      (simulating one frame already received from the Windows capture agent).
//   2. Run the node's MessageRouter poll loop on a background thread.
//   3. Publish `ui_command` from a test PUB socket bound to the node's
//      wsBridgeSubPort (role of ws_bridge relay).
//   4. Subscribe to the node's pubPort for topic `snapshot_response`.
//   5. Assert the JSON payload carries source=screen, width/height match,
//      and base64-decoded jpeg_b64 begins with the JPEG magic 0xFFD8FF.
//
// This is the C++ side of the contract the omniedge-mcp `describe_screen`
// tool relies on. No mocks — real ZMQ, real base64, real ScreenIngestNode.

#include <gtest/gtest.h>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "ingest/screen_ingest_node.hpp"
#include "common/constants/ingest_constants.hpp"
#include "zmq/zmq_endpoint.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

// Test ports sit at +20000 offset so they never clash with a running daemon.
constexpr int kTestPubPort         = 25577;
constexpr int kTestWsBridgeSubPort = 25570;

/// Fabricate a 4-byte "JPEG" (magic + EOI) — sufficient for the round-trip
/// test since handleSnapshotRequest does not decode. Real JPEG round-trips
/// are covered by the Python-side live test.
std::vector<std::uint8_t> makeTinyJpeg()
{
    return {0xFFu, 0xD8u, 0xFFu, 0xD9u};
}

/// Receive a message on `sub` with a deadline, filter to `expectedTopic`.
/// OmniEdge's wire format is a single ZMQ frame `"<topic> <json>"` (space
/// separated) — see ZmqPublisher::publish. We mirror that parsing here.
/// Returns (topic, body) on success, or ("", empty json) on timeout.
std::pair<std::string, nlohmann::json>
recvWithTimeout(zmq::socket_t& sub, const std::string& expectedTopic,
                std::chrono::milliseconds budget)
{
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline) {
        zmq::pollitem_t item{sub.handle(), 0, ZMQ_POLLIN, 0};
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) break;
        zmq::poll(&item, 1, std::chrono::milliseconds(remaining));
        if (!(item.revents & ZMQ_POLLIN)) continue;

        zmq::message_t msg;
        if (!sub.recv(msg, zmq::recv_flags::dontwait)) continue;

        const std::string_view raw{static_cast<const char*>(msg.data()), msg.size()};
        const auto space = raw.find(' ');
        if (space == std::string_view::npos) continue;

        const std::string topic{raw.substr(0, space)};
        if (topic != expectedTopic) continue;

        try {
            auto body = nlohmann::json::parse(raw.substr(space + 1));
            return {topic, body};
        } catch (const std::exception&) {
            continue;
        }
    }
    return {"", nlohmann::json{}};
}

/// Format a command in the same single-frame form ZmqPublisher emits:
/// `"<topic> <json>"`.
void publishSingleFrame(zmq::socket_t& pub, const std::string& topic,
                        const nlohmann::json& payload)
{
    const std::string frame = topic + " " + payload.dump();
    pub.send(zmq::buffer(frame), zmq::send_flags::none);
}

} // namespace

/// Grants the test access to ScreenIngestNode's private onTcpFrame seed path.
/// Declared as a friend in screen_ingest_node.hpp.
class ScreenIngestSnapshotTest : public ::testing::Test {
protected:
    static void seedFrame(ScreenIngestNode& node,
                          std::span<const std::uint8_t> jpeg,
                          uint32_t width, uint32_t height)
    {
        node.onTcpFrame(jpeg, width, height);
    }

    static ScreenIngestNode::Config makeCfg(int pubPort, int subPort,
                                            std::string name)
    {
        ScreenIngestNode::Config cfg;
        cfg.moduleName       = std::move(name);
        cfg.pubPort          = pubPort;
        cfg.wsBridgeSubPort  = subPort;
        cfg.windowsHostIp    = "127.0.0.1";
        cfg.tcpPort          = 1;               // unused in test — TCP just retries
        cfg.healthTimeoutSec = 5;
        return cfg;
    }
};

TEST_F(ScreenIngestSnapshotTest, SnapshotResponseDeliversBase64Jpeg)
{
    auto cfg = makeCfg(kTestPubPort, kTestWsBridgeSubPort, "screen_ingest_test");

    ScreenIngestNode node(cfg);
    ASSERT_NO_THROW(node.initialize());

    // Seed the lastJpeg_ cache before the poll loop starts. onTcpFrame takes
    // the same mutex the snapshot handler uses, so it is safe to call from
    // the test thread.
    const auto jpeg = makeTinyJpeg();
    seedFrame(node, std::span<const std::uint8_t>(jpeg.data(), jpeg.size()),
              1920, 1080);

    std::thread runner([&] { node.run(); });

    // Give the poll loop a moment to begin polling.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    zmq::context_t ctx(1);

    // Test PUB binds on wsBridgeSubPort — the node's SUB connects to this.
    // OmniEdge uses IPC (Unix domain sockets), not TCP — see zmq_endpoint.hpp.
    // The node already connected its SUB here in initialize(); remove any
    // stale file a prior crashed run left behind before binding.
    cleanStaleEndpoint(cfg.wsBridgeSubPort);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.bind(zmqEndpoint(cfg.wsBridgeSubPort));

    // Test SUB connects to the node's pubPort (MessageRouter binds PUB there).
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect(zmqEndpoint(cfg.pubPort));
    sub.set(zmq::sockopt::subscribe, std::string{kZmqTopicSnapshotResponse});

    // PUB/SUB has no handshake, so we resend the command periodically until
    // the response arrives or the deadline is reached. This defeats the
    // classic slow-joiner race without relying on a hardcoded settle time.
    nlohmann::json cmd;
    cmd["action"] = "snapshot";
    cmd["source"] = "screen";

    std::string  topic;
    nlohmann::json body;
    const auto overallDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < overallDeadline) {
        publishSingleFrame(pub, "ui_command", cmd);
        auto [tp, bd] = recvWithTimeout(
            sub, std::string{kZmqTopicSnapshotResponse},
            std::chrono::milliseconds(250));
        if (!tp.empty()) {
            topic = std::move(tp);
            body  = std::move(bd);
            break;
        }
    }

    node.stop();
    if (runner.joinable()) runner.join();

    ASSERT_EQ(topic, std::string{kZmqTopicSnapshotResponse})
        << "no snapshot_response within deadline — subscribe wiring broken?";
    EXPECT_EQ(body.value("source", std::string{}), "screen");
    EXPECT_EQ(body.value("width",  uint32_t{0}), uint32_t{1920});
    EXPECT_EQ(body.value("height", uint32_t{0}), uint32_t{1080});
    EXPECT_EQ(body.value("jpeg_len", std::size_t{0}), jpeg.size());

    // Expected base64 of {0xFF, 0xD8, 0xFF, 0xD9} is "/9j/2Q==".
    EXPECT_EQ(body.value("jpeg_b64", std::string{}), "/9j/2Q==");
}

TEST_F(ScreenIngestSnapshotTest, SnapshotResponseReportsErrorWhenNoFrameCached)
{
    // Offset ports so a rapid successive run doesn't hit TIME_WAIT on the
    // previous test's bind.
    auto cfg = makeCfg(kTestPubPort + 2, kTestWsBridgeSubPort + 2,
                       "screen_ingest_test_empty");

    ScreenIngestNode node(cfg);
    ASSERT_NO_THROW(node.initialize());
    std::thread runner([&] { node.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    zmq::context_t ctx(1);
    cleanStaleEndpoint(cfg.wsBridgeSubPort);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.bind(zmqEndpoint(cfg.wsBridgeSubPort));

    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect(zmqEndpoint(cfg.pubPort));
    sub.set(zmq::sockopt::subscribe, std::string{kZmqTopicSnapshotResponse});

    nlohmann::json cmd;
    cmd["action"] = "snapshot";
    cmd["source"] = "screen";

    std::string  topic;
    nlohmann::json body;
    const auto overallDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < overallDeadline) {
        publishSingleFrame(pub, "ui_command", cmd);
        auto [tp, bd] = recvWithTimeout(
            sub, std::string{kZmqTopicSnapshotResponse},
            std::chrono::milliseconds(250));
        if (!tp.empty()) {
            topic = std::move(tp);
            body  = std::move(bd);
            break;
        }
    }

    node.stop();
    if (runner.joinable()) runner.join();

    ASSERT_EQ(topic, std::string{kZmqTopicSnapshotResponse});
    EXPECT_EQ(body.value("source", std::string{}), "screen");
    EXPECT_EQ(body.value("jpeg_len", std::size_t{0}), std::size_t{0});
    EXPECT_FALSE(body.value("error", std::string{}).empty())
        << "expected an explanatory `error` field when no frame is cached";
}
