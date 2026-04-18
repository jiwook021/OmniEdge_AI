// test_ws_bridge_relay.cpp — Integration tests for ZMQ ↔ WebSocket relay paths
//
// Covers:
//   - Upstream ZMQ PUB → node SUB → IWsRelay::broadcastJson (JSON topics)
//   - Rapid handleClientCommand burst → ZMQ PUB verification
//   - All user action types accepted without error
//   - Node lifecycle (start/stop) safety

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "common/constants/ws_bridge_constants.hpp"
#include "common/runtime_defaults.hpp"
#include "ws_bridge/websocket_bridge_node.hpp"


namespace {

std::atomic<int> relayPortCounter_{0};

int nextRelayPortBase()
{
	return 28000 + relayPortCounter_.fetch_add(1) * 200;
}

// ── Thread-safe MockRelay with CV-based waiting ──────────────────────────────

class ThreadSafeMockRelay final : public IWsRelay {
public:
	void run() override
	{
		{
			std::lock_guard<std::mutex> lk(runMu_);
			runEntered_.store(true, std::memory_order_release);
		}
		runCv_.notify_all();
		std::unique_lock<std::mutex> lk(stopMu_);
		stopCv_.wait(lk, [this] { return stopRequested_.load(std::memory_order_acquire); });
	}

	void stop() override
	{
		{
			std::lock_guard<std::mutex> lk(stopMu_);
			stopRequested_.store(true, std::memory_order_release);
		}
		stopCv_.notify_all();
	}

	void broadcastVideo(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastVideo_ = std::move(data);
		videoCt_.fetch_add(1, std::memory_order_release);
		dataCv_.notify_all();
	}

	void broadcastAudio(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastAudio_ = std::move(data);
		audioCt_.fetch_add(1, std::memory_order_release);
		dataCv_.notify_all();
	}

	void broadcastJson(nlohmann::json msg) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastJson_ = std::move(msg);
		jsonCt_.fetch_add(1, std::memory_order_release);
		dataCv_.notify_all();
	}

	void broadcastDenoisedVideo(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastDenoisedVideo_ = std::move(data);
		dataCv_.notify_all();
	}

	void broadcastDenoisedAudio(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastDenoisedAudio_ = std::move(data);
		dataCv_.notify_all();
	}

	void broadcastSam2Video(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastSam2Video_ = std::move(data);
		dataCv_.notify_all();
	}

	void broadcastScreenVideo(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastScreenVideo_ = std::move(data);
		dataCv_.notify_all();
	}

	void broadcastSecurityVideo(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastSecurityVideo_ = std::move(data);
		dataCv_.notify_all();
	}

	void broadcastBeautyVideo(std::vector<uint8_t> data) override
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		lastBeautyVideo_ = std::move(data);
		dataCv_.notify_all();
	}

	// ── Wait helpers ─────────────────────────────────────────────────────

	bool waitForRunEntered(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000))
	{
		std::unique_lock<std::mutex> lk(runMu_);
		return runCv_.wait_for(lk, timeout, [this] {
			return runEntered_.load(std::memory_order_acquire);
		});
	}

	bool waitForJson(int minCount = 1,
	                 std::chrono::milliseconds timeout = std::chrono::milliseconds(2000))
	{
		std::unique_lock<std::mutex> lk(dataMu_);
		return dataCv_.wait_for(lk, timeout, [this, minCount] {
			return jsonCt_.load(std::memory_order_acquire) >= minCount;
		});
	}

	// ── Getters ──────────────────────────────────────────────────────────

	nlohmann::json getLastJson() const
	{
		std::lock_guard<std::mutex> lk(dataMu_);
		return lastJson_;
	}

	int jsonCount() const { return jsonCt_.load(std::memory_order_acquire); }

private:
	mutable std::mutex dataMu_;
	std::condition_variable dataCv_;

	std::mutex runMu_;
	std::condition_variable runCv_;
	std::atomic<bool> runEntered_{false};

	std::mutex stopMu_;
	std::condition_variable stopCv_;
	std::atomic<bool> stopRequested_{false};

	std::vector<uint8_t> lastVideo_;
	std::vector<uint8_t> lastAudio_;
	std::vector<uint8_t> lastDenoisedVideo_;
	std::vector<uint8_t> lastDenoisedAudio_;
	std::vector<uint8_t> lastSam2Video_;
	std::vector<uint8_t> lastScreenVideo_;
	std::vector<uint8_t> lastSecurityVideo_;
	std::vector<uint8_t> lastBeautyVideo_;
	nlohmann::json lastJson_;

	std::atomic<int> videoCt_{0};
	std::atomic<int> audioCt_{0};
	std::atomic<int> jsonCt_{0};
};

// ── Helpers ──────────────────────────────────────────────────────────────────

WebSocketBridgeNode::Config makeIsolatedConfig(int base)
{
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort     = base;
	cfg.blurSubPort         = base + 1;
	cfg.ttsSubPort          = base + 2;
	cfg.llmSubPort          = base + 3;
	cfg.daemonSubPort       = base + 4;
	cfg.faceSubPort         = base + 5;
	cfg.videoIngestSubPort  = base + 6;
	cfg.videoDenoiseSubPort = base + 7;
	cfg.audioDenoiseSubPort = base + 8;
	cfg.pollTimeout         = std::chrono::milliseconds(10);
	return cfg;
}

void publishTopicJson(zmq::socket_t& pub, std::string_view topic,
                      const nlohmann::json& payload)
{
	std::string msg = std::format("{} {}", topic, payload.dump());
	pub.send(zmq::buffer(msg), zmq::send_flags::none);
}

std::optional<nlohmann::json> recvJson(zmq::socket_t& sub, int timeoutMs)
{
	zmq::pollitem_t item{static_cast<void*>(sub), 0, ZMQ_POLLIN, 0};
	zmq::poll(&item, 1, std::chrono::milliseconds(timeoutMs));
	if (!(item.revents & ZMQ_POLLIN)) return std::nullopt;

	zmq::message_t msg;
	const auto recvRes = sub.recv(msg);
	if (!recvRes) return std::nullopt;
	std::string raw(static_cast<const char*>(msg.data()), msg.size());
	auto sp = raw.find(' ');
	if (sp == std::string::npos) return std::nullopt;
	return nlohmann::json::parse(raw.substr(sp + 1));
}

/// RAII wrapper that runs node.run() in a background thread and guarantees
/// clean shutdown + join in the destructor.
struct NodeRunner {
	WebSocketBridgeNode& node;
	ThreadSafeMockRelay& relay;
	std::thread thread;

	explicit NodeRunner(WebSocketBridgeNode& n, ThreadSafeMockRelay& r)
		: node(n), relay(r), thread([&n]() { n.run(); })
	{
		if (!relay.waitForRunEntered()) {
			throw std::runtime_error("Relay did not start within timeout");
		}
	}

	~NodeRunner()
	{
		node.stop();
		if (thread.joinable()) thread.join();
	}
};

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
//  ZMQ → Relay: upstream PUB → node SUB → MockRelay::broadcastJson
// ═══════════════════════════════════════════════════════════════════════════════

TEST(WebSocketBridgeRelayTest, ForwardsLlmResponseToJson)
{
	const int base = nextRelayPortBase();
	auto cfg = makeIsolatedConfig(base);

	auto relay = std::make_unique<ThreadSafeMockRelay>();
	auto* rp = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();
	NodeRunner runner(node, *rp);

	zmq::context_t ctx{1};
	zmq::socket_t pub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", base + 3));
	pub.bind(std::format("ipc:///tmp/omniedge_{}", base + 3));
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	nlohmann::json payload = {
		{"type", std::string(kZmqTopicLlmResponse)}, {"token", "Hello"}, {"finished", false}};
	publishTopicJson(pub, std::string(kZmqTopicLlmResponse), payload);

	ASSERT_TRUE(rp->waitForJson()) << "broadcastJson not called for llm_response";
	auto got = rp->getLastJson();
	EXPECT_EQ(got.value("type", ""), std::string(kZmqTopicLlmResponse));
	EXPECT_EQ(got.value("token", ""), "Hello");
}

TEST(WebSocketBridgeRelayTest, ForwardsModuleStatusToJson)
{
	const int base = nextRelayPortBase();
	auto cfg = makeIsolatedConfig(base);

	auto relay = std::make_unique<ThreadSafeMockRelay>();
	auto* rp = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();
	NodeRunner runner(node, *rp);

	zmq::context_t ctx{1};
	zmq::socket_t pub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", base + 4));
	pub.bind(std::format("ipc:///tmp/omniedge_{}", base + 4));
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	nlohmann::json payload = {
		{"type", std::string(kZmqTopicModuleStatus)}, {"module", "stt"}, {"state", "ready"}};
	publishTopicJson(pub, std::string(kZmqTopicModuleStatus), payload);

	ASSERT_TRUE(rp->waitForJson());
	auto got = rp->getLastJson();
	EXPECT_EQ(got.value("module", ""), "stt");
	EXPECT_EQ(got.value("state", ""), "ready");
}

TEST(WebSocketBridgeRelayTest, ForwardsFaceRegisteredToJson)
{
	const int base = nextRelayPortBase();
	auto cfg = makeIsolatedConfig(base);

	auto relay = std::make_unique<ThreadSafeMockRelay>();
	auto* rp = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();
	NodeRunner runner(node, *rp);

	zmq::context_t ctx{1};
	zmq::socket_t pub(ctx, ZMQ_PUB);
	std::filesystem::remove(std::format("/tmp/omniedge_{}", base + 5));
	pub.bind(std::format("ipc:///tmp/omniedge_{}", base + 5));
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	nlohmann::json payload = {
		{"type", "face_registered"}, {"name", "Alice"}, {"success", true}};
	publishTopicJson(pub, "face_registered", payload);

	ASSERT_TRUE(rp->waitForJson());
	auto got = rp->getLastJson();
	EXPECT_EQ(got.value("name", ""), "Alice");
	EXPECT_TRUE(got.value("success", false));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  User input → ZMQ PUB: handleClientCommand burst
// ═══════════════════════════════════════════════════════════════════════════════

TEST(WebSocketBridgeRelayTest, RapidCommandBurstAllPublished)
{
	const int base = nextRelayPortBase();
	auto cfg = makeIsolatedConfig(base);

	auto relay = std::make_unique<ThreadSafeMockRelay>();
	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	constexpr int kBurstSize = 20;
	for (int i = 0; i < kBurstSize; ++i) {
		node.handleClientCommand({
			{"v", 1},
			{"type", "ui_command"},
			{"action", "text_input"},
			{"text", std::format("msg_{}", i)},
		});
	}

	int received = 0;
	for (int i = 0; i < kBurstSize; ++i) {
		auto msg = recvJson(sub, 500);
		if (msg.has_value()) ++received;
	}
	EXPECT_EQ(received, kBurstSize)
		<< "Not all messages received in rapid burst";

	node.stop();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  All user action types accepted without error
// ═══════════════════════════════════════════════════════════════════════════════

TEST(WebSocketBridgeRelayTest, AllUserActionTypesAccepted)
{
	const int base = nextRelayPortBase();
	auto cfg = makeIsolatedConfig(base);

	auto relay = std::make_unique<ThreadSafeMockRelay>();
	auto* rp = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(30));

	std::vector<nlohmann::json> actions = {
		{{"action", "text_input"}, {"text", "hello"}},
		{{"action", "push_to_talk"}, {"state", true}},
		{{"action", "push_to_talk"}, {"state", false}},
		{{"action", "describe_scene"}},
		{{"action", "cancel_generation"}},
		{{"action", "switch_mode"}, {"mode", "conversation"}},
		{{"action", "set_bg_mode"}, {"mode", "blur"}},
		{{"action", "set_bg_mode"}, {"mode", "none"}},
		{{"action", "tts_complete"}},
		{{"action", "flush_tts"}},
		{{"action", "stop_playback"}},
		{{"action", "toggle_stt"}, {"enabled", true}},
		{{"action", "toggle_tts"}, {"enabled", false}},
		{{"action", "toggle_background_blur"}, {"enabled", true}},
		{{"action", "toggle_video_denoise"}, {"enabled", false}},
		{{"action", "toggle_audio_denoise"}, {"enabled", true}},
		{{"action", "toggle_face_recognition"}, {"enabled", false}},
	};

	for (auto& a : actions) {
		a["v"] = 1;
		a["type"] = "ui_command";
		node.handleClientCommand(a);

		auto last = rp->getLastJson();
		if (last.is_object()) {
			EXPECT_NE(last.value("type", ""), "error")
				<< "Error for action " << a.value("action", "?") << ": "
				<< last.value("message", "?");
		}
	}

	int received = 0;
	for (std::size_t i = 0; i < actions.size(); ++i) {
		auto msg = recvJson(sub, 500);
		if (msg.has_value()) ++received;
	}
	EXPECT_EQ(received, static_cast<int>(actions.size()));

	node.stop();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Node lifecycle: repeated start + stop without crash
// ═══════════════════════════════════════════════════════════════════════════════

TEST(WebSocketBridgeRelayTest, NodeStartStopCyclesAreSafe)
{
	const int base = nextRelayPortBase();

	for (int cycle = 0; cycle < 3; ++cycle) {
		auto cfg = makeIsolatedConfig(base + cycle * 200);

		auto relay = std::make_unique<ThreadSafeMockRelay>();
		auto* rp = relay.get();

		WebSocketBridgeNode node(cfg);
		node.setRelay(std::move(relay));
		node.initialize();

		std::thread t([&node]() { node.run(); });
		ASSERT_TRUE(rp->waitForRunEntered()) << "Cycle " << cycle << " stalled";

		node.handleClientCommand({
			{"v", 1},
			{"type", "ui_command"},
			{"action", "describe_scene"},
		});

		node.stop();
		t.join();
	}
}

