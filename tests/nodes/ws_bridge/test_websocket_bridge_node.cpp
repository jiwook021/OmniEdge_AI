#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "common/runtime_defaults.hpp"
#include "ws_bridge/websocket_bridge_node.hpp"


namespace {

/// Atomic port counter to avoid port collisions between tests.
/// Each test claims a block of 10 ports (pub + several sub ports).
std::atomic<int> portCounter_{0};

int nextPortBase()
{
	return 25700 + portCounter_.fetch_add(1) * 200;
}

class MockRelay final : public IWsRelay {
public:
	void run() override
	{
		runEntered.store(true, std::memory_order_release);
		while (!stopRequested.load(std::memory_order_acquire)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
	}

	void stop() override
	{
		stopRequested.store(true, std::memory_order_release);
	}

	void broadcastVideo(std::vector<uint8_t> data) override
	{
		lastVideo = std::move(data);
	}

	void broadcastAudio(std::vector<uint8_t> data) override
	{
		lastAudio = std::move(data);
	}

	void broadcastJson(nlohmann::json msg) override
	{
		lastJson = std::move(msg);
	}

	void broadcastDenoisedVideo(std::vector<uint8_t> data) override
	{
		lastDenoisedVideo = std::move(data);
	}

	void broadcastDenoisedAudio(std::vector<uint8_t> data) override
	{
		lastDenoisedAudio = std::move(data);
	}

	void broadcastSam2Video(std::vector<uint8_t> data) override
	{
		lastSam2Video = std::move(data);
	}
	void broadcastScreenVideo(std::vector<uint8_t> data) override
	{
		lastScreenVideo = std::move(data);
	}

	void broadcastSecurityVideo(std::vector<uint8_t> data) override
	{
		lastSecurityVideo = std::move(data);
	}

	void broadcastBeautyVideo(std::vector<uint8_t> data) override
	{
		lastBeautyVideo = std::move(data);
	}

	std::atomic<bool> runEntered{false};
	std::atomic<bool> stopRequested{false};
	std::vector<uint8_t> lastVideo;
	std::vector<uint8_t> lastAudio;
	std::vector<uint8_t> lastDenoisedVideo;
	std::vector<uint8_t> lastDenoisedAudio;
	std::vector<uint8_t> lastSam2Video;
	std::vector<uint8_t> lastScreenVideo;
	std::vector<uint8_t> lastSecurityVideo;
	std::vector<uint8_t> lastBeautyVideo;
	nlohmann::json lastJson;
};

std::optional<nlohmann::json> recvJsonWithTopic(zmq::socket_t& sub, int timeoutMs)
{
	zmq::pollitem_t item{static_cast<void*>(sub), 0, ZMQ_POLLIN, 0};
	zmq::poll(&item, 1, std::chrono::milliseconds(timeoutMs));
	if (!(item.revents & ZMQ_POLLIN)) {
		return std::nullopt;
	}

	zmq::message_t msg;
	const auto recvRes = sub.recv(msg);
	if (!recvRes) {
		return std::nullopt;
	}
	std::string raw(static_cast<const char*>(msg.data()), msg.size());
	const auto space = raw.find(' ');
	if (space == std::string::npos) {
		return std::nullopt;
	}
	return nlohmann::json::parse(raw.substr(space + 1));
}

} // namespace

TEST(WebSocketBridgeNodeTest, PublishesValidatedUiCommand)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));

	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1},
		{"type", "ui_command"},
		{"action", "text_input"},
		{"text", "hello"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value());
	EXPECT_EQ(msg->value("type", ""), "ui_command");
	EXPECT_EQ(msg->value("action", ""), "text_input");
	EXPECT_EQ(msg->value("text", ""), "hello");

	node.stop();
	EXPECT_FALSE(relayPtr->runEntered.load(std::memory_order_acquire));
}

TEST(WebSocketBridgeNodeTest, InvalidCommandBroadcastsError)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	node.handleClientCommand({
		{"v", 1},
		{"type", "ui_command"},
		{"action", "push_to_talk"},
		{"state", "bad"},
	});

	ASSERT_TRUE(relayPtr->lastJson.is_object());
	EXPECT_EQ(relayPtr->lastJson.value("type", ""), "error");
	EXPECT_FALSE(relayPtr->lastJson.value("message", "").empty());

	node.stop();
}

// ---------------------------------------------------------------------------
// handleBinaryUpload — image upload binary frame parsing + ZMQ publish
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, HandleBinaryUploadWritesTempFileAndPublishes)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	// Subscribe to ui_command to verify the published message
	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// Build a binary frame with a minimal JPEG (just the magic bytes + some data)
	nlohmann::json jsonHeader = {{"action", "describe_uploaded_image"}};
	std::string jsonStr = jsonHeader.dump();
	uint32_t jsonLen = static_cast<uint32_t>(jsonStr.size());

	// Minimal JPEG: FF D8 FF E0 + padding (not a real image, but enough for magic check)
	std::vector<uint8_t> fakeJpeg = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46};

	// Assemble binary frame: [4B len LE][JSON][image]
	std::string frame;
	frame.resize(4 + jsonStr.size() + fakeJpeg.size());
	std::memcpy(frame.data(), &jsonLen, 4);
	std::memcpy(frame.data() + 4, jsonStr.data(), jsonStr.size());
	std::memcpy(frame.data() + 4 + jsonStr.size(), fakeJpeg.data(), fakeJpeg.size());

	node.handleBinaryUpload(std::string_view(frame));

	// Verify: the ZMQ message should contain describe_uploaded_image + image_path
	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ui_command received after binary upload";
	EXPECT_EQ(msg->value("action", ""), "describe_uploaded_image");

	// Verify image_path starts with expected prefix and ends with .jpg
	std::string imagePath = msg->value("image_path", "");
	EXPECT_TRUE(imagePath.find("/tmp/oe_upload_") == 0)
		<< "Unexpected image_path: " << imagePath;
	EXPECT_TRUE(imagePath.find(".jpg") != std::string::npos)
		<< "Expected .jpg extension: " << imagePath;

	// Verify the temp file exists and has the correct content
	ASSERT_TRUE(std::filesystem::exists(imagePath))
		<< "Temp file should exist: " << imagePath;
	std::ifstream f(imagePath, std::ios::binary);
	std::vector<uint8_t> content((std::istreambuf_iterator<char>(f)),
	                              std::istreambuf_iterator<char>());
	EXPECT_EQ(content, fakeJpeg);

	// Clean up
	std::filesystem::remove(imagePath);
	node.stop();
}

TEST(WebSocketBridgeNodeTest, HandleBinaryUploadRejectsTooSmallFrame)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	// Send only 2 bytes — too small for the 4-byte header
	node.handleBinaryUpload(std::string_view("ab"));

	// Should broadcast an error
	ASSERT_TRUE(relayPtr->lastJson.is_object());
	EXPECT_EQ(relayPtr->lastJson.value("type", ""), "error");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, HandleBinaryUploadRejectsUnknownImageFormat)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	// Build a binary frame with non-image data (GIF magic bytes — not supported)
	nlohmann::json jsonHeader = {{"action", "describe_uploaded_image"}};
	std::string jsonStr = jsonHeader.dump();
	uint32_t jsonLen = static_cast<uint32_t>(jsonStr.size());

	// GIF magic: 47 49 46 38
	std::vector<uint8_t> gifData = {0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00};

	std::string frame;
	frame.resize(4 + jsonStr.size() + gifData.size());
	std::memcpy(frame.data(), &jsonLen, 4);
	std::memcpy(frame.data() + 4, jsonStr.data(), jsonStr.size());
	std::memcpy(frame.data() + 4 + jsonStr.size(), gifData.data(), gifData.size());

	node.handleBinaryUpload(std::string_view(frame));

	// Should broadcast an error about unsupported format
	ASSERT_TRUE(relayPtr->lastJson.is_object());
	EXPECT_EQ(relayPtr->lastJson.value("type", ""), "error");
	EXPECT_TRUE(relayPtr->lastJson.value("message", "").find("Unsupported") != std::string::npos);

	node.stop();
}

TEST(WebSocketBridgeNodeTest, HandleBinaryUploadAcceptsPng)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// Build a binary frame with PNG magic bytes
	nlohmann::json jsonHeader = {{"action", "describe_uploaded_image"}};
	std::string jsonStr = jsonHeader.dump();
	uint32_t jsonLen = static_cast<uint32_t>(jsonStr.size());

	// PNG 8-byte signature
	std::vector<uint8_t> fakePng = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
	                                 0x00, 0x00, 0x00, 0x0D};

	std::string frame;
	frame.resize(4 + jsonStr.size() + fakePng.size());
	std::memcpy(frame.data(), &jsonLen, 4);
	std::memcpy(frame.data() + 4, jsonStr.data(), jsonStr.size());
	std::memcpy(frame.data() + 4 + jsonStr.size(), fakePng.data(), fakePng.size());

	node.handleBinaryUpload(std::string_view(frame));

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ui_command for PNG upload";
	EXPECT_EQ(msg->value("action", ""), "describe_uploaded_image");

	std::string imagePath = msg->value("image_path", "");
	EXPECT_TRUE(imagePath.find(".png") != std::string::npos)
		<< "Expected .png extension: " << imagePath;

	// Clean up
	if (std::filesystem::exists(imagePath)) {
		std::filesystem::remove(imagePath);
	}
	node.stop();
}

// ---------------------------------------------------------------------------
// New UI command tests: select_conversation_model via ZMQ publish
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, PublishesSelectConversationModel)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1},
		{"type", "ui_command"},
		{"action", "select_conversation_model"},
		{"model", "gemma_e2b"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for select_conversation_model";
	EXPECT_EQ(msg->value("action", ""), "select_conversation_model");
	EXPECT_EQ(msg->value("model", ""), "gemma_e2b");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, RejectsFormerlyMissingSelectConversationModel)
{
	// Regression: select_conversation_model used to be rejected with
	// "unsupported ui_command action". Verify it no longer is.
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	// This command used to fail with "unsupported ui_command action"
	node.handleClientCommand({
		{"v", 1},
		{"type", "ui_command"},
		{"action", "select_conversation_model"},
		{"model", "gemma_e4b"},
	});

	// Should NOT broadcast an error
	if (relayPtr->lastJson.is_object()) {
		EXPECT_NE(relayPtr->lastJson.value("type", ""), "error")
			<< "select_conversation_model should no longer be rejected. "
			<< "Error: " << relayPtr->lastJson.value("message", "");
	}

	node.stop();
}

TEST(WebSocketBridgeNodeTest, AcceptsSwitchModeConversation)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;

	auto relay = std::make_unique<MockRelay>();
	MockRelay* relayPtr = relay.get();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	// switch_mode with "conversation" — default mode at boot
	node.handleClientCommand({
		{"v", 1},
		{"type", "ui_command"},
		{"action", "switch_mode"},
		{"mode", "conversation"},
	});

	// Should NOT broadcast an error
	if (relayPtr->lastJson.is_object()) {
		EXPECT_NE(relayPtr->lastJson.value("type", ""), "error")
			<< "switch_mode conversation should be accepted. "
			<< "Error: " << relayPtr->lastJson.value("message", "");
	}

	node.stop();
}

// ---------------------------------------------------------------------------
// Blur toggle: set_bg_mode blur/none — verifies the blur preview UI path
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, PublishesSetBgModeBlur)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "set_bg_mode"}, {"mode", "blur"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for set_bg_mode blur";
	EXPECT_EQ(msg->value("action", ""), "set_bg_mode");
	EXPECT_EQ(msg->value("mode", ""), "blur");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesSetBgModeNone)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "set_bg_mode"}, {"mode", "none"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for set_bg_mode none";
	EXPECT_EQ(msg->value("action", ""), "set_bg_mode");
	EXPECT_EQ(msg->value("mode", ""), "none");

	node.stop();
}

// ---------------------------------------------------------------------------
// Action-driven GPU priority: toggle_video_denoise triggers VSR++ priority
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, PublishesToggleVideoDenoiseOn)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// Enable BasicVSR++ — should become highest GPU priority, evict LLM
	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_video_denoise"}, {"enabled", true},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for toggle_video_denoise on";
	EXPECT_EQ(msg->value("action", ""), "toggle_video_denoise");
	EXPECT_EQ(msg->value("enabled", false), true);

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesToggleVideoDenoiseOff)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// Disable BasicVSR++ — LLM should resume as highest priority
	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_video_denoise"}, {"enabled", false},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for toggle_video_denoise off";
	EXPECT_EQ(msg->value("action", ""), "toggle_video_denoise");
	EXPECT_EQ(msg->value("enabled", true), false);

	node.stop();
}

// ---------------------------------------------------------------------------
// Audio denoise toggle
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, PublishesToggleAudioDenoise)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_audio_denoise"}, {"enabled", true},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for toggle_audio_denoise";
	EXPECT_EQ(msg->value("action", ""), "toggle_audio_denoise");
	EXPECT_EQ(msg->value("enabled", false), true);

	node.stop();
}

// ---------------------------------------------------------------------------
// Complete UI action coverage: remaining actions that pass through ws_bridge
// to ZMQ. Each test verifies: user clicks button → JSON command → validation →
// ZMQ publish on ui_command topic.
// ---------------------------------------------------------------------------

TEST(WebSocketBridgeNodeTest, PublishesPushToTalk)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "push_to_talk"}, {"state", true},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for push_to_talk";
	EXPECT_EQ(msg->value("action", ""), "push_to_talk");
	EXPECT_EQ(msg->value("state", false), true);

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesDescribeScene)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "describe_scene"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for describe_scene";
	EXPECT_EQ(msg->value("action", ""), "describe_scene");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesRegisterFace)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "register_face"}, {"name", "alice"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for register_face";
	EXPECT_EQ(msg->value("action", ""), "register_face");
	EXPECT_EQ(msg->value("name", ""), "alice");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesCancelGeneration)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "cancel_generation"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for cancel_generation";
	EXPECT_EQ(msg->value("action", ""), "cancel_generation");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesTtsComplete)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "tts_complete"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for tts_complete";
	EXPECT_EQ(msg->value("action", ""), "tts_complete");

	node.stop();
}

TEST(WebSocketBridgeNodeTest, PublishesFlushTts)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort = base;
	cfg.blurSubPort     = base + 1;
	cfg.ttsSubPort      = base + 2;
	cfg.llmSubPort      = base + 3;
	cfg.daemonSubPort   = base + 4;
	cfg.faceSubPort     = base + 5;
	cfg.pollTimeout     = std::chrono::milliseconds(10);

	auto relay = std::make_unique<MockRelay>();

	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "flush_tts"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for flush_tts";
	EXPECT_EQ(msg->value("action", ""), "flush_tts");

	node.stop();
}

// ---------------------------------------------------------------------------
// SAM2 actions: stop_tracking, bg_mode, bg_color, bg_image
// ---------------------------------------------------------------------------

TEST(WsBridge, Sam2StopTracking_AcceptedAndPublished)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort      = base;
	cfg.blurSubPort          = base + 1;
	cfg.videoIngestSubPort   = base + 2;
	cfg.ttsSubPort           = base + 3;
	cfg.llmSubPort           = base + 4;
	cfg.daemonSubPort        = base + 5;
	cfg.faceSubPort          = base + 6;
	cfg.videoDenoiseSubPort  = base + 7;
	cfg.audioDenoiseSubPort  = base + 8;
	cfg.sam2SubPort          = base + 9;

	auto relay = std::make_unique<MockRelay>();
	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "sam2_stop_tracking"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for sam2_stop_tracking";
	EXPECT_EQ(msg->value("action", ""), "sam2_stop_tracking");

	node.stop();
}

TEST(WsBridge, Sam2SetBgMode_AcceptedAndPublished)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort      = base;
	cfg.blurSubPort          = base + 1;
	cfg.videoIngestSubPort   = base + 2;
	cfg.ttsSubPort           = base + 3;
	cfg.llmSubPort           = base + 4;
	cfg.daemonSubPort        = base + 5;
	cfg.faceSubPort          = base + 6;
	cfg.videoDenoiseSubPort  = base + 7;
	cfg.audioDenoiseSubPort  = base + 8;
	cfg.sam2SubPort          = base + 9;

	auto relay = std::make_unique<MockRelay>();
	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "set_sam2_bg_mode"}, {"mode", "blur"},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for set_sam2_bg_mode";
	EXPECT_EQ(msg->value("action", ""), "set_sam2_bg_mode");
	EXPECT_EQ(msg->value("mode", ""), "blur");

	node.stop();
}

TEST(WsBridge, Sam2SetBgMode_MissingModeRejected)
{
	auto result = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "set_sam2_bg_mode"},  // missing "mode"
	});
	ASSERT_FALSE(result.has_value())
		<< "set_sam2_bg_mode without mode field must be rejected";
}

TEST(WsBridge, Sam2SetBgColor_AcceptedAndPublished)
{
	const int base = nextPortBase();
	WebSocketBridgeNode::Config cfg;
	cfg.wsBridgePubPort      = base;
	cfg.blurSubPort          = base + 1;
	cfg.videoIngestSubPort   = base + 2;
	cfg.ttsSubPort           = base + 3;
	cfg.llmSubPort           = base + 4;
	cfg.daemonSubPort        = base + 5;
	cfg.faceSubPort          = base + 6;
	cfg.videoDenoiseSubPort  = base + 7;
	cfg.audioDenoiseSubPort  = base + 8;
	cfg.sam2SubPort          = base + 9;

	auto relay = std::make_unique<MockRelay>();
	WebSocketBridgeNode node(cfg);
	node.setRelay(std::move(relay));
	node.initialize();

	zmq::context_t ctx{1};
	zmq::socket_t sub(ctx, ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string("ui_command"));
	sub.connect(std::format("ipc:///tmp/omniedge_{}", base));
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	node.handleClientCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "set_sam2_bg_color"}, {"r", 255}, {"g", 0}, {"b", 128},
	});

	auto msg = recvJsonWithTopic(sub, 500);
	ASSERT_TRUE(msg.has_value()) << "No ZMQ message for set_sam2_bg_color";
	EXPECT_EQ(msg->value("action", ""), "set_sam2_bg_color");

	node.stop();
}

