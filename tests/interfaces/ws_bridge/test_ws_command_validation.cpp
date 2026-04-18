#include <gtest/gtest.h>

#include "ws_bridge/websocket_bridge_node.hpp"


TEST(WsCommandValidationTest, AcceptsPrimaryActions)
{
	std::vector<nlohmann::json> commands = {
		{{"v", 1}, {"type", "ui_command"}, {"action", "push_to_talk"}, {"state", true}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "text_input"}, {"text", "hello"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "describe_scene"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "describe_uploaded_image"}, {"image_path", "/tmp/oe_upload_1.jpg"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "register_face"}, {"name", "alice"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "cancel_generation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "conversation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "tts_complete"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "blur"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "none"}},
	};

	for (const auto& cmd : commands) {
		auto validated = WebSocketBridgeNode::validateUiCommand(cmd);
		ASSERT_TRUE(validated.has_value()) << validated.error();
		EXPECT_EQ(validated->value("type", ""), "ui_command");
	}
}

TEST(WsCommandValidationTest, RejectsUnknownAction)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "unsupported_action"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsSwitchModeWithoutMode)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsSwitchModeUnknownMode)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "invalid"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsPushToTalkWithoutBooleanState)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "push_to_talk"}, {"state", "yes"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsBgModeWithoutMode)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsBgModeUnknownMode)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "bokeh"},
	});
	ASSERT_FALSE(validated.has_value());
}

// ---------------------------------------------------------------------------
// describe_uploaded_image validation
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AcceptsDescribeUploadedImageWithPath)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "describe_uploaded_image"},
		{"image_path", "/tmp/oe_upload_42.jpg"},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("action", ""), "describe_uploaded_image");
	EXPECT_EQ(validated->value("image_path", ""), "/tmp/oe_upload_42.jpg");
}

TEST(WsCommandValidationTest, RejectsDescribeUploadedImageWithoutPath)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "describe_uploaded_image"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsDescribeUploadedImageWithNonStringPath)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "describe_uploaded_image"},
		{"image_path", 12345},
	});
	ASSERT_FALSE(validated.has_value());
}

// ---------------------------------------------------------------------------
// New UI commands: select_conversation_model,
// expanded switch_mode modes
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AcceptsSelectConversationModelWithModel)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "select_conversation_model"},
		{"model", "gemma_e4b"},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("action", ""), "select_conversation_model");
	EXPECT_EQ(validated->value("model", ""), "gemma_e4b");
}

TEST(WsCommandValidationTest, RejectsSelectConversationModelWithoutModel)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "select_conversation_model"},
	});
	ASSERT_FALSE(validated.has_value());
}

TEST(WsCommandValidationTest, RejectsSelectConversationModelWithNonStringModel)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "select_conversation_model"},
		{"model", 42},
	});
	ASSERT_FALSE(validated.has_value());
}

// ---------------------------------------------------------------------------
// Expanded switch_mode: conversation, sam2_segmentation, vision_model, security, beauty
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AcceptsSwitchModeConversation)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "switch_mode"},
		{"mode", "conversation"},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("mode", ""), "conversation");
}

// ---------------------------------------------------------------------------
// Action-driven GPU priority tests: verify commands that drive GPU allocation
// When BasicVSR++ is enabled, it sends toggle_video_denoise which the daemon
// interprets as highest priority, evicting LLM. These tests verify the ws_bridge
// correctly validates the commands that UI button clicks produce.
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, ActionDrivenVsrppEnableCommand)
{
	// When user clicks BasicVSR++ button → toggle_video_denoise enabled=true
	// Daemon should make VSR++ highest priority and evict LLM
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_video_denoise"},
		{"enabled", true},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("action", ""), "toggle_video_denoise");
	EXPECT_EQ(validated->value("enabled", false), true);
}

TEST(WsCommandValidationTest, ActionDrivenVsrppDisableCommand)
{
	// When user clicks BasicVSR++ off → toggle_video_denoise enabled=false
	// Daemon should restore LLM as highest priority
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_video_denoise"},
		{"enabled", false},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("enabled", true), false);
}

TEST(WsCommandValidationTest, AllNewActionTypesAccepted)
{
	// Comprehensive check that all new UI actions are accepted
	std::vector<nlohmann::json> commands = {
		{{"v", 1}, {"type", "ui_command"}, {"action", "select_conversation_model"}, {"model", "gemma_e4b"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "conversation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "sam2_segmentation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "vision_model"}},
	};

	for (const auto& cmd : commands) {
		auto validated = WebSocketBridgeNode::validateUiCommand(cmd);
		ASSERT_TRUE(validated.has_value())
			<< "Failed for action: " << cmd.value("action", "?")
			<< " — error: " << validated.error();
	}
}

// ---------------------------------------------------------------------------
// toggle_audio_denoise validation
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AcceptsToggleAudioDenoiseOn)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_audio_denoise"},
		{"enabled", true},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("action", ""), "toggle_audio_denoise");
	EXPECT_EQ(validated->value("enabled", false), true);
}

TEST(WsCommandValidationTest, RejectsToggleAudioDenoiseWithoutBoolean)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "toggle_audio_denoise"},
		{"enabled", "true"},
	});
	ASSERT_FALSE(validated.has_value());
}

// ---------------------------------------------------------------------------
// Full action-driven UI workflow: blur + denoise commands in sequence
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, ActionDrivenUiWorkflowSequence)
{
	// Simulate a complete user interaction:
	// 1. Enable blur (set_bg_mode blur)
	// 2. Enable BasicVSR++ (toggle_video_denoise true) — evicts LLM
	// 3. Select conversation model (select_conversation_model)
	// 4. Disable BasicVSR++ (toggle_video_denoise false) — LLM resumes
	// 5. Disable blur (set_bg_mode none)
	std::vector<nlohmann::json> commands = {
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "blur"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "toggle_video_denoise"}, {"enabled", true}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "select_conversation_model"}, {"model", "gemma_e2b"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "toggle_video_denoise"}, {"enabled", false}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "none"}},
	};

	for (const auto& cmd : commands) {
		auto validated = WebSocketBridgeNode::validateUiCommand(cmd);
		ASSERT_TRUE(validated.has_value())
			<< "Workflow step failed for action: " << cmd.value("action", "?")
			<< " — error: " << validated.error();
	}
}

// ---------------------------------------------------------------------------
// flush_tts validation
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AcceptsFlushTts)
{
	auto validated = WebSocketBridgeNode::validateUiCommand({
		{"v", 1}, {"type", "ui_command"},
		{"action", "flush_tts"},
	});
	ASSERT_TRUE(validated.has_value()) << validated.error();
	EXPECT_EQ(validated->value("action", ""), "flush_tts");
}

// ---------------------------------------------------------------------------
// Complete action coverage: verify every frontend action is accepted
// ---------------------------------------------------------------------------

TEST(WsCommandValidationTest, AllFrontendActionsAccepted)
{
	// Exhaustive list of all actions the frontend JS (controls.js) sends.
	// Any action missing from validateUiCommand's allow-list will fail here.
	std::vector<nlohmann::json> commands = {
		{{"v", 1}, {"type", "ui_command"}, {"action", "text_input"}, {"text", "hi"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "push_to_talk"}, {"state", true}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "describe_scene"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "describe_uploaded_image"}, {"image_path", "/tmp/x.jpg"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "register_face"}, {"name", "bob"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "cancel_generation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "tts_complete"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "flush_tts"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "conversation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "sam2_segmentation"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "switch_mode"}, {"mode", "vision_model"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "blur"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "set_bg_mode"}, {"mode", "none"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "select_conversation_model"}, {"model", "gemma_e4b"}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "toggle_video_denoise"}, {"enabled", true}},
		{{"v", 1}, {"type", "ui_command"}, {"action", "toggle_audio_denoise"}, {"enabled", true}},
	};

	for (const auto& cmd : commands) {
		auto validated = WebSocketBridgeNode::validateUiCommand(cmd);
		ASSERT_TRUE(validated.has_value())
			<< "Action rejected: " << cmd.value("action", "?")
			<< " — error: " << validated.error();
	}
}

