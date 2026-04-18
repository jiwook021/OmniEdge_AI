#include "ws_bridge/websocket_bridge_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <jpeglib.h>

#include "common/file.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/constants/cv_constants.hpp"
#include "common/constants/ingest_constants.hpp"
#include "common/constants/beauty_constants.hpp"
#include "common/constants/security_constants.hpp"
#include "common/constants/video_denoise_constants.hpp"
#include "common/constants/ws_bridge_constants.hpp"
#include "common/shm_tts_header.hpp"
#include "shm/shm_circular_buffer.hpp"


namespace {

[[nodiscard]] bool isActionWithoutExtraFields(std::string_view action) noexcept
{
	return action == "describe_scene"
		|| action == "cancel_generation"
		|| action == "flush_tts"
		|| action == "stop_playback"
		|| action == "tts_complete"
		|| action == "get_conversation_history"
		|| action == "clear_conversation_history";
}

constexpr std::array<std::string_view, 3> kReleaseModes = {
	"conversation", "security", "beauty"
};

constexpr std::array<std::string_view, 2> kLegacyCompatModes = {
	"sam2_segmentation", "vision_model"
};

template <std::size_t N>
[[nodiscard]] bool containsMode(
	std::string_view mode,
	const std::array<std::string_view, N>& modes) noexcept
{
	return std::find(modes.begin(), modes.end(), mode) != modes.end();
}

[[nodiscard]] bool isSupportedReleaseMode(std::string_view mode) noexcept
{
	return containsMode(mode, kReleaseModes);
}

[[nodiscard]] bool isSupportedLegacyMode(std::string_view mode) noexcept
{
	return containsMode(mode, kLegacyCompatModes);
}

/// JPEG magic bytes: FF D8 FF
constexpr uint8_t kJpegMagic[] = {0xFF, 0xD8, 0xFF};

/// PNG 8-byte signature: 89 50 4E 47 0D 0A 1A 0A
constexpr uint8_t kPngMagic[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

/// Minimum interval between successive publishes of the same high-frequency
/// ui_command action. A slider drag can emit ~60 msg/s; without coalescing,
/// low-frequency text_input / mode_switch commands queue behind that spam
/// on the ui_command ZMQ topic and observed chat latency balloons to >10 s.
constexpr std::chrono::milliseconds kUiCommandCoalesceWindow{50};

/// Actions whose payloads are idempotent "keep the latest, drop the rest."
/// Everything not in this set publishes unconditionally.
[[nodiscard]] bool isCoalescableUiCommand(std::string_view action) noexcept
{
	return action == "set_image_adjust";
}

/// Detect JPEG/PNG from magic bytes and return the file extension.
[[nodiscard]] std::string_view detectImageExtension(const uint8_t* data, std::size_t size) noexcept
{
	if (size >= sizeof(kJpegMagic)
	    && std::memcmp(data, kJpegMagic, sizeof(kJpegMagic)) == 0) {
		return ".jpg";
	}
	if (size >= sizeof(kPngMagic)
	    && std::memcmp(data, kPngMagic, sizeof(kPngMagic)) == 0) {
		return ".png";
	}
	return "";
}

[[nodiscard]] bool isSupportedMode(std::string_view mode) noexcept
{
	return isSupportedReleaseMode(mode) || isSupportedLegacyMode(mode);
}

[[nodiscard]] std::string supportedModesForError()
{
	std::string out;
	bool first = true;
	const auto appendMode = [&out, &first](std::string_view mode) {
		if (!first) out += ", ";
		out += mode;
		first = false;
	};

	for (const auto mode : kReleaseModes) {
		appendMode(mode);
	}
	for (const auto mode : kLegacyCompatModes) {
		appendMode(mode);
	}
	return out;
}

[[nodiscard]] std::unordered_set<std::string> topicsForMode(std::string_view mode)
{
	std::unordered_set<std::string> topics = {
		std::string(kZmqTopicModuleStatus),
		std::string(kZmqTopicOrchestratorReady),
		std::string(kZmqTopicConversationHistory),
	};

	if (mode == "conversation") {
		topics.insert({
			std::string(kZmqTopicVideoFrame),
			std::string(kZmqTopicBlurredFrame),
			std::string(kZmqTopicTtsAudio),
			std::string(kZmqTopicLlmResponse),
			std::string(kZmqTopicFaceRegistered),
			std::string(kZmqTopicDenoisedFrame),
			std::string(kZmqTopicDenoisedAudio),
			std::string(kZmqTopicScreenFrame),
			std::string(kZmqTopicScreenHealth),
		});
		return topics;
	}

	if (mode == "security") {
		topics.insert({
			std::string(kZmqTopicVideoFrame),
			std::string(kZmqTopicScreenHealth),
			std::string(kZmqTopicSecurityDetection),
			std::string(kZmqTopicSecurityEvent),
			std::string(kZmqTopicSecurityRecordingStatus),
			std::string(kZmqTopicSecurityVlmAnalysis),
		});
		return topics;
	}

	if (mode == "beauty") {
		topics.insert({
			std::string(kZmqTopicVideoFrame),
			std::string(kZmqTopicBeautyFrame),
			std::string(kZmqTopicScreenHealth),
		});
		return topics;
	}

	// Legacy internal compatibility modes.
	if (mode == "sam2_segmentation") {
		topics.insert({
			std::string(kZmqTopicVideoFrame),
			std::string(kZmqTopicSegmentationMask),
			std::string(kZmqTopicScreenHealth),
		});
		return topics;
	}

	if (mode == "vision_model") {
		topics.insert({
			std::string(kZmqTopicVideoFrame),
			std::string(kZmqTopicLlmResponse),
			std::string(kZmqTopicTtsAudio),
			std::string(kZmqTopicScreenFrame),
			std::string(kZmqTopicScreenHealth),
		});
		return topics;
	}

	// Unknown mode: keep control-plane visibility instead of dropping all.
	return topics;
}

} // namespace

WebSocketBridgeNode::WebSocketBridgeNode(const Config& config)
	: config_(config)
	, messageRouter_(MessageRouter::Config{
		.moduleName  = config.moduleName,
		.pubPort     = config.wsBridgePubPort,
		.pubHwm      = kPublisherControlHighWaterMark,
		.pollTimeout = config.pollTimeout,
	})
{}

WebSocketBridgeNode::~WebSocketBridgeNode()
{
	stop();
}

void WebSocketBridgeNode::setRelay(std::unique_ptr<IWsRelay> relay)
{
	relay_ = std::move(relay);
}

void WebSocketBridgeNode::initialize()
{
	if (!relay_) {
		throw std::runtime_error("WebSocketBridgeNode::initialize: relay not set");
	}

	OeLogger::instance().setModule(config_.moduleName);

	// Register a handler for each upstream topic. The handlers run on the
	// relay thread (inside messageRouter_.run()) and forward to the uWS event loop
	// via relay_->broadcastVideo/Audio/Json.
	auto makeHandler = [this](std::string topic) -> std::function<void(const nlohmann::json&)> {
		return [this, topic = std::move(topic)](const nlohmann::json& payload) {
			// Check if this topic is currently active (mode filtering).
			{
				const std::lock_guard<std::mutex> lock(activeMutex_);
				if (activeTopics_.find(topic) == activeTopics_.end()) {
					return;
				}
			}
			relayTopicPayload_(topic, payload);
		};
	};

	messageRouter_.subscribe(config_.blurSubPort,        kZmqTopicBlurredFrame,   true,
	                  makeHandler(std::string(kZmqTopicBlurredFrame)));
	messageRouter_.subscribe(config_.videoIngestSubPort,  kZmqTopicVideoFrame,    true,
	                  makeHandler(std::string(kZmqTopicVideoFrame)));
	messageRouter_.subscribe(config_.ttsSubPort,          kZmqTopicTtsAudio,      false,
	                  makeHandler(std::string(kZmqTopicTtsAudio)));
	// Gemma-4 conversation model publishes llm_response. llmSubPort and
	// conversationSubPort resolve to the same kConversationModel port today;
	// subscribing twice delivers every token twice.
	messageRouter_.subscribe(config_.conversationSubPort, kZmqTopicLlmResponse,   false,
	                  makeHandler(std::string(kZmqTopicLlmResponse)));
	messageRouter_.subscribe(config_.daemonSubPort,       kZmqTopicModuleStatus,  false,
	                  makeHandler(std::string(kZmqTopicModuleStatus)));
	messageRouter_.subscribe(config_.daemonSubPort,       kZmqTopicConversationHistory, false,
	                  makeHandler(std::string(kZmqTopicConversationHistory)));
	messageRouter_.subscribe(config_.daemonSubPort,       kZmqTopicOrchestratorReady, false,
	                  makeHandler(std::string(kZmqTopicOrchestratorReady)));
	messageRouter_.subscribe(config_.faceSubPort,         kZmqTopicFaceRegistered, false,
	                  makeHandler(std::string(kZmqTopicFaceRegistered)));
	messageRouter_.subscribe(config_.videoDenoiseSubPort, kZmqTopicDenoisedFrame, true,
	                  makeHandler(std::string(kZmqTopicDenoisedFrame)));
	messageRouter_.subscribe(config_.audioDenoiseSubPort, kZmqTopicDenoisedAudio, false,
	                  makeHandler(std::string(kZmqTopicDenoisedAudio)));
	messageRouter_.subscribe(config_.sam2SubPort, kZmqTopicSegmentationMask, false,
	                  makeHandler(std::string(kZmqTopicSegmentationMask)));
	messageRouter_.subscribe(config_.screenIngestSubPort, kZmqTopicScreenFrame, true,
	                  makeHandler(std::string(kZmqTopicScreenFrame)));
	messageRouter_.subscribe(config_.screenIngestSubPort, kZmqTopicScreenHealth, false,
	                  makeHandler(std::string(kZmqTopicScreenHealth)));

	// Security camera — detection events + recording status (JSON → /chat),
	// annotated JPEG handled via SHM read in relayTopicPayload_.
	messageRouter_.subscribe(config_.securityCameraSubPort, kZmqTopicSecurityDetection, false,
	                  makeHandler(std::string(kZmqTopicSecurityDetection)));
	messageRouter_.subscribe(config_.securityCameraSubPort, kZmqTopicSecurityEvent, false,
	                  makeHandler(std::string(kZmqTopicSecurityEvent)));
	messageRouter_.subscribe(config_.securityCameraSubPort, kZmqTopicSecurityRecordingStatus, false,
	                  makeHandler(std::string(kZmqTopicSecurityRecordingStatus)));
	messageRouter_.subscribe(config_.securityCameraSubPort, kZmqTopicSecurityVlmAnalysis, false,
	                  makeHandler(std::string(kZmqTopicSecurityVlmAnalysis)));

	// Beauty — JPEG frames from BeautyNode via SHM.
	messageRouter_.subscribe(config_.beautySubPort, kZmqTopicBeautyFrame, true,
	                  makeHandler(std::string(kZmqTopicBeautyFrame)));

	if (!isSupportedMode(config_.startupMode)) {
		OE_LOG_WARN("ws_bridge_invalid_startup_mode: mode='{}' fallback='conversation'",
		            config_.startupMode);
		setMode_("conversation");
	} else {
		setMode_(config_.startupMode);
	}

	messageRouter_.publishModuleReady();

	OE_LOG_INFO("ws_bridge_initialized: pub_port={}, mode={}",
		config_.wsBridgePubPort, currentMode_);
}

void WebSocketBridgeNode::run()
{
	running_.store(true, std::memory_order_release);

	// messageRouter_.run() is blocking — run it on the relay thread.
	// The handlers registered in initialize() execute on this thread.
	std::thread relayThread([this]() {
		messageRouter_.run();
	});

	// uWebSockets event loop runs on the main thread (blocks until stop).
	relay_->run();

	// uWS loop exited — shut down the relay thread.
	running_.store(false, std::memory_order_release);
	messageRouter_.stop();

	if (relayThread.joinable()) {
		relayThread.join();
	}
}

void WebSocketBridgeNode::stop() noexcept
{
	running_.store(false, std::memory_order_release);
	try {
		messageRouter_.stop();
	} catch (const std::exception& e) {
		OE_LOG_WARN("OE-WS-2010 ws_bridge_stop_router_swallowed: {}", e.what());
	}
	if (relay_) {
		try {
			relay_->stop();
		} catch (const std::exception& e) {
			OE_LOG_WARN("OE-WS-2011 ws_bridge_stop_relay_swallowed: {}", e.what());
		}
	}
}

tl::expected<nlohmann::json, std::string>
WebSocketBridgeNode::validateUiCommand(const nlohmann::json& rawCommand)
{
	if (!rawCommand.is_object()) {
		return tl::unexpected(std::string("ui_command must be a JSON object"));
	}

	nlohmann::json cmd = rawCommand;
	cmd["v"] = cmd.value("v", kSchemaVersion);
	cmd["type"] = "ui_command";

	if (!cmd.contains("action") || !cmd["action"].is_string()) {
		return tl::unexpected(std::string("ui_command.action must be a string"));
	}

	const std::string action = cmd["action"].get<std::string>();

	if (action == "push_to_talk") {
		if (!cmd.contains("state") || !cmd["state"].is_boolean()) {
			return tl::unexpected(std::string("push_to_talk requires boolean state"));
		}
		return cmd;
	}
	if (action == "text_input") {
		if (!cmd.contains("text") || !cmd["text"].is_string()) {
			return tl::unexpected(std::string("text_input requires string text"));
		}
		return cmd;
	}
	if (action == "register_face") {
		if (!cmd.contains("name") || !cmd["name"].is_string()) {
			return tl::unexpected(std::string("register_face requires string name"));
		}
		return cmd;
	}
	if (action == "switch_mode") {
		if (!cmd.contains("mode") || !cmd["mode"].is_string()) {
			return tl::unexpected(std::string("switch_mode requires string mode"));
		}
		const std::string mode = cmd["mode"].get<std::string>();
		if (!isSupportedMode(mode)) {
			return tl::unexpected(std::format(
				"switch_mode.mode not supported: {} (supported: {})",
				mode, supportedModesForError()));
		}
		return cmd;
	}
	if (action == "switch_boot_mode") {
		if (!cmd.contains("mode") || !cmd["mode"].is_string()) {
			return tl::unexpected(std::string("switch_boot_mode requires string mode"));
		}
		const std::string bootMode = cmd["mode"].get<std::string>();
		if (bootMode != "simple_llm" && bootMode != "full_conversation") {
			return tl::unexpected(std::string("switch_boot_mode.mode must be \"simple_llm\" or \"full_conversation\""));
		}
		return cmd;
	}
	if (action == "set_image_adjust") {
		if (!cmd.contains("brightness") || !cmd.contains("contrast")
		 || !cmd.contains("saturation") || !cmd.contains("sharpness")) {
			return tl::unexpected(std::string("set_image_adjust requires brightness/contrast/saturation/sharpness"));
		}
		return cmd;
	}
	if (action == "set_bg_mode") {
		if (!cmd.contains("mode") || !cmd["mode"].is_string()) {
			return tl::unexpected(std::string("set_bg_mode requires string mode"));
		}
		const std::string bgMode = cmd["mode"].get<std::string>();
		if (bgMode != "blur" && bgMode != "none") {
			return tl::unexpected(std::string("set_bg_mode.mode must be \"blur\" or \"none\""));
		}
		return cmd;
	}
	if (action == "update_shapes") {
		if (!cmd.contains("shapes") || !cmd["shapes"].is_array()) {
			return tl::unexpected(std::string("update_shapes requires array shapes"));
		}
		return cmd;
	}
	if (action == "toggle_video_denoise" || action == "toggle_audio_denoise"
	    || action == "toggle_stt" || action == "toggle_tts"
	    || action == "toggle_face_recognition" || action == "toggle_background_blur"
	    || action == "toggle_sam2" || action == "toggle_face_filter"
	    || action == "toggle_video_conversation" || action == "toggle_beauty"
	    || action == "toggle_security_mode") {
		if (!cmd.contains("enabled") || !cmd["enabled"].is_boolean()) {
			return tl::unexpected(std::format("{} requires boolean enabled", action));
		}
		return cmd;
	}
	if (action == "select_conversation_model") {
		if (!cmd.contains("model") || !cmd["model"].is_string()) {
			return tl::unexpected(std::string("select_conversation_model requires string model"));
		}
		return cmd;
	}
	if (action == "select_conversation_source") {
		if (!cmd.contains("source") || !cmd["source"].is_string()) {
			return tl::unexpected(std::string("select_conversation_source requires string source"));
		}
		const auto src = cmd["source"].get<std::string>();
		if (src != "camera" && src != "screen") {
			return tl::unexpected(std::string("select_conversation_source.source must be \"camera\" or \"screen\""));
		}
		return cmd;
	}
	if (action == "describe_uploaded_image") {
		if (!cmd.contains("image_path") || !cmd["image_path"].is_string()) {
			return tl::unexpected(std::string("describe_uploaded_image requires string image_path"));
		}
		return cmd;
	}
	if (action == "sam2_segment_point") {
		if (!cmd.contains("x") || !cmd.contains("y")) {
			return tl::unexpected(std::string("sam2_segment_point requires x and y"));
		}
		return cmd;
	}
	if (action == "sam2_segment_box") {
		if (!cmd.contains("x1") || !cmd.contains("y1")
		 || !cmd.contains("x2") || !cmd.contains("y2")) {
			return tl::unexpected(std::string("sam2_segment_box requires x1, y1, x2, y2"));
		}
		return cmd;
	}
	if (action == "sam2_segment_mask") {
		return cmd;
	}
	if (action == "sam2_stop_tracking") {
		return cmd;
	}
	if (action == "set_sam2_bg_mode") {
		if (!cmd.contains("mode") || !cmd["mode"].is_string()) {
			return tl::unexpected(std::string("set_sam2_bg_mode requires string mode"));
		}
		return cmd;
	}
	if (action == "set_sam2_bg_color") {
		if (!cmd.contains("r") || !cmd.contains("g") || !cmd.contains("b")) {
			return tl::unexpected(std::string("set_sam2_bg_color requires r, g, b"));
		}
		if (!cmd["r"].is_number() || !cmd["g"].is_number() || !cmd["b"].is_number()) {
			return tl::unexpected(std::string("set_sam2_bg_color r/g/b must be numbers"));
		}
		const int r = cmd["r"].get<int>();
		const int g = cmd["g"].get<int>();
		const int b = cmd["b"].get<int>();
		if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) {
			return tl::unexpected(std::string("set_sam2_bg_color r/g/b must be in [0, 255]"));
		}
		return cmd;
	}
	if (action == "set_sam2_bg_image") {
		if (!cmd.contains("image_path") || !cmd["image_path"].is_string()) {
			return tl::unexpected(std::string("set_sam2_bg_image requires string image_path"));
		}
		return cmd;
	}

	// Beauty family: preset + slider groups. Payload schemas are owned by the
	// beauty node; ws_bridge just forwards.
	if (action == "set_beauty_preset" || action == "set_beauty_skin"
	    || action == "set_beauty_shape" || action == "set_beauty_light"
	    || action == "set_beauty_bg") {
		return cmd;
	}

	// Security family: roi/classes/style/recordings/analyze. Owned by the
	// security_vlm node.
	if (action == "security_list_recordings" || action == "security_update_classes"
	    || action == "security_set_style" || action == "security_vlm_analyze"
	    || action == "security_set_roi") {
		return cmd;
	}

	// VLM explorer: point/region prompts.
	if (action == "vlm_describe_point") {
		if (!cmd.contains("x") || !cmd.contains("y")) {
			return tl::unexpected(std::string("vlm_describe_point requires x and y"));
		}
		return cmd;
	}
	if (action == "vlm_describe_region") {
		if (!cmd.contains("x1") || !cmd.contains("y1")
		 || !cmd.contains("x2") || !cmd.contains("y2")) {
			return tl::unexpected(std::string("vlm_describe_region requires x1, y1, x2, y2"));
		}
		return cmd;
	}

	// Siri mode emits ptt_start/ptt_stop as interim states around push_to_talk.
	if (action == "ptt_start" || action == "ptt_stop") {
		return cmd;
	}

	if (isActionWithoutExtraFields(action)) {
		return cmd;
	}

	return tl::unexpected(std::format("unsupported ui_command action: {}", action));
}

void WebSocketBridgeNode::handleClientCommand(const nlohmann::json& rawCommand)
{
	OE_ZONE_SCOPED;
	auto validated = validateUiCommand(rawCommand);
	if (!validated) {
		OE_LOG_WARN("OE-WS-2100 ws_ui_command_rejected: reason={}", validated.error());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", validated.error()},
		});
		return;
	}

	nlohmann::json command = std::move(validated.value());
	const std::string action = command["action"].get<std::string>();

	if (isCoalescableUiCommand(action)) {
		const auto now = std::chrono::steady_clock::now();
		auto& last = lastUiCommandPublishTime_[action];
		if (last.time_since_epoch().count() != 0
		    && (now - last) < kUiCommandCoalesceWindow) {
			OE_LOG_DEBUG("ws_ui_command_coalesced: action={}", action);
			return;
		}
		last = now;
	}

	OE_LOG_DEBUG("ws_ui_command_forwarded: action={}", action);

	if (action == "switch_mode") {
		setMode_(command["mode"].get<std::string>());
	}

	// publish() touches only the PUB socket, which is exclusively used from
	// this (uWS main) thread — safe despite messageRouter_.run() on the relay thread.
	messageRouter_.publish(kZmqTopicUiCommand, std::move(command));
}

void WebSocketBridgeNode::handleBinaryUpload(std::string_view binaryFrame)
{
	OE_ZONE_SCOPED;
	// Binary frame format:
	//   [4 bytes: JSON header length, little-endian uint32]
	//   [N bytes: UTF-8 JSON: {"action":"describe_uploaded_image",...}]
	//   [remaining bytes: raw JPEG/PNG image data]

	constexpr std::size_t kHeaderSize = kImageUploadHeaderSize;

	if (binaryFrame.size() < kHeaderSize) {
		OE_LOG_WARN("OE-WS-2101 ws_binary_upload_too_small: size={}", binaryFrame.size());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", "Binary upload too small"},
		});
		return;
	}

	// Read JSON header length (little-endian uint32)
	uint32_t jsonLen = 0;
	std::memcpy(&jsonLen, binaryFrame.data(), sizeof(uint32_t));

	if (jsonLen == 0 || static_cast<size_t>(kHeaderSize) + static_cast<size_t>(jsonLen) > binaryFrame.size()) {
		OE_LOG_WARN("OE-WS-2102 ws_binary_upload_bad_header: json_len={}, total={}", jsonLen, binaryFrame.size());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", "Invalid binary upload header"},
		});
		return;
	}

	// Parse JSON header
	nlohmann::json meta;
	try {
		meta = nlohmann::json::parse(
			binaryFrame.data() + kHeaderSize,
			binaryFrame.data() + kHeaderSize + jsonLen);
	} catch (const nlohmann::json::exception& ex) {
		OE_LOG_WARN("OE-WS-2103 ws_binary_upload_json_error: error={}", ex.what());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", std::format("Upload JSON parse error: {}", ex.what())},
		});
		return;
	}

	const std::string action = meta.value("action", std::string{});
	if (action != "describe_uploaded_image" && action != "set_sam2_bg_image") {
		OE_LOG_WARN("OE-WS-2104 ws_binary_upload_unknown_action: action={}", action);
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", std::format("Unknown binary upload action: {}", action)},
		});
		return;
	}

	// Extract image data
	const std::size_t imageOffset = kHeaderSize + jsonLen;
	const std::size_t imageSize   = binaryFrame.size() - imageOffset;
	const auto* imageData = reinterpret_cast<const uint8_t*>(binaryFrame.data() + imageOffset);

	if (imageSize == 0 || imageSize > kMaxImageUploadBytes) {
		OE_LOG_WARN("OE-WS-2105 ws_binary_upload_bad_size: image_bytes={}", imageSize);
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", std::format("Image size invalid: {} bytes (max {})",
				imageSize, kMaxImageUploadBytes)},
		});
		return;
	}

	// Detect image format from magic bytes
	const auto ext = detectImageExtension(imageData, imageSize);
	if (ext.empty()) {
		OE_LOG_WARN("OE-WS-2106 ws_binary_upload_unknown_format: first_byte=0x{:02x}", imageData[0]);
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", "Unsupported image format — only JPEG and PNG accepted"},
		});
		return;
	}

	// Write to temp file with unique name
	const uint64_t uploadId = ++uploadCounter_;
	const std::string tempPath = std::format("{}/oe_upload_{}{}",
		kUploadTempDir, uploadId, ext);

	{
		auto writeResult = writeBinary(
			tempPath,
			std::span<const uint8_t>(imageData, imageSize));
		if (!writeResult) {
			OE_LOG_WARN("OE-WS-2107 ws_binary_upload_write_failed: path={}, error={}",
			            tempPath, writeResult.error());
			relay_->broadcastJson({
				{"v", kSchemaVersion},
				{"type", "error"},
				{"message", "Failed to write uploaded image to disk"},
			});
			return;
		}
	}

	OE_LOG_INFO("ws_image_uploaded: path={}, size={}, ext={}, action={}",
		tempPath, imageSize, ext, action);

	// Build and forward the ui_command
	nlohmann::json command = {
		{"v",          kSchemaVersion},
		{"type",       "ui_command"},
		{"action",     action},
		{"image_path", tempPath},
	};

	messageRouter_.publish(kZmqTopicUiCommand, std::move(command));
}

void WebSocketBridgeNode::setMode_(std::string_view mode)
{
	// activeTopics_ and currentMode_ are read by messageRouter_.run() handlers
	// on the relay thread: guard the write.
	{
		const std::lock_guard<std::mutex> lock(activeMutex_);
		currentMode_ = std::string(mode);
		activeTopics_ = topicsForMode(mode);
	}
	OE_LOG_INFO("ws_bridge_mode_set: mode={}", currentMode_);
}

bool WebSocketBridgeNode::shouldPace(
	std::chrono::steady_clock::time_point lastSend,
	int pacingMs) noexcept
{
	const auto now = std::chrono::steady_clock::now();
	const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		now - lastSend);
	return elapsed.count() >= pacingMs;
}

void WebSocketBridgeNode::relayTopicPayload_(std::string_view topic,
                                             const nlohmann::json& payload)
{
	if (topic == kZmqTopicBlurredFrame) {
		auto frame = readBlurredFrameFromShm_(payload);
		if (!frame.empty()) {
			// Frame pacing: only send if enough time has elapsed
			if (shouldPace(lastBlurSendTime_, kBlurFramePacingMs)) {
				lastBlurFrame_ = std::move(frame);  // move to cache (no copy)
				lastBlurSendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastBlurVideo(
					std::vector<uint8_t>(lastBlurFrame_.begin(), lastBlurFrame_.end()));
			} else {
				// Multiple frames arrived between ticks — skip to newest (cache it)
				lastBlurFrame_ = std::move(frame);
			}
		} else if (shouldPace(lastBlurSendTime_, kBlurFramePacingMs)
		           && !lastBlurFrame_.empty()) {
			// No new frame at this tick — re-send cached frame (avoid copy)
			lastBlurSendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastBlurVideo(
				std::vector<uint8_t>(lastBlurFrame_.begin(), lastBlurFrame_.end()));
		}
		return;
	}
	if (topic == kZmqTopicVideoFrame) {
		// Raw camera feed — always relayed on /video. Downstream CV overlays
		// (blur, denoise, beauty, sam2, security, filter) publish to their own
		// dedicated channels and must not suppress the raw tile.
		if (!shouldPace(lastVideoSendTime_, kBlurFramePacingMs)) {
			return;
		}
		auto frame = readRawVideoFrameFromShm_(payload);
		if (!frame.empty()) {
			lastVideoSendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastVideo(std::move(frame));
		}
		return;
	}
	if (topic == kZmqTopicDenoisedFrame) {
		auto frame = readDenoisedFrameFromShm_(payload);
		if (!frame.empty()) {
			// Frame pacing for denoise channel
			if (shouldPace(lastDenoiseSendTime_, kDenoiseFramePacingMs)) {
				lastDenoiseFrame_ = std::move(frame);  // move to cache (no copy)
				lastDenoiseSendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastDenoisedVideo(
					std::vector<uint8_t>(lastDenoiseFrame_.begin(), lastDenoiseFrame_.end()));
			} else {
				lastDenoiseFrame_ = std::move(frame);
			}
		} else if (shouldPace(lastDenoiseSendTime_, kDenoiseFramePacingMs)
		           && !lastDenoiseFrame_.empty()) {
			lastDenoiseSendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastDenoisedVideo(std::vector<uint8_t>(lastDenoiseFrame_));
		}
		return;
	}
	if (topic == kZmqTopicDenoisedAudio) {
		auto pcm = readDenoisedAudioFromShm_(payload);
		if (!pcm.empty()) {
			relay_->broadcastDenoisedAudio(std::move(pcm));
		}
		return;
	}
	if (topic == kZmqTopicTtsAudio) {
		auto pcm = readTtsAudioFromShm_(payload);
		if (!pcm.empty()) {
			relay_->broadcastAudio(std::move(pcm));
		}
		return;
	}
	if (topic == kZmqTopicSegmentationMask) {
		// SAM2 continuous tracking: apply frame pacing at 33ms (~30fps).
		auto mask = readSam2MaskFromShm_(payload);
		if (!mask.empty()) {
			if (shouldPace(lastSam2SendTime_, kSam2FramePacingMs)) {
				lastSam2Frame_ = std::move(mask);
				lastSam2SendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastSam2Video(
					std::vector<uint8_t>(lastSam2Frame_.begin(), lastSam2Frame_.end()));
			} else {
				lastSam2Frame_ = std::move(mask);
			}
		} else if (shouldPace(lastSam2SendTime_, kSam2FramePacingMs)
		           && !lastSam2Frame_.empty()) {
			lastSam2SendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastSam2Video(
				std::vector<uint8_t>(lastSam2Frame_.begin(), lastSam2Frame_.end()));
		}
		// Also forward the JSON metadata (iou_score, stability, tracking) to /chat.
		relay_->broadcastJson(payload);
		return;
	}

	// Screen frame: read BGR24 from SHM, JPEG-compress, send to /screen_video.
	if (topic == kZmqTopicScreenFrame) {
		auto frame = readScreenFrameFromShm_(payload);
		if (!frame.empty()) {
			relay_->broadcastScreenVideo(std::move(frame));
		}
		return;
	}
	// Screen health: forward JSON to /chat (UI shows error banner).
	if (topic == kZmqTopicScreenHealth) {
		relay_->broadcastJson(payload);
		return;
	}

	// Security camera: read annotated JPEG from SHM → /security_video,
	// and forward the JSON detection/event data to /chat.
	if (topic == kZmqTopicSecurityDetection) {
		// Read annotated JPEG from SHM and broadcast to /security_video.
		auto frame = readSecurityFrameFromShm_(payload);
		if (!frame.empty()) {
			if (shouldPace(lastSecuritySendTime_, kSecurityFramePacingMs)) {
				lastSecurityFrame_ = std::move(frame);
				lastSecuritySendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastSecurityVideo(
					std::vector<uint8_t>(lastSecurityFrame_.begin(), lastSecurityFrame_.end()));
			} else {
				lastSecurityFrame_ = std::move(frame);
			}
		} else if (shouldPace(lastSecuritySendTime_, kSecurityFramePacingMs)
		           && !lastSecurityFrame_.empty()) {
			lastSecuritySendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastSecurityVideo(
				std::vector<uint8_t>(lastSecurityFrame_.begin(), lastSecurityFrame_.end()));
		}
		// Also forward the JSON detection data to /chat.
		relay_->broadcastJson(payload);
		return;
	}
	if (topic == kZmqTopicSecurityEvent
	    || topic == kZmqTopicSecurityRecordingStatus
	    || topic == kZmqTopicSecurityVlmAnalysis) {
		relay_->broadcastJson(payload);
		return;
	}

	// Beauty: read processed JPEG from SHM → /beauty_video.
	if (topic == kZmqTopicBeautyFrame) {
		auto frame = readBeautyFrameFromShm_(payload);
		if (!frame.empty()) {
			if (shouldPace(lastBeautySendTime_, kBeautyFramePacingMs)) {
				lastBeautyFrame_ = std::move(frame);
				lastBeautySendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastBeautyVideo(
					std::vector<uint8_t>(lastBeautyFrame_.begin(), lastBeautyFrame_.end()));
			} else {
				lastBeautyFrame_ = std::move(frame);
			}
		} else if (shouldPace(lastBeautySendTime_, kBeautyFramePacingMs)
		           && !lastBeautyFrame_.empty()) {
			lastBeautySendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastBeautyVideo(
				std::vector<uint8_t>(lastBeautyFrame_.begin(), lastBeautyFrame_.end()));
		}
		return;
	}

	relay_->broadcastJson(payload);
}

std::vector<uint8_t> WebSocketBridgeNode::readBlurredFrameFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		OE_LOG_DEBUG("shm_blur_cache_miss: name={}, size={}", shmName, kJpegShmSegmentByteSize);
		auto mapping = std::make_unique<ShmMapping>(shmName, kJpegShmSegmentByteSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	auto* ctrl = reinterpret_cast<const ShmJpegControl*>(it->second->bytes());
	const uint32_t slot = ctrl->writeIndex.load(std::memory_order_acquire);
	if (slot > 1) return {};  // double-buffer: only slots 0 and 1 are valid
	const uint32_t size = ctrl->jpegSize[slot];
	OE_LOG_DEBUG("shm_blur_read: slot={}, jpeg_size={}", slot, size);
	if (size == 0 || size > kMaxJpegBytesPerSlot) {
		return {};
	}

	const std::size_t offset = sizeof(ShmJpegControl)
		+ static_cast<std::size_t>(slot) * kMaxJpegBytesPerSlot;
	const uint8_t* src = it->second->bytes() + offset;
	std::vector<uint8_t> result(src, src + size);

	// Stale-read guard: if writer flipped during our read, discard
	if (ctrl->writeIndex.load(std::memory_order_acquire) != slot) return {};

	return result;
}

// ---------------------------------------------------------------------------
// Raw BGR24 → JPEG fallback (used when BackgroundBlurNode is not running)
// ---------------------------------------------------------------------------

// Helper: JPEG-compress a raw BGR24 buffer.  Isolated from STL containers to
// avoid -Wclobbered warnings (longjmp + C++ RAII don't mix).
static std::vector<uint8_t> compressBgrToJpeg(
	const uint8_t* bgr, uint32_t w, uint32_t h)
{
	struct OeJpegErrorMgr {
		jpeg_error_mgr pub;
		jmp_buf setjmpBuffer;
	};
	static const auto jpegErrorExit = [](j_common_ptr cinfo) {
		auto* myerr = reinterpret_cast<OeJpegErrorMgr*>(cinfo->err);
		longjmp(myerr->setjmpBuffer, 1);
	};

	struct jpeg_compress_struct cinfo{};
	OeJpegErrorMgr jerr{};
	cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = jpegErrorExit;

	uint8_t* outBuf = nullptr;
	unsigned long outSize = 0;

	if (setjmp(jerr.setjmpBuffer)) {
		jpeg_destroy_compress(&cinfo);
		std::free(outBuf);  // P0 fix: free jpeg_mem_dest buffer on error path
		return {};
	}

	jpeg_create_compress(&cinfo);

	jpeg_mem_dest(&cinfo, &outBuf, &outSize);

	cinfo.image_width  = w;
	cinfo.image_height = h;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_EXT_BGR;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 75, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	while (cinfo.next_scanline < cinfo.image_height) {
		auto* row = const_cast<uint8_t*>(bgr + cinfo.next_scanline * w * 3u);
		jpeg_write_scanlines(&cinfo, &row, 1);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	std::vector<uint8_t> jpeg(outBuf, outBuf + outSize);
	std::free(outBuf);
	return jpeg;
}

std::vector<uint8_t> WebSocketBridgeNode::readRawVideoFrameFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	// Video SHM layout: [ShmVideoHeader][ShmCircularControl][slot0]...[slotN-1]
	constexpr uint32_t w = kMaxInputWidth;
	constexpr uint32_t h = kMaxInputHeight;
	constexpr std::size_t frameBytes = static_cast<std::size_t>(w) * h * 3u;
	const std::size_t shmSize = ShmCircularBuffer<ShmVideoHeader>::segmentSize(
		kCircularBufferSlotCount, frameBytes);

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		auto mapping = std::make_unique<ShmMapping>(shmName, shmSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	// Read latest slot via ShmCircularReader
	ShmCircularReader reader(*it->second, sizeof(ShmVideoHeader));
	const auto slot = reader.latestSlotIndex();
	if (!slot) return {};   // no frames yet
	const uint8_t* bgr = reader.slotData(*slot);

	// JPEG-compress in isolated scope (no STL containers in longjmp range)
	return compressBgrToJpeg(bgr, w, h);
}

std::vector<uint8_t> WebSocketBridgeNode::readTtsAudioFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		auto mapping = std::make_unique<ShmMapping>(shmName, kTtsShmSegmentByteSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	auto* ttsHeader = reinterpret_cast<const ShmTtsHeader*>(it->second->bytes());

	// Stale-read guard: snapshot seqNumber before copying data
	const uint64_t seqBefore = ttsHeader->seqNumber;

	const std::size_t bytes = static_cast<std::size_t>(ttsHeader->numSamples) * sizeof(float);
	OE_LOG_DEBUG("shm_tts_read: samples={}, sample_rate={}, bytes={}",
	           ttsHeader->numSamples, ttsHeader->sampleRateHz, bytes);
	if (bytes == 0 || bytes > (it->second->size() - sizeof(ShmTtsHeader))) {
		return {};
	}

	const uint8_t* src = it->second->bytes() + sizeof(ShmTtsHeader);
	std::vector<uint8_t> result(src, src + bytes);

	// Stale-read guard: discard if producer overwrote during our copy
	const uint64_t seqAfter = ttsHeader->seqNumber;
	if (seqAfter != seqBefore) return {};

	return result;
}

std::vector<uint8_t> WebSocketBridgeNode::readDenoisedFrameFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		auto mapping = std::make_unique<ShmMapping>(shmName, kJpegShmSegmentByteSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	auto* ctrl = reinterpret_cast<const ShmJpegControl*>(it->second->bytes());
	const uint32_t slot = ctrl->writeIndex.load(std::memory_order_acquire);
	if (slot > 1) return {};  // double-buffer: only slots 0 and 1 are valid
	const uint32_t size = ctrl->jpegSize[slot];
	if (size == 0 || size > kMaxJpegBytesPerSlot) {
		return {};
	}

	const std::size_t offset = sizeof(ShmJpegControl)
		+ static_cast<std::size_t>(slot) * kMaxJpegBytesPerSlot;
	const uint8_t* src = it->second->bytes() + offset;
	std::vector<uint8_t> result(src, src + size);

	// Stale-read guard: if writer flipped during our read, discard
	if (ctrl->writeIndex.load(std::memory_order_acquire) != slot) return {};

	return result;
}

std::vector<uint8_t> WebSocketBridgeNode::readDenoisedAudioFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		auto mapping = std::make_unique<ShmMapping>(
			shmName, kDenoiseAudioShmSegmentByteSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	auto* hdr = reinterpret_cast<const ShmAudioHeader*>(it->second->bytes());

	// Stale-read guard: snapshot seqNumber before copying data
	const uint64_t seqBefore = hdr->seqNumber;

	const std::size_t bytes =
		static_cast<std::size_t>(hdr->numSamples) * sizeof(float);
	if (bytes == 0 || bytes > (it->second->size() - sizeof(ShmAudioHeader))) {
		return {};
	}

	const uint8_t* src = it->second->bytes() + sizeof(ShmAudioHeader);
	std::vector<uint8_t> result(src, src + bytes);

	// Stale-read guard: discard if producer overwrote during our copy
	const uint64_t seqAfter = hdr->seqNumber;
	if (seqAfter != seqBefore) return {};

	return result;
}

std::vector<uint8_t> WebSocketBridgeNode::readSam2MaskFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		OE_LOG_DEBUG("shm_sam2_cache_miss: name={}, size={}", shmName, kJpegShmSegmentByteSize);
		auto mapping = std::make_unique<ShmMapping>(shmName, kJpegShmSegmentByteSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	}

	auto* ctrl = reinterpret_cast<const ShmJpegControl*>(it->second->bytes());
	const uint32_t slot = ctrl->writeIndex.load(std::memory_order_acquire);
	if (slot > 1) return {};  // double-buffer: only slots 0 and 1 are valid
	const uint32_t size = ctrl->jpegSize[slot];
	if (size == 0 || size > kMaxJpegBytesPerSlot) {
		return {};
	}

	const std::size_t offset = sizeof(ShmJpegControl)
		+ static_cast<std::size_t>(slot) * kMaxJpegBytesPerSlot;
	const uint8_t* src = it->second->bytes() + offset;
	std::vector<uint8_t> result(src, src + size);

	// Stale-read guard: if writer flipped during our read, discard
	if (ctrl->writeIndex.load(std::memory_order_acquire) != slot) return {};

	return result;
}

std::vector<uint8_t> WebSocketBridgeNode::readScreenFrameFromShm_(
	const nlohmann::json& payload)
{
	if (!payload.contains("shm") || !payload["shm"].is_string()) {
		return {};
	}
	const std::string shmName = payload["shm"].get<std::string>();

	// Screen SHM carries width/height in the ZMQ notification.
	const uint32_t w = payload.value("w", 0u);
	const uint32_t h = payload.value("h", 0u);
	if (w == 0 || h == 0 || w > 7680 || h > 4320) {
		return {};
	}

	const std::size_t frameBytes = static_cast<std::size_t>(w) * h * 3u;
	const std::size_t shmSize = ShmCircularBuffer<ShmVideoHeader>::segmentSize(
		kCircularBufferSlotCount, frameBytes);

	auto it = shmCache_.find(shmName);
	if (it == shmCache_.end()) {
		auto mapping = std::make_unique<ShmMapping>(shmName, shmSize, false);
		it = shmCache_.emplace(shmName, std::move(mapping)).first;
	} else if (it->second->size() != shmSize) {
		// Resolution changed — ScreenIngestNode destroyed and recreated the SHM
		// segment with a different size. Re-open so we map the new segment.
		it->second = std::make_unique<ShmMapping>(shmName, shmSize, false);
	}

	ShmCircularReader reader(*it->second, sizeof(ShmVideoHeader));
	const auto slot = reader.latestSlotIndex();
	if (!slot) return {};
	const uint8_t* bgr = reader.slotData(*slot);

	return compressBgrToJpeg(bgr, w, h);
}

// == Config validation ========================================================

std::vector<uint8_t> WebSocketBridgeNode::readSecurityFrameFromShm_(
	const nlohmann::json& /*payload*/)
{
	// Security camera writes [uint32 jpegSize][JPEG bytes] to a flat SHM segment.
	static constexpr std::string_view kSecurityShmName = "/oe.cv.security.jpeg";
	static constexpr std::size_t kSecurityShmSize = 4 * 1024 * 1024;  // 4 MB

	std::string shmKey(kSecurityShmName);
	auto it = shmCache_.find(shmKey);
	if (it == shmCache_.end()) {
		try {
			auto mapping = std::make_unique<ShmMapping>(shmKey, kSecurityShmSize, false);
			it = shmCache_.emplace(shmKey, std::move(mapping)).first;
		} catch (const std::exception& e) {
			OE_LOG_DEBUG("security_shm_open_failed: {}", e.what());
			return {};
		}
	}

	const auto* base = static_cast<const uint8_t*>(it->second->data());
	if (!base) return {};

	// Read JPEG size (first 4 bytes).
	uint32_t jpegSize = 0;
	std::memcpy(&jpegSize, base, sizeof(jpegSize));

	if (jpegSize == 0 || jpegSize > kSecurityShmSize - sizeof(jpegSize)) {
		return {};
	}

	// Validate JPEG magic bytes.
	const auto* jpegData = base + sizeof(jpegSize);
	if (jpegSize < 3 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 || jpegData[2] != 0xFF) {
		return {};
	}

	return std::vector<uint8_t>(jpegData, jpegData + jpegSize);
}

std::vector<uint8_t> WebSocketBridgeNode::readBeautyFrameFromShm_(
	const nlohmann::json& /*payload*/)
{
	// Beauty uses ShmJpegControl double-buffer layout (same as blur/face_filter):
	//   [ShmJpegControl (64 B)][slot 0 JPEG (1 MiB)][slot 1 JPEG (1 MiB)]
	static constexpr std::string_view kBeautyShmName = "/oe.cv.beauty.jpeg";

	std::string shmKey(kBeautyShmName);
	auto it = shmCache_.find(shmKey);
	if (it == shmCache_.end()) {
		try {
			auto mapping = std::make_unique<ShmMapping>(
				shmKey, kJpegShmSegmentByteSize, false);
			it = shmCache_.emplace(shmKey, std::move(mapping)).first;
		} catch (const std::exception& e) {
			OE_LOG_DEBUG("beauty_shm_open_failed: {}", e.what());
			return {};
		}
	}

	auto* ctrl = reinterpret_cast<const ShmJpegControl*>(it->second->bytes());
	const uint32_t slot = ctrl->writeIndex.load(std::memory_order_acquire);
	if (slot > 1) return {};  // double-buffer: only slots 0 and 1 are valid
	const uint32_t size = ctrl->jpegSize[slot];
	if (size == 0 || size > kMaxJpegBytesPerSlot) {
		return {};
	}

	const std::size_t offset = sizeof(ShmJpegControl)
		+ static_cast<std::size_t>(slot) * kMaxJpegBytesPerSlot;
	const uint8_t* src = it->second->bytes() + offset;
	std::vector<uint8_t> result(src, src + size);

	// Stale-read guard: if writer flipped during our read, discard
	if (ctrl->writeIndex.load(std::memory_order_acquire) != slot) return {};

	return result;
}

tl::expected<WebSocketBridgeNode::Config, std::string>
WebSocketBridgeNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("wsBridgePubPort", raw.wsBridgePubPort);
	v.requirePort("blurSubPort", raw.blurSubPort);
	v.requirePort("videoIngestSubPort", raw.videoIngestSubPort);
	v.requirePort("ttsSubPort", raw.ttsSubPort);
	v.requirePort("llmSubPort", raw.llmSubPort);
	v.requirePort("daemonSubPort", raw.daemonSubPort);
	v.requirePort("faceSubPort", raw.faceSubPort);
	v.requirePort("videoDenoiseSubPort", raw.videoDenoiseSubPort);
	v.requirePort("audioDenoiseSubPort", raw.audioDenoiseSubPort);
	v.requirePort("sam2SubPort", raw.sam2SubPort);
	v.requirePort("screenIngestSubPort", raw.screenIngestSubPort);
	v.requirePort("securityCameraSubPort", raw.securityCameraSubPort);
	v.requirePort("beautySubPort", raw.beautySubPort);
	v.requireNonEmpty("startupMode", raw.startupMode);
	v.requireNonEmpty("moduleName", raw.moduleName);
	if (auto err = v.finish(); !err.empty())
		return tl::unexpected(err);
	return raw;
}
