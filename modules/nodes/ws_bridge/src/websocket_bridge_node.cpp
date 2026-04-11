#include "ws_bridge/websocket_bridge_node.hpp"

#include <algorithm>
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
		|| action == "start_image_transform";
}

/// JPEG magic bytes: FF D8 FF
constexpr uint8_t kJpegMagic[] = {0xFF, 0xD8, 0xFF};

/// PNG 8-byte signature: 89 50 4E 47 0D 0A 1A 0A
constexpr uint8_t kPngMagic[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

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
	return mode == "conversation"
		|| mode == "super_resolution" || mode == "image_transform"
		|| mode == "sam2_segmentation";
}

[[nodiscard]] std::unordered_set<std::string> topicsForMode(std::string_view /*mode*/)
{
	return {
		std::string(kZmqTopicBlurredFrame),
		std::string(kZmqTopicVideoFrame),
		std::string(kZmqTopicTtsAudio),
		std::string(kZmqTopicLlmResponse),
		std::string(kZmqTopicModuleStatus),
		std::string(kZmqTopicFaceRegistered),
		std::string(kZmqTopicDenoisedFrame),
		std::string(kZmqTopicDenoisedAudio),
		std::string(kZmqTopicSegmentationMask),
	};
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
	messageRouter_.subscribe(config_.llmSubPort,          kZmqTopicLlmResponse,   false,
	                  makeHandler(std::string(kZmqTopicLlmResponse)));
	messageRouter_.subscribe(config_.daemonSubPort,       kZmqTopicModuleStatus,  false,
	                  makeHandler(std::string(kZmqTopicModuleStatus)));
	messageRouter_.subscribe(config_.faceSubPort,         kZmqTopicFaceRegistered, false,
	                  makeHandler(std::string(kZmqTopicFaceRegistered)));
	messageRouter_.subscribe(config_.videoDenoiseSubPort, kZmqTopicDenoisedFrame, true,
	                  makeHandler(std::string(kZmqTopicDenoisedFrame)));
	messageRouter_.subscribe(config_.audioDenoiseSubPort, kZmqTopicDenoisedAudio, false,
	                  makeHandler(std::string(kZmqTopicDenoisedAudio)));
	messageRouter_.subscribe(config_.sam2SubPort, kZmqTopicSegmentationMask, false,
	                  makeHandler(std::string(kZmqTopicSegmentationMask)));

	setMode_("conversation");

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
	} catch (...) {}
	if (relay_) {
		try {
			relay_->stop();
		} catch (...) {}
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
		if (!isSupportedMode(cmd["mode"].get<std::string>())) {
			return tl::unexpected(std::string("switch_mode.mode not supported"));
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
	    || action == "toggle_sam2") {
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
	if (action == "select_style") {
		if (!cmd.contains("style") || !cmd["style"].is_string()) {
			return tl::unexpected(std::string("select_style requires string style"));
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
		OE_LOG_WARN("ws_ui_command_rejected: reason={}", validated.error());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", validated.error()},
		});
		return;
	}

	nlohmann::json command = std::move(validated.value());
	const std::string action = command["action"].get<std::string>();

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
		OE_LOG_WARN("ws_binary_upload_too_small: size={}", binaryFrame.size());
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
		OE_LOG_WARN("ws_binary_upload_bad_header: json_len={}, total={}", jsonLen, binaryFrame.size());
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
		OE_LOG_WARN("ws_binary_upload_json_error: error={}", ex.what());
		relay_->broadcastJson({
			{"v", kSchemaVersion},
			{"type", "error"},
			{"message", std::format("Upload JSON parse error: {}", ex.what())},
		});
		return;
	}

	const std::string action = meta.value("action", std::string{});
	if (action != "describe_uploaded_image") {
		OE_LOG_WARN("ws_binary_upload_unknown_action: action={}", action);
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
		OE_LOG_WARN("ws_binary_upload_bad_size: image_bytes={}", imageSize);
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
		OE_LOG_WARN("ws_binary_upload_unknown_format: first_byte=0x{:02x}", imageData[0]);
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
			OE_LOG_WARN("ws_binary_upload_write_failed: path={}, error={}",
			            tempPath, writeResult.error());
			relay_->broadcastJson({
				{"v", kSchemaVersion},
				{"type", "error"},
				{"message", "Failed to write uploaded image to disk"},
			});
			return;
		}
	}

	OE_LOG_INFO("ws_image_uploaded: path={}, size={}, ext={}", tempPath, imageSize, ext);

	// Build and forward the ui_command
	nlohmann::json command = {
		{"v",          kSchemaVersion},
		{"type",       "ui_command"},
		{"action",     "describe_uploaded_image"},
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
		lastBlurFrameTime_ = std::chrono::steady_clock::now();
		auto frame = readBlurredFrameFromShm_(payload);
		if (!frame.empty()) {
			// Frame pacing: only send if enough time has elapsed
			if (shouldPace(lastVideoSendTime_, kBlurFramePacingMs)) {
				lastBlurFrame_ = std::move(frame);  // move to cache (no copy)
				lastVideoSendTime_ = std::chrono::steady_clock::now();
				relay_->broadcastVideo(
					std::vector<uint8_t>(lastBlurFrame_.begin(), lastBlurFrame_.end()));
			} else {
				// Multiple frames arrived between ticks — skip to newest (cache it)
				lastBlurFrame_ = std::move(frame);
			}
		} else if (shouldPace(lastVideoSendTime_, kBlurFramePacingMs)
		           && !lastBlurFrame_.empty()) {
			// No new frame at this tick — re-send cached frame (avoid copy)
			lastVideoSendTime_ = std::chrono::steady_clock::now();
			relay_->broadcastVideo(
				std::vector<uint8_t>(lastBlurFrame_.begin(), lastBlurFrame_.end()));
		}
		return;
	}
	if (topic == kZmqTopicVideoFrame) {
		// Raw video fallback — only relay if blurred_frame is not active
		// (i.e. background_blur module is not producing frames).
		const auto elapsed = std::chrono::steady_clock::now() - lastBlurFrameTime_;
		if (elapsed < kBlurredFrameFallbackTimeout) {
			return;  // bg_blur is active — suppress raw fallback
		}
		// Apply same pacing to raw fallback
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
		// SAM2 is on-demand (not continuous) — no frame pacing needed.
		// Read JPEG overlay from SHM and broadcast to /sam2_video clients.
		auto mask = readSam2MaskFromShm_(payload);
		if (!mask.empty()) {
			relay_->broadcastSam2Video(std::move(mask));
		}
		// Also forward the JSON metadata (iou_score, stability) to /chat.
		relay_->broadcastJson(payload);
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

// == Config validation ========================================================

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
	v.requireNonEmpty("moduleName", raw.moduleName);
	if (auto err = v.finish(); !err.empty())
		return tl::unexpected(err);
	return raw;
}

