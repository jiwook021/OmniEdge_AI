// ---------------------------------------------------------------------------
// BeautyNode — real-time face beautification pipeline stage
//
// Input:  BGR24 via ShmCircularBuffer from /oe.vid.ingest (or upstream node)
// GPU:    Face mesh detection → skin smoothing + feature enhancement
// Output: JPEG to /oe.cv.beauty.jpeg (WebSocketBridge display), or
//         BGR24 to /oe.cv.beauty.bgr (downstream chaining)
// ---------------------------------------------------------------------------
#include "cv/beauty_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"
#include "common/ui_action.hpp"

#include <cstring>
#include <filesystem>
#include <format>
#include <csetjmp>
#include <cstdlib>
#include <thread>

#include <nlohmann/json.hpp>
#include <jpeglib.h>

#include "common/zmq_messages.hpp"
#include "zmq/jpeg_constants.hpp"


// SHM layout: [ShmJpegControl (64 B)][slot 0 JPEG][slot 1 JPEG]

namespace {

std::vector<uint8_t> compressBgrToJpeg(const uint8_t* bgr, uint32_t width,
                                       uint32_t height, int jpegQuality)
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
		std::free(outBuf);
		return {};
	}

	jpeg_create_compress(&cinfo);
	jpeg_mem_dest(&cinfo, &outBuf, &outSize);

	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_EXT_BGR;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, jpegQuality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	while (cinfo.next_scanline < cinfo.image_height) {
		auto* row = const_cast<uint8_t*>(bgr + cinfo.next_scanline * width * 3u);
		jpeg_write_scanlines(&cinfo, &row, 1);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	std::vector<uint8_t> jpeg(outBuf, outBuf + outSize);
	std::free(outBuf);
	return jpeg;
}

} // namespace

// --- Construction / Destruction ---

BeautyNode::BeautyNode(const Config& config)
	: config_(config),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  config.zmqSendHighWaterMark,
		  config.pollTimeout}),
	  beautyEnabled_(config.enabledAtStartup)
{
}

BeautyNode::~BeautyNode()
{
	stop();
}

// --- Config::validate ---

tl::expected<BeautyNode::Config, std::string>
BeautyNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("videoSubPort", raw.videoSubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);
	v.requireRange("jpegQuality", raw.jpegQuality, 1, 100);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// --- configureTransport() ---

tl::expected<void, std::string> BeautyNode::configureTransport()
{
	OE_ZONE_SCOPED;
	auto transport = oe::cv::configureCvTransport(
		config_.moduleName, config_.inputShmName, config_.outputFormat,
		config_.outputBgrShmName, config_.outputSlotCount, config_.outputShmName);
	if (!transport) return tl::unexpected(transport.error());
	shmIn_     = std::move(transport->shmIn);
	shmOut_    = std::move(transport->shmOutJpeg);
	shmOutBgr_ = std::move(transport->shmOutBgr);

	// ZMQ subscriptions via MessageRouter (configurable input topic)
	messageRouter_.subscribe(config_.videoSubPort, config_.inputTopic, /*conflate=*/true,
		[this](const nlohmann::json& msg) { processFrame(msg); });
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) { handleUiCommand(msg); });

	OE_LOG_INFO("beauty_configured: pub={}, model={}, enabled={}, output={}, input_shm={}, input_topic={}",
		config_.pubPort, config_.faceMeshOnnxPath, beautyEnabled_.load(),
		outputFormatName(config_.outputFormat), config_.inputShmName, config_.inputTopic);

	return {};
}

// --- loadInferencer() ---

tl::expected<void, std::string> BeautyNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer — call setInferencer() before initialize()"));
	}

	if (!config_.faceMeshOnnxPath.empty()) {
		if (!std::filesystem::exists(config_.faceMeshOnnxPath)) {
			OE_LOG_WARN("beauty_model_missing: path={} — running in pass-through mode "
			            "(re-run scripts/install_models.sh to enable filters)",
			            config_.faceMeshOnnxPath);
			modelLoaded_ = false;
			return {};
		}
		try {
			inferencer_->loadModel(config_.faceMeshOnnxPath);
			modelLoaded_ = true;
		} catch (const std::exception& e) {
			OE_LOG_WARN("beauty_model_load_failed: {} — running in pass-through mode", e.what());
			modelLoaded_ = false;
			return {};
		}
	}

	OE_LOG_INFO("beauty_inferencer_loaded: model={} loaded={}",
	           config_.faceMeshOnnxPath, modelLoaded_);

	return {};
}

// --- processFrame() — one frame through the beauty pipeline ---

void BeautyNode::processFrame(const nlohmann::json& /*frameMetadata*/)
{
	OE_ZONE_SCOPED;
	OE_FRAME_MARK;

	// 0. Skip processing when beauty is toggled off
	if (!beautyEnabled_.load(std::memory_order_relaxed)) return;

	// 1. Read latest BGR24 frame from input SHM
	const auto frame = readLatestBgrFrame(*shmIn_);
	if (!frame.data) return;

	// Pass-through mode: FaceMesh model missing. Keep /beauty_video alive by
	// relaying camera frames without GPU beautification.
	if (!modelLoaded_) {
		if (config_.outputFormat == OutputFormat::kBgr24) {
			const std::size_t frameBytes = static_cast<std::size_t>(frame.width)
			                             * frame.height * kBgr24BytesPerPixel;
			if (frameBytes > kMaxBgr24FrameBytes) {
				OE_LOG_WARN("beauty_passthrough_bgr_too_large: bytes={}, max={}",
				            frameBytes, kMaxBgr24FrameBytes);
				return;
			}

			auto [slotPtr, slotIdx] = shmOutBgr_->acquireWriteSlot();
			std::memcpy(slotPtr, frame.data, frameBytes);

			auto* hdr = shmOutBgr_->header();
			hdr->width         = frame.width;
			hdr->height        = frame.height;
			hdr->bytesPerPixel = kBgr24BytesPerPixel;
			hdr->seqNumber     = ++frameSeq_;
			hdr->timestampNs   = static_cast<uint64_t>(
				std::chrono::steady_clock::now().time_since_epoch().count());
			shmOutBgr_->commitWrite();

			OE_LOG_DEBUG("beauty_passthrough_bgr_frame: slot={}, seq={}, {}x{}",
			             slotIdx, frameSeq_, frame.width, frame.height);
			oe::cv::publishBgrFrame(messageRouter_, config_.bgrTopic,
				config_.outputBgrShmName, frameSeq_, frame.width, frame.height);
		} else {
			auto jpeg = compressBgrToJpeg(
				frame.data, frame.width, frame.height, config_.jpegQuality);
			if (jpeg.empty()) {
				OE_LOG_WARN("beauty_passthrough_encode_failed: {}x{}",
				            frame.width, frame.height);
				return;
			}
			if (jpeg.size() > kMaxJpegBytesPerSlot) {
				OE_LOG_WARN("beauty_passthrough_jpeg_too_large: bytes={}, max={}",
				            jpeg.size(), kMaxJpegBytesPerSlot);
				return;
			}

			auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
			const uint32_t writeSlot =
				1u - ctrl->writeIndex.load(std::memory_order_relaxed);
			uint8_t* outSlot = shmOut_->bytes()
			                 + sizeof(ShmJpegControl)
			                 + writeSlot * kMaxJpegBytesPerSlot;
			std::memcpy(outSlot, jpeg.data(), jpeg.size());

			ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpeg.size());
			ctrl->seqNumber[writeSlot] = ++frameSeq_;
			ctrl->writeIndex.store(writeSlot, std::memory_order_release);

			publishBeautyFrame(jpeg.size());
		}
		return;
	}

	const auto frameStart = std::chrono::steady_clock::now();

	// 2. VRAM pre-flight
	if (auto check = ensureVramAvailableMiB(kBeautyInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("beauty_vram_preflight_failed: {}", check.error());
		return;
	}

	// 2a. Deadline check
	const auto elapsed = std::chrono::steady_clock::now() - frameStart;
	const auto deadlineMs = std::chrono::milliseconds{kBeautyFramePacingMs};
	if (elapsed >= deadlineMs) {
		OE_LOG_WARN("beauty_deadline_exceeded_early: elapsed_ms={:.1f}, frame_dropped",
			std::chrono::duration<double, std::milli>(elapsed).count());
		return;
	}

	// 3. Run inference — output path depends on outputFormat
	if (config_.outputFormat == OutputFormat::kBgr24) {
		// --- BGR24 pipeline chaining path ---
		auto [slotPtr, slotIdx] = shmOutBgr_->acquireWriteSlot();

		auto result = inferencer_->processFrameGetBgr(
			frame.data, frame.width, frame.height,
			slotPtr, kMaxBgr24FrameBytes);

		if (!result) {
			OE_LOG_WARN("beauty_process_bgr_error: error={}", result.error());
			return;
		}

		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

		auto* hdr = shmOutBgr_->header();
		hdr->width        = frame.width;
		hdr->height       = frame.height;
		hdr->bytesPerPixel = kBgr24BytesPerPixel;
		hdr->seqNumber    = ++frameSeq_;
		hdr->timestampNs  = static_cast<uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());
		shmOutBgr_->commitWrite();

		OE_LOG_DEBUG("beauty_bgr_frame_done: slot={}, seq={}, {}x{}",
		             slotIdx, frameSeq_, frame.width, frame.height);
		oe::cv::publishBgrFrame(messageRouter_, config_.bgrTopic,
			config_.outputBgrShmName, frameSeq_, frame.width, frame.height);
	} else {
		// --- JPEG display path (default) ---
		auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
		const uint32_t writeSlot =
			1u - ctrl->writeIndex.load(std::memory_order_relaxed);
		uint8_t* outSlot = shmOut_->bytes()
		                   + sizeof(ShmJpegControl)
		                   + writeSlot * kMaxJpegBytesPerSlot;

		auto result = inferencer_->processFrame(
			frame.data, frame.width, frame.height,
			outSlot, kMaxJpegBytesPerSlot);

		if (!result) {
			OE_LOG_WARN("beauty_process_error: error={}", result.error());
			return;
		}
		const std::size_t jpegSize = result.value();

		const auto pipelineElapsed = std::chrono::steady_clock::now() - frameStart;
		OE_LOG_DEBUG("beauty_frame_done: jpeg_size={}, pipeline_ms={:.1f}, "
		             "params_identity={}, vram_used={}MiB",
		             jpegSize,
		             std::chrono::duration<double, std::milli>(pipelineElapsed).count(),
		             beautyParams_.isIdentity(),
		             queryVramMiB().usedMiB);

		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

		ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
		ctrl->seqNumber[writeSlot] = ++frameSeq_;
		ctrl->writeIndex.store(writeSlot, std::memory_order_release);
		publishBeautyFrame(jpegSize);
	}
}

// --- handleUiCommand() ---

void BeautyNode::handleUiCommand(const nlohmann::json& cmd)
{
	const auto action = parseUiAction(cmd.value("action", std::string{}));
	switch (action) {
	case UiAction::kToggleBeauty: {
		// Frontend sends toggle_beauty with explicit enabled=true/false on connect.
		// Respect that value when present; fallback to legacy toggle semantics.
		if (cmd.contains("enabled") && cmd["enabled"].is_boolean()) {
			beautyEnabled_ = cmd["enabled"].get<bool>();
		} else {
			beautyEnabled_ = !beautyEnabled_;
		}
		OE_LOG_INFO("beauty_toggled: enabled={}", beautyEnabled_.load());
		break;
	}
	case UiAction::kSetBeautySkin: {
		beautyParams_.smoothing   = cmd.value("smoothing",    beautyParams_.smoothing);
		beautyParams_.toneEven    = cmd.value("tone_even",    beautyParams_.toneEven);
		beautyParams_.darkCircles = cmd.value("dark_circles", beautyParams_.darkCircles);
		beautyParams_.blemish     = cmd.value("blemish",      beautyParams_.blemish);
		beautyParams_.sharpen     = cmd.value("sharpen",      beautyParams_.sharpen);
		inferencer_->setBeautyParams(beautyParams_);
		OE_LOG_DEBUG("beauty_skin_updated: smoothing={:.1f} tone={:.1f} "
		             "dark_circles={:.1f} blemish={:.1f} sharpen={:.1f}",
		             beautyParams_.smoothing, beautyParams_.toneEven,
		             beautyParams_.darkCircles, beautyParams_.blemish,
		             beautyParams_.sharpen);
		break;
	}
	case UiAction::kSetBeautyShape: {
		beautyParams_.faceSlim   = cmd.value("face_slim",    beautyParams_.faceSlim);
		beautyParams_.eyeEnlarge = cmd.value("eye_enlarge",  beautyParams_.eyeEnlarge);
		beautyParams_.noseNarrow = cmd.value("nose_narrow",  beautyParams_.noseNarrow);
		beautyParams_.jawReshape = cmd.value("jaw_reshape",  beautyParams_.jawReshape);
		inferencer_->setBeautyParams(beautyParams_);
		OE_LOG_DEBUG("beauty_shape_updated: slim={:.1f} eyes={:.1f} nose={:.1f} jaw={:.1f}",
		             beautyParams_.faceSlim, beautyParams_.eyeEnlarge,
		             beautyParams_.noseNarrow, beautyParams_.jawReshape);
		break;
	}
	case UiAction::kSetBeautyLight: {
		beautyParams_.brightness = cmd.value("brightness",  beautyParams_.brightness);
		beautyParams_.warmth     = cmd.value("warmth",      beautyParams_.warmth);
		beautyParams_.shadowFill = cmd.value("shadow_fill", beautyParams_.shadowFill);
		beautyParams_.highlight  = cmd.value("highlight",   beautyParams_.highlight);
		inferencer_->setBeautyParams(beautyParams_);
		OE_LOG_DEBUG("beauty_light_updated: brightness={:.1f} warmth={:.1f} "
		             "shadow={:.1f} highlight={:.1f}",
		             beautyParams_.brightness, beautyParams_.warmth,
		             beautyParams_.shadowFill, beautyParams_.highlight);
		break;
	}
	case UiAction::kSetBeautyBg: {
		const int mode = cmd.value("bg_mode", 0);
		if (mode < 0 || mode > static_cast<int>(BeautyParams::BgMode::kImage)) {
			OE_LOG_WARN("beauty_bg_invalid: bg_mode={} out of range [0,{}]",
				mode, static_cast<int>(BeautyParams::BgMode::kImage));
			break;
		}
		beautyParams_.bgMode = static_cast<BeautyParams::BgMode>(mode);
		inferencer_->setBeautyParams(beautyParams_);
		OE_LOG_DEBUG("beauty_bg_updated: mode={}", mode);
		break;
	}
	case UiAction::kSetBeautyPreset: {
		const std::string preset = cmd.value("preset", std::string{});
		OE_LOG_INFO("beauty_preset_applied: preset={}", preset);
		// Preset values are applied by the frontend — it sends individual
		// set_beauty_skin/shape/light commands with the preset's values.
		break;
	}
	default:
		break;
	}
}

// --- publishBeautyFrame() ---

void BeautyNode::publishBeautyFrame(std::size_t jpegSize)
{
	static thread_local BeautyFrameMsg payload;
	payload.shm  = config_.outputShmName;
	payload.size = static_cast<int64_t>(jpegSize);
	payload.seq  = frameSeq_;
	messageRouter_.publish("beauty_frame", nlohmann::json(payload));
}
