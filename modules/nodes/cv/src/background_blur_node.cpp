// ---------------------------------------------------------------------------
// BackgroundBlurNode — real-time background blur pipeline stage
//
// Input:  BGR24 via ShmCircularBuffer from /oe.vid.ingest (or upstream node)
// GPU:    MediaPipe Selfie Seg ONNX → person mask → Gaussian blur compositing
//         ISP adjustments (brightness, contrast, saturation, sharpness)
// Output: JPEG to /oe.cv.blur.jpeg (WebSocketBridge display), or
//         BGR24 to /oe.cv.blur.bgr (downstream chaining)
// ---------------------------------------------------------------------------
#include "cv/background_blur_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "common/ui_action.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"

#include <cstring>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


// --- Construction / Destruction ---

BackgroundBlurNode::BackgroundBlurNode(const Config& config)
	: config_(config),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  config.zmqSendHighWaterMark,
		  config.pollTimeout})
{
}

BackgroundBlurNode::~BackgroundBlurNode()
{
	stop();
}

// --- Config::validate ---

tl::expected<BackgroundBlurNode::Config, std::string>
BackgroundBlurNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("videoSubPort", raw.videoSubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);
	v.requireNonEmpty("selfieSegModelPath", raw.selfieSegModelPath);
	v.requirePositive("blurKernelSize", raw.blurKernelSize);
	v.requireRange("jpegQuality", raw.jpegQuality, 1, 100);
	v.requireRangeF("blurSigma", raw.blurSigma, 0.1f, 100.0f);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// --- configureTransport() ---

tl::expected<void, std::string> BackgroundBlurNode::configureTransport()
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

	OE_LOG_INFO("bg_blur_configured: pub={}, engine={}, jpeg_q={}, output={}, input_shm={}, input_topic={}",
		config_.pubPort, config_.selfieSegModelPath, config_.jpegQuality,
		outputFormatName(config_.outputFormat), config_.inputShmName, config_.inputTopic);

	return {};
}

// --- loadInferencer() ---

tl::expected<void, std::string> BackgroundBlurNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer — call setInferencer() before initialize()"));
	}
	inferencer_->loadEngine(config_.selfieSegModelPath,
	                     config_.inputWidth, config_.inputHeight);

	OE_LOG_INFO("bg_blur_inferencer_loaded: engine={}", config_.selfieSegModelPath);

	return {};
}

// --- processFrame() — one frame through the GPU pipeline ---
//
// Pipeline: SHM read → inferencer (ISP → YOLO-seg → blur → composite → JPEG)
//         → stale-read guard → SHM write → ZMQ publish

void BackgroundBlurNode::processFrame(const nlohmann::json& /*frameMetadata*/)
{
	OE_ZONE_SCOPED;
	const auto frameStart = std::chrono::steady_clock::now();

	// 1. Read latest BGR24 frame from input SHM
	const auto frame = readLatestBgrFrame(*shmIn_);
	if (!frame.data) return;

	// 2. VRAM pre-flight
	if (auto check = ensureVramAvailableMiB(kBgBlurInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("blur_vram_preflight_failed: {}", check.error());
		return;
	}

	// 2a. Deadline check before entering the GPU pipeline
	const auto elapsed = std::chrono::steady_clock::now() - frameStart;
	const auto deadlineMs = std::chrono::milliseconds{kBlurDeadlineMs};
	bool blurred = true;

	if (elapsed >= deadlineMs) {
		OE_LOG_WARN("blur_deadline_exceeded_early: elapsed_ms={:.1f}, frame_dropped",
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
			if (!processErrorLogged_) {
				OE_LOG_WARN("blur_process_bgr_error: error={}", result.error());
				processErrorLogged_ = true;
			}
			return;
		}

		// GPU fence — block until GPU has finished writing BGR24
		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		// Stale-read guard
		if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

		// Update header and commit
		auto* hdr = shmOutBgr_->header();
		hdr->width        = frame.width;
		hdr->height       = frame.height;
		hdr->bytesPerPixel = kBgr24BytesPerPixel;
		hdr->seqNumber    = ++frameSeq_;
		hdr->timestampNs  = static_cast<uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());
		shmOutBgr_->commitWrite();

		OE_LOG_DEBUG("blur_bgr_frame_done: slot={}, seq={}, {}x{}",
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
			if (!processErrorLogged_) {
				OE_LOG_WARN("blur_process_error: error={}", result.error());
				processErrorLogged_ = true;
			}
			return;
		}
		const std::size_t jpegSize = result.value();

		// Check if pipeline exceeded the deadline
		const auto pipelineElapsed = std::chrono::steady_clock::now() - frameStart;
		if (pipelineElapsed > deadlineMs) {
			blurred = false;
			OE_LOG_DEBUG("blur_deadline_exceeded: pipeline_ms={:.1f}",
				std::chrono::duration<double, std::milli>(pipelineElapsed).count());
		}

		OE_LOG_DEBUG("blur_frame_done: jpeg_size={}, blurred={}, vram_used={}MiB, host_rss={}KiB",
		             jpegSize, blurred, queryVramMiB().usedMiB, hostRssKiB());

		// GPU fence — block until JPEG is written
		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		// Stale-read guard
		if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

		// Commit JPEG to output SHM and publish
		ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
		ctrl->seqNumber[writeSlot] = ++frameSeq_;
		ctrl->writeIndex.store(writeSlot, std::memory_order_release);
		publishBlurredFrame(jpegSize, blurred);
	}
}

// --- handleUiCommand() ---

void BackgroundBlurNode::handleUiCommand(const nlohmann::json& cmd)
{
	const auto action = parseUiAction(cmd.value("action", std::string{}));
	switch (action) {
	case UiAction::kSetImageAdjust: {
		IspParams isp;
		isp.brightness = cmd.value("brightness", 0.0f);
		isp.contrast   = cmd.value("contrast",   1.0f);
		isp.saturation = cmd.value("saturation", 1.0f);
		isp.sharpness  = cmd.value("sharpness",  0.0f);
		inferencer_->setIspParams(isp);

		OE_LOG_INFO("bg_blur_isp_adjusted: b={}, c={}, s={}, sh={}",
			isp.brightness, isp.contrast, isp.saturation, isp.sharpness);
		break;
	}
	default:
		break;
	}
}

// --- publishBlurredFrame() ---

void BackgroundBlurNode::publishBlurredFrame(std::size_t jpegSize, bool blurred)
{
	static thread_local BlurredFrameMsg payload;
	payload.shm     = config_.outputShmName;
	payload.size    = static_cast<int64_t>(jpegSize);
	payload.seq     = frameSeq_;
	payload.blurred = blurred;
	messageRouter_.publish("blurred_frame", nlohmann::json(payload));
}


