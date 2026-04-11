#include "cv/background_blur_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "common/ui_action.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"

#include <cstring>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


// SHM layout: [ShmJpegControl (64 B)][slot 0 JPEG][slot 1 JPEG]
// Size constant moved to cv/cv_constants.hpp as kJpegShmSegmentByteSize.

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
	v.requireNonEmpty("yolov8EnginePath", raw.yolov8EnginePath);
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
	OeLogger::instance().setModule(config_.moduleName);

	// Input SHM: /oe.vid.ingest (consumer — retry up to 10 times waiting for producer)
	const std::size_t inShmSize =
		ShmCircularBuffer<ShmVideoHeader>::segmentSize(
			kCircularBufferSlotCount, kMaxBgr24FrameBytes);
	auto shmResult = oe::shm::openConsumerWithRetry(
		config_.inputShmName, inShmSize, "bg_blur");
	if (!shmResult) return tl::unexpected(shmResult.error());
	shmIn_ = std::move(*shmResult);

	// Output SHM: /oe.cv.blur.jpeg (producer — will unlink on close)
	shmOut_ = oe::shm::createProducer(config_.outputShmName, kJpegShmSegmentByteSize);

	// ZMQ subscriptions via MessageRouter
	messageRouter_.subscribe(config_.videoSubPort, "video_frame", /*conflate=*/true,
		[this](const nlohmann::json& msg) { processFrame(msg); });
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) { handleUiCommand(msg); });

	OE_LOG_INFO("bg_blur_configured: pub={}, engine={}, jpeg_q={}",
		config_.pubPort, config_.yolov8EnginePath, config_.jpegQuality);

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
	inferencer_->loadEngine(config_.yolov8EnginePath,
	                     config_.inputWidth, config_.inputHeight);

	OE_LOG_INFO("bg_blur_inferencer_loaded: engine={}", config_.yolov8EnginePath);

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

	// 2. Pick the inactive output slot (double-buffer flip)
	auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
	const uint32_t writeSlot =
		1u - ctrl->writeIndex.load(std::memory_order_relaxed);
	uint8_t* outSlot = shmOut_->bytes()
	                   + sizeof(ShmJpegControl)
	                   + writeSlot * kMaxJpegBytesPerSlot;

	// 3. VRAM pre-flight + inference + JPEG encode
	if (auto check = ensureVramAvailableMiB(kBgBlurInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("blur_vram_preflight_failed: {}", check.error());
		return;
	}

	// 3a. Check deadline before entering the GPU pipeline.
	//     If the frame acquisition + VRAM check already consumed the budget,
	//     skip enhancement and fall back to raw JPEG passthrough.
	const auto elapsed = std::chrono::steady_clock::now() - frameStart;
	const auto deadlineMs = std::chrono::milliseconds{kBlurDeadlineMs};
	bool blurred = true;

	if (elapsed >= deadlineMs) {
		// Deadline already exceeded — frame is stale, skip entirely
		OE_LOG_WARN("blur_deadline_exceeded_early: elapsed_ms={:.1f}, frame_dropped",
			std::chrono::duration<double, std::milli>(elapsed).count());
		return;
	}

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

	// 3b. Check if the full pipeline exceeded the deadline
	const auto pipelineElapsed = std::chrono::steady_clock::now() - frameStart;
	if (pipelineElapsed > deadlineMs) {
		blurred = false;
		OE_LOG_DEBUG("blur_deadline_exceeded: pipeline_ms={:.1f}",
			std::chrono::duration<double, std::milli>(pipelineElapsed).count());
	}

	OE_LOG_DEBUG("blur_frame_done: jpeg_size={}, blurred={}, vram_used={}MiB, host_rss={}KiB",
	             jpegSize, blurred, queryVramMiB().usedMiB, hostRssKiB());

	// 4. GPU fence — block until the GPU has finished writing the JPEG
	//    to the output slot. Without this, the consumer (WS Bridge) could
	//    read a partially-written frame.
	gpuFence_.record(inferencer_->cudaStream());
	gpuFence_.synchronize();

	// 5. Stale-read guard — discard if input was overwritten during inference
	if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

	// 6. Commit JPEG to output SHM and publish
	ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
	ctrl->seqNumber[writeSlot] = ++frameSeq_;
	ctrl->writeIndex.store(writeSlot, std::memory_order_release);
	publishBlurredFrame(jpegSize, blurred);
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

