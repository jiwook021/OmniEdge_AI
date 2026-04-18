// ---------------------------------------------------------------------------
// VideoDenoiseNode — temporal video denoising pipeline stage (BasicVSR++)
//
// Input:  BGR24 via ShmCircularBuffer from /oe.vid.ingest (or upstream node)
// GPU:    BasicVSR++ ONNX model over a sliding temporal window of N frames
// Output: JPEG to /oe.cv.denoise.jpeg (WebSocketBridge display), or
//         BGR24 to /oe.cv.denoise.bgr (downstream chaining)
// ---------------------------------------------------------------------------
#include "cv/video_denoise_node.hpp"

#include "common/oe_tracy.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"

#include <cstring>
#include <format>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


// --- Construction / Destruction ---

VideoDenoiseNode::VideoDenoiseNode(const Config& config,
                                   std::unique_ptr<DenoiseInferencer> inferencer)
	: config_(config),
	  inferencer_(std::move(inferencer)),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  kPublisherDataHighWaterMark,
		  config.pollTimeout})
{
	// Pre-allocate temporal frame window and pointer buffer
	frameWindow_.resize(config_.temporalWindowSize);
	framePtrsBuffer_.resize(config_.temporalWindowSize);
}

VideoDenoiseNode::~VideoDenoiseNode()
{
	stop();
}

// --- Config::validate ---

tl::expected<VideoDenoiseNode::Config, std::string>
VideoDenoiseNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("videoSubPort", raw.videoSubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);
	v.requireNonEmpty("onnxModelPath", raw.onnxModelPath);
	v.requirePositive("temporalWindowSize", static_cast<int>(raw.temporalWindowSize));

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// --- configureTransport() ---

tl::expected<void, std::string> VideoDenoiseNode::configureTransport()
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
		[this](const nlohmann::json& msg) { handleVideoFrame(msg); });

	OE_LOG_INFO("video_denoise_configured: pub={}, model={}, window={}, output={}, input_topic={}",
		config_.pubPort, config_.onnxModelPath, config_.temporalWindowSize,
		outputFormatName(config_.outputFormat), config_.inputTopic);

	return {};
}

// --- loadInferencer() ---

tl::expected<void, std::string> VideoDenoiseNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer — pass via constructor"));
	}

	auto loadResult = inferencer_->loadModel(config_.onnxModelPath);
	if (!loadResult) {
		return tl::unexpected(
			std::format("Inferencer loadModel failed: {}", loadResult.error()));
	}

	OE_LOG_INFO("video_denoise_inferencer_loaded: model={}", config_.onnxModelPath);

	return {};
}

// --- handleVideoFrame() ---

void VideoDenoiseNode::handleVideoFrame(const nlohmann::json& /*msg*/)
{
	OE_ZONE_SCOPED;
	// 1. Read latest BGR24 frame from input SHM
	auto frameData = readLatestBgrFrame();
	if (frameData.empty()) return;

	// 2. Store in temporal ring buffer
	frameWindow_[frameWindowHead_] = std::move(frameData);
	frameWindowHead_ = (frameWindowHead_ + 1) % config_.temporalWindowSize;
	if (frameWindowCount_ < config_.temporalWindowSize) {
		++frameWindowCount_;
	}

	// 3. Only run denoising when we have a full temporal window
	if (frameWindowCount_ >= config_.temporalWindowSize) {
		runDenoise();
	}
}

// --- readLatestBgrFrame() ---

std::vector<uint8_t> VideoDenoiseNode::readLatestBgrFrame()
{
	const auto frame = ::readLatestBgrFrame(*shmIn_);
	if (!frame.data) return {};

	const std::size_t frameBytes =
		static_cast<std::size_t>(frame.width) * frame.height * 3u;

	std::vector<uint8_t> result(frameBytes);
	std::memcpy(result.data(), frame.data, frameBytes);

	// Stale-read guard — discard if overwritten during memcpy
	if (::currentShmSlotIndex(*shmIn_) != frame.slotIndex) {
		return {};
	}

	return result;
}

// --- runDenoise() ---

void VideoDenoiseNode::runDenoise()
{
	OE_ZONE_SCOPED;
	const auto frameStart = std::chrono::steady_clock::now();

	// Build ordered pointer array from ring buffer (oldest frame first).
	// frameWindowHead_ points to the NEXT write position (post-increment in
	// handleVideoFrame), so the oldest frame is at head - count.
	for (uint32_t i = 0; i < config_.temporalWindowSize; ++i) {
		const uint32_t idx =
			(frameWindowHead_ - frameWindowCount_ + i + config_.temporalWindowSize)
			% config_.temporalWindowSize;
		framePtrsBuffer_[i] = frameWindow_[idx].data();
	}

	// Read frame dimensions from input SHM header
	const auto* videoHeader = reinterpret_cast<const ShmVideoHeader*>(
		shmIn_->bytes());
	const uint32_t width  = videoHeader->width;
	const uint32_t height = videoHeader->height;

	// VRAM pre-flight
	if (auto check = ensureVramAvailableMiB(kBasicVsrppInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("denoise_vram_preflight_failed: {}", check.error());
		return;
	}

	// Run inference — output path depends on outputFormat
	if (config_.outputFormat == OutputFormat::kBgr24) {
		// --- BGR24 pipeline chaining path ---
		auto [slotPtr, slotIdx] = shmOutBgr_->acquireWriteSlot();

		auto result = inferencer_->processFramesGetBgr(
			framePtrsBuffer_.data(),
			config_.temporalWindowSize,
			width, height,
			slotPtr, kMaxBgr24FrameBytes);

		if (!result) {
			OE_LOG_WARN("video_denoise_process_bgr_error: error={}", result.error());
			return;
		}

		// GPU fence — block until GPU has finished writing BGR24
		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		// Update header and commit
		auto* hdr = shmOutBgr_->header();
		hdr->width        = width;
		hdr->height       = height;
		hdr->bytesPerPixel = kBgr24BytesPerPixel;
		hdr->seqNumber    = static_cast<uint32_t>(++frameSeq_);
		hdr->timestampNs  = static_cast<uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());
		shmOutBgr_->commitWrite();

		OE_LOG_DEBUG("denoise_bgr_frame_done: slot={}, seq={}, {}x{}",
		             slotIdx, frameSeq_, width, height);
		oe::cv::publishBgrFrame(messageRouter_, config_.bgrTopic,
			config_.outputBgrShmName, frameSeq_, width, height);
	} else {
		// --- JPEG display path (default) ---
		auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
		const uint32_t writeSlot =
			1u - ctrl->writeIndex.load(std::memory_order_relaxed);
		uint8_t* outSlot = shmOut_->bytes()
		                    + sizeof(ShmJpegControl)
		                    + writeSlot * kMaxJpegBytesPerSlot;

		auto result = inferencer_->processFrames(
			framePtrsBuffer_.data(),
			config_.temporalWindowSize,
			width, height,
			outSlot, kMaxJpegBytesPerSlot);

		if (!result) {
			OE_LOG_WARN("video_denoise_process_error: error={}", result.error());
			return;
		}
		const std::size_t jpegSize = result.value();

		// Check if the pipeline exceeded the deadline
		const auto pipelineElapsed = std::chrono::steady_clock::now() - frameStart;
		const auto deadlineMs = std::chrono::milliseconds{kDenoiseDeadlineMs};
		const bool enhanced = (pipelineElapsed <= deadlineMs);

		if (!enhanced) {
			OE_LOG_DEBUG("denoise_deadline_exceeded: pipeline_ms={:.1f}",
				std::chrono::duration<double, std::milli>(pipelineElapsed).count());
		}

		OE_LOG_DEBUG("video_denoise_done: jpeg_size={}, enhanced={}, vram_used={}MiB, host_rss={}KiB",
		             jpegSize, enhanced, queryVramMiB().usedMiB, hostRssKiB());

		// GPU fence — block until the GPU has finished writing the JPEG
		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		// Commit JPEG to output SHM
		ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
		ctrl->seqNumber[writeSlot] = static_cast<uint32_t>(++frameSeq_);
		ctrl->writeIndex.store(writeSlot, std::memory_order_release);

		// Publish ZMQ notification with quality indicator
		static thread_local DenoisedFrameMsg payload;
		payload.shm      = config_.outputShmName;
		payload.size     = static_cast<int64_t>(jpegSize);
		payload.seq      = frameSeq_;
		payload.enhanced = enhanced;
		messageRouter_.publish("denoised_frame", nlohmann::json(payload));
	}
}


