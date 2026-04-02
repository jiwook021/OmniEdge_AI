#include "cv/video_denoise_node.hpp"

#include "common/oe_tracy.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"

#include <cstring>
#include <format>

#include <nlohmann/json.hpp>


// SHM output layout: [ShmJpegControl (64 B)][slot 0 JPEG][slot 1 JPEG]
// Size constant moved to video_denoise/video_denoise_constants.hpp as
// kJpegShmSegmentByteSize.

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
	v.requirePort("subVideoPort", raw.subVideoPort);
	v.requirePort("subWsBridgePort", raw.subWsBridgePort);
	v.requireNonEmpty("onnxModelPath", raw.onnxModelPath);
	v.requirePositive("temporalWindowSize", static_cast<int>(raw.temporalWindowSize));

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// --- onConfigure() ---

tl::expected<void, std::string> VideoDenoiseNode::onConfigure()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// Input SHM: /oe.vid.ingest (consumer — circular buffer layout)
	const std::size_t inShmSize =
		ShmCircularBuffer<ShmVideoHeader>::segmentSize(
			kCircularBufferSlotCount, kMaxBgr24FrameBytes);
	shmInput_ = std::make_unique<ShmMapping>(
		config_.shmInput, inShmSize, /*create=*/false);

	// Output SHM: /oe.vid.denoise (producer — will unlink on close)
	shmOutput_ = std::make_unique<ShmMapping>(
		config_.shmOutput, kJpegShmSegmentByteSize, /*create=*/true);
	std::memset(shmOutput_->data(), 0, kJpegShmSegmentByteSize);

	// ZMQ subscriptions via MessageRouter
	messageRouter_.subscribe(config_.subVideoPort, "video_frame", /*conflate=*/true,
		[this](const nlohmann::json& msg) { handleVideoFrame(msg); });

	OE_LOG_INFO("video_denoise_configured: pub={}, model={}, window={}",
		config_.pubPort, config_.onnxModelPath, config_.temporalWindowSize);

	return {};
}

// --- onLoadInferencer() ---

tl::expected<void, std::string> VideoDenoiseNode::onLoadInferencer()
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
	const auto frame = ::readLatestBgrFrame(*shmInput_);
	if (!frame.data) return {};

	const std::size_t frameBytes =
		static_cast<std::size_t>(frame.width) * frame.height * 3u;

	std::vector<uint8_t> result(frameBytes);
	std::memcpy(result.data(), frame.data, frameBytes);

	// Stale-read guard — discard if overwritten during memcpy
	if (::currentShmSlotIndex(*shmInput_) != frame.slotIndex) {
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
		shmInput_->bytes());
	const uint32_t width  = videoHeader->width;
	const uint32_t height = videoHeader->height;

	// Pick the inactive output JPEG slot (double-buffer flip)
	auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOutput_->bytes());
	const uint32_t writeSlot =
		1u - ctrl->writeIndex.load(std::memory_order_relaxed);
	uint8_t* outSlot = shmOutput_->bytes()
	                    + sizeof(ShmJpegControl)
	                    + writeSlot * kMaxJpegBytesPerSlot;

	// VRAM pre-flight + BasicVSR++ inference + JPEG encode
	if (auto check = ensureVramAvailableMiB(kBasicVsrppInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("denoise_vram_preflight_failed: {}", check.error());
		return;
	}
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
	ctrl->seqNumber[writeSlot] = static_cast<uint32_t>(++seqNumber_);
	ctrl->writeIndex.store(writeSlot, std::memory_order_release);

	// Publish ZMQ notification with quality indicator
	static thread_local nlohmann::json payload = {
		{"v",        kSchemaVersion},
		{"type",     "denoised_frame"},
		{"shm",      ""},
		{"size",     int64_t{0}},
		{"seq",      uint64_t{0}},
		{"enhanced", false},
	};
	payload["shm"]      = config_.shmOutput;
	payload["size"]     = static_cast<int64_t>(jpegSize);
	payload["seq"]      = seqNumber_;
	payload["enhanced"] = enhanced;
	messageRouter_.publish("denoised_frame", payload);
}

