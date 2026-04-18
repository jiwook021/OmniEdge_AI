// ---------------------------------------------------------------------------
// FaceFilterNode — real-time AR face filter pipeline stage
//
// Input:  BGR24 via ShmCircularBuffer from /oe.vid.ingest (or upstream node)
// GPU:    Face mesh landmark detection → filter texture overlay
// Output: JPEG to /oe.cv.filter.jpeg (WebSocketBridge display), or
//         BGR24 to /oe.cv.filter.bgr (downstream chaining)
// ---------------------------------------------------------------------------
#include "cv/face_filter_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"
#include "common/ui_action.hpp"

#include <cstring>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


// SHM layout: [ShmJpegControl (64 B)][slot 0 JPEG][slot 1 JPEG]

// --- Construction / Destruction ---

FaceFilterNode::FaceFilterNode(const Config& config)
	: config_(config),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  config.zmqSendHighWaterMark,
		  config.pollTimeout}),
	  filterEnabled_(config.enabledAtStartup),
	  activeFilterId_(config.activeFilterId)
{
}

FaceFilterNode::~FaceFilterNode()
{
	stop();
}

// --- Config::validate ---

tl::expected<FaceFilterNode::Config, std::string>
FaceFilterNode::Config::validate(const Config& raw)
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

tl::expected<void, std::string> FaceFilterNode::configureTransport()
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

	OE_LOG_INFO("face_filter_configured: pub={}, model={}, manifest={}, enabled={}, filter={}, output={}, input_topic={}",
		config_.pubPort, config_.faceMeshOnnxPath, config_.filterManifestPath,
		filterEnabled_, activeFilterId_.empty() ? "(none)" : activeFilterId_,
		outputFormatName(config_.outputFormat), config_.inputTopic);

	return {};
}

// --- loadInferencer() ---

tl::expected<void, std::string> FaceFilterNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer — call setInferencer() before initialize()"));
	}

	if (!config_.faceMeshOnnxPath.empty()) {
		try {
			inferencer_->loadModel(config_.faceMeshOnnxPath);
		} catch (const std::exception& e) {
			return tl::unexpected(std::format(
				"FaceMesh model load failed: {}", e.what()));
		}
	}

	if (!config_.filterManifestPath.empty()) {
		try {
			inferencer_->loadFilterAssets(config_.filterManifestPath);
		} catch (const std::exception& e) {
			OE_LOG_WARN("face_filter_manifest_load_failed: {}", e.what());
			// Non-fatal — filters just won't be available
		}
	}

	if (!activeFilterId_.empty()) {
		inferencer_->setActiveFilter(activeFilterId_);
	}

	OE_LOG_INFO("face_filter_inferencer_loaded: model={}", config_.faceMeshOnnxPath);

	return {};
}

// --- processFrame() — one frame through the face filter pipeline ---
//
// Pipeline: SHM read -> inferencer (FaceMesh -> warp -> composite -> JPEG)
//         -> stale-read guard -> SHM write -> ZMQ publish

void FaceFilterNode::processFrame(const nlohmann::json& /*frameMetadata*/)
{
	OE_ZONE_SCOPED;
	const auto frameStart = std::chrono::steady_clock::now();

	// 1. Read latest BGR24 frame from input SHM
	const auto frame = readLatestBgrFrame(*shmIn_);
	if (!frame.data) return;

	// 2. VRAM pre-flight
	if (auto check = ensureVramAvailableMiB(kFaceFilterInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("face_filter_vram_preflight_failed: {}", check.error());
		return;
	}

	// 2a. Deadline check before entering the GPU pipeline
	const auto elapsed = std::chrono::steady_clock::now() - frameStart;
	const auto deadlineMs = std::chrono::milliseconds{kFaceFilterDeadlineMs};

	if (elapsed >= deadlineMs) {
		OE_LOG_WARN("face_filter_deadline_exceeded_early: elapsed_ms={:.1f}, frame_dropped",
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
			OE_LOG_WARN("face_filter_process_bgr_error: error={}", result.error());
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

		OE_LOG_DEBUG("face_filter_bgr_frame_done: slot={}, seq={}, {}x{}",
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
			OE_LOG_WARN("face_filter_process_error: error={}", result.error());
			return;
		}
		const std::size_t jpegSize = result.value();

		// Check if the full pipeline exceeded the deadline
		const auto pipelineElapsed = std::chrono::steady_clock::now() - frameStart;
		const bool filtered = filterEnabled_
			&& (pipelineElapsed <= deadlineMs);

		if (pipelineElapsed > deadlineMs) {
			OE_LOG_DEBUG("face_filter_deadline_exceeded: pipeline_ms={:.1f}",
				std::chrono::duration<double, std::milli>(pipelineElapsed).count());
		}

		OE_LOG_DEBUG("face_filter_frame_done: jpeg_size={}, filtered={}, filter_id={}, "
		             "vram_used={}MiB, host_rss={}KiB",
		             jpegSize, filtered,
		             activeFilterId_.empty() ? "(none)" : activeFilterId_,
		             queryVramMiB().usedMiB, hostRssKiB());

		// GPU fence — block until GPU has finished writing JPEG
		gpuFence_.record(inferencer_->cudaStream());
		gpuFence_.synchronize();

		// Stale-read guard — discard if input was overwritten during inference
		if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

		// Commit JPEG to output SHM and publish
		ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
		ctrl->seqNumber[writeSlot] = ++frameSeq_;
		ctrl->writeIndex.store(writeSlot, std::memory_order_release);
		publishFilteredFrame(jpegSize, filtered);
	}
}

// --- handleUiCommand() ---

void FaceFilterNode::handleUiCommand(const nlohmann::json& cmd)
{
	const auto action = parseUiAction(cmd.value("action", std::string{}));
	switch (action) {
	case UiAction::kToggleFaceFilter: {
		filterEnabled_ = !filterEnabled_;
		OE_LOG_INFO("face_filter_toggled: enabled={}", filterEnabled_);
		break;
	}
	case UiAction::kSelectFilter: {
		const std::string filterId = cmd.value("filter_id", std::string{});
		activeFilterId_ = filterId;
		inferencer_->setActiveFilter(filterId);
		OE_LOG_INFO("face_filter_selected: filter_id={}",
			filterId.empty() ? "(none)" : filterId);
		break;
	}
	default:
		break;
	}
}

// --- publishFilteredFrame() ---

void FaceFilterNode::publishFilteredFrame(std::size_t jpegSize, bool filtered)
{
	static thread_local FilteredFrameMsg payload;
	payload.shm       = config_.outputShmName;
	payload.size      = static_cast<int64_t>(jpegSize);
	payload.seq       = frameSeq_;
	payload.filtered  = filtered;
	payload.filter_id = activeFilterId_;
	messageRouter_.publish("filtered_frame", nlohmann::json(payload));
}


