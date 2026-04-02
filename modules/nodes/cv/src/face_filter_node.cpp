#include "cv/face_filter_node.hpp"

#include "common/oe_tracy.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "common/ui_action.hpp"

#include <cstring>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>


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

// --- onConfigure() ---

tl::expected<void, std::string> FaceFilterNode::onConfigure()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// Input SHM: /oe.vid.ingest (consumer — retry up to 10 times waiting for producer)
	const std::size_t inShmSize =
		ShmCircularBuffer<ShmVideoHeader>::segmentSize(
			kCircularBufferSlotCount, kMaxBgr24FrameBytes);
	for (int attempt = 1; attempt <= 10; ++attempt) {
		try {
			shmIn_ = std::make_unique<ShmMapping>(
				config_.inputShmName, inShmSize, /*create=*/false);
			break;
		} catch (const std::runtime_error& e) {
			if (attempt == 10) {
				return tl::unexpected(std::format(
					"SHM open failed after 10 retries: {}", e.what()));
			}
			OE_LOG_WARN("face_filter_shm_retry: attempt={}/10, waiting for {}: {}",
				attempt, config_.inputShmName, e.what());
			std::this_thread::sleep_for(std::chrono::milliseconds{kInferencerRetryDelayMs});
		}
	}

	// Output SHM: /oe.cv.facefilter.jpeg (producer — will unlink on close)
	shmOut_ = std::make_unique<ShmMapping>(
		config_.outputShmName, kJpegShmSegmentByteSize, /*create=*/true);
	std::memset(shmOut_->data(), 0, kJpegShmSegmentByteSize);

	// ZMQ subscriptions via MessageRouter
	messageRouter_.subscribe(config_.videoSubPort, "video_frame", /*conflate=*/true,
		[this](const nlohmann::json& msg) { processFrame(msg); });
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) { handleUiCommand(msg); });

	OE_LOG_INFO("face_filter_configured: pub={}, model={}, manifest={}, enabled={}, filter={}",
		config_.pubPort, config_.faceMeshOnnxPath, config_.filterManifestPath,
		filterEnabled_, activeFilterId_.empty() ? "(none)" : activeFilterId_);

	return {};
}

// --- onLoadInferencer() ---

tl::expected<void, std::string> FaceFilterNode::onLoadInferencer()
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

	// 2. Pick the inactive output slot (double-buffer flip)
	auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
	const uint32_t writeSlot =
		1u - ctrl->writeIndex.load(std::memory_order_relaxed);
	uint8_t* outSlot = shmOut_->bytes()
	                   + sizeof(ShmJpegControl)
	                   + writeSlot * kMaxJpegBytesPerSlot;

	// 3. VRAM pre-flight
	if (auto check = ensureVramAvailableMiB(kFaceFilterInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("face_filter_vram_preflight_failed: {}", check.error());
		return;
	}

	// 3a. Deadline check before entering the GPU pipeline
	const auto elapsed = std::chrono::steady_clock::now() - frameStart;
	const auto deadlineMs = std::chrono::milliseconds{kFaceFilterDeadlineMs};

	if (elapsed >= deadlineMs) {
		OE_LOG_WARN("face_filter_deadline_exceeded_early: elapsed_ms={:.1f}, frame_dropped",
			std::chrono::duration<double, std::milli>(elapsed).count());
		return;
	}

	// 4. Run inferencer pipeline (FaceMesh -> warp -> composite -> JPEG)
	//    If filter is disabled, inferencer still encodes the raw frame as JPEG.
	auto result = inferencer_->processFrame(
		frame.data, frame.width, frame.height,
		outSlot, kMaxJpegBytesPerSlot);

	if (!result) {
		OE_LOG_WARN("face_filter_process_error: error={}", result.error());
		return;
	}
	const std::size_t jpegSize = result.value();

	// 4a. Check if the full pipeline exceeded the deadline
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

	// 5. GPU fence — block until GPU has finished writing JPEG
	gpuFence_.record(inferencer_->cudaStream());
	gpuFence_.synchronize();

	// 6. Stale-read guard — discard if input was overwritten during inference
	if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

	// 7. Commit JPEG to output SHM and publish
	ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
	ctrl->seqNumber[writeSlot] = ++frameSeq_;
	ctrl->writeIndex.store(writeSlot, std::memory_order_release);
	publishFilteredFrame(jpegSize, filtered);
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
	static thread_local nlohmann::json payload = {
		{"v",         kSchemaVersion},
		{"type",      "filtered_frame"},
		{"shm",       ""},
		{"size",      int64_t{0}},
		{"seq",       uint64_t{0}},
		{"filtered",  false},
		{"filter_id", ""},
	};
	payload["shm"]       = config_.outputShmName;
	payload["size"]      = static_cast<int64_t>(jpegSize);
	payload["seq"]       = frameSeq_;
	payload["filtered"]  = filtered;
	payload["filter_id"] = activeFilterId_;
	messageRouter_.publish("filtered_frame", payload);
}

