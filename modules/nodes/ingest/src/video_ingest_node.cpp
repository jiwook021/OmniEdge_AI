#include "ingest/video_ingest_node.hpp"
#include "ingest/gstreamer_pipeline.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_logger.hpp"
#include "common/time_utils.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>

#include <nlohmann/json.hpp>


namespace {

constexpr uint32_t kVideoSlotCount = kCircularBufferSlotCount;  // 4

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

VideoIngestNode::VideoIngestNode(const Config& config)
	: config_(config)
	, messageRouter_(MessageRouter::Config{
		config.moduleName,
		config.pubPort,
		config.zmqSendHighWaterMark,
		config.pollTimeout})
{
}

VideoIngestNode::~VideoIngestNode()
{
	stop();
}

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<VideoIngestNode::Config, std::string>
VideoIngestNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePositive("frameWidth", static_cast<int>(raw.frameWidth));
	v.requirePositive("frameHeight", static_cast<int>(raw.frameHeight));
	v.requireNonEmpty("v4l2Device", raw.v4l2Device);
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// ---------------------------------------------------------------------------
// onConfigure
// ---------------------------------------------------------------------------

tl::expected<void, std::string> VideoIngestNode::onConfigure()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// 1 — Initialise GStreamer (idempotent if called multiple times)
	gst_init(nullptr, nullptr);

	// 2 — Create POSIX SHM circular buffer producer
	const std::size_t frameBytes =
		static_cast<std::size_t>(config_.frameWidth) * config_.frameHeight * 3u;

	shm_ = std::make_unique<ShmCircularBuffer<ShmVideoHeader>>(
		"/oe.vid.ingest", kVideoSlotCount, frameBytes, /*create=*/true);

	// Write the video header (resolution is fixed for the session)
	auto* videoHeader = shm_->header();
	videoHeader->width         = config_.frameWidth;
	videoHeader->height        = config_.frameHeight;
	videoHeader->bytesPerPixel = 3;

	// 3 — Subscribe to ui_command via MessageRouter (no conflation — control
	//     messages are sequential).  The handler runs on the poll thread.
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) {
			const auto action = parseUiAction(
				msg.value("action", std::string{}));
			handleUiCommand(action);
		});

	// Register per-iteration callback for cross-thread drain and bus checks.
	// This runs on the poll thread (main thread), where ZMQ sends are safe.
	messageRouter_.setOnPollCallback([this]() -> bool {
		// Drain pending video_frame notification from the GStreamer callback
		// thread.  ZMQ sockets are not thread-safe, so all sends happen here.
		if (pendingPublish_.exchange(false, std::memory_order_acquire)) {
			publishFrameNotification(
				lastPublishSeq_.load(std::memory_order_relaxed),
				lastPublishTs_.load(std::memory_order_relaxed));
		}

		// Non-blocking GStreamer bus poll — detect ERROR/EOS from pipeline.
		if (pipeline_->isRunning() && pipeline_->checkBusErrors()) {
			OE_LOG_ERROR("video_ingest_pipeline_error");
			return false;  // break out of poll loop
		}

		return true;  // continue polling
	});

	OE_LOG_INFO("video_ingest_configured: shm=/oe.vid.ingest, size={}, slots={}, pub={}",
		shm_->totalSize(), kVideoSlotCount, config_.pubPort);

	return {};
}

// ---------------------------------------------------------------------------
// onLoadInferencer — build GStreamer pipeline object (do NOT start it)
// ---------------------------------------------------------------------------

tl::expected<void, std::string> VideoIngestNode::onLoadInferencer()
{
	OE_ZONE_SCOPED;

	// Build GStreamer pipeline.
	// Prefer V4L2 webcam capture. If the device path is missing, fall back to
	// videotestsrc so end-to-end pipeline validation still works.
	std::string pipelineStr;
	if (std::filesystem::exists(config_.v4l2Device)) {
		// Most USB webcams over usbipd only support MJPG natively —
		// requesting raw formats via S_FMT fails with "Device busy".
		// Capture as MJPG, decode, then convert to BGR24.
		pipelineStr = std::format(
			"v4l2src device={} "
			"! image/jpeg,width={},height={} "
			"! jpegdec ! videoconvert "
			"! video/x-raw,format=BGR,width={},height={} "
			"! appsink name=video_sink drop=true max-buffers=2",
			config_.v4l2Device,
			config_.frameWidth, config_.frameHeight,
			config_.frameWidth, config_.frameHeight);
	} else {
		pipelineStr = std::format(
			"videotestsrc is-live=true pattern=ball "
			"! video/x-raw,width={},height={},format=BGR "
			"! appsink name=video_sink drop=true max-buffers=2",
			config_.frameWidth, config_.frameHeight);
		OE_LOG_WARN("v4l2_device_missing_fallback_videotestsrc: device={}",
			config_.v4l2Device);
	}

	pipeline_ = std::make_unique<GStreamerPipeline>(
		pipelineStr, "video_sink",
		[this](const uint8_t* data, std::size_t size, uint64_t pts) {
			onVideoFrame(data, size, pts);
		});

	return {};
}

// ---------------------------------------------------------------------------
// onBeforeRun — starts GStreamer pipeline
// ---------------------------------------------------------------------------

void VideoIngestNode::onBeforeRun()
{
	pipeline_->start();
}

// ---------------------------------------------------------------------------
// onAfterStop — stops GStreamer pipeline
// ---------------------------------------------------------------------------

void VideoIngestNode::onAfterStop()
{
	pipeline_->stop();
	OE_LOG_INFO("video_ingest_run_exited");
}

// ---------------------------------------------------------------------------
// handleUiCommand() — dispatch a parsed ui_command on the main thread
// ---------------------------------------------------------------------------

void VideoIngestNode::handleUiCommand(UiAction action)
{
	switch (action) {
	case UiAction::kDisableWebcam:
		pipeline_->stop();
		OE_LOG_INFO("webcam_disabled");
		break;

	case UiAction::kEnableWebcam:
		if (!pipeline_->isRunning()) {
			pipeline_->start();
			OE_LOG_INFO("webcam_enabled");
		}
		break;

	default:
		break;
	}
}

// ---------------------------------------------------------------------------
// onVideoFrame() — GStreamer streaming thread
// ---------------------------------------------------------------------------

void VideoIngestNode::onVideoFrame(const uint8_t* data,
									std::size_t    size,
									uint64_t       /*pts*/)
{
	OE_ZONE_SCOPED;
	const std::size_t frameBytes =
		static_cast<std::size_t>(config_.frameWidth) *
		config_.frameHeight * 3u;

	if (size < frameBytes) {
		OE_LOG_WARN("video_frame_too_small: got={}, expected={}",
			size, frameBytes);
		return;
	}

	// Circular buffer write: acquire next slot, write, commit.
	auto [dst, slotIdx] = shm_->acquireWriteSlot();
	std::memcpy(dst, data, frameBytes);

	// Write per-slot metadata and global header.
	auto* videoHeader = shm_->header();
	const uint64_t seq = frameSeq_.fetch_add(1, std::memory_order_relaxed);
	const uint64_t ts  = static_cast<uint64_t>(steadyClockNanoseconds());

	// Per-slot metadata (ShmSlotMetadata) at slot start is written by the
	// consumer-visible data itself — the header carries global/legacy fields.
	videoHeader->seqNumber   = seq;
	videoHeader->timestampNs = ts;

	// Commit: increments writePos with release semantics.
	shm_->commitWrite();

	OE_LOG_DEBUG("video_frame_written: slot={}, seq={}, frame_bytes={}",
	           slotIdx, seq, frameBytes);

	// Signal the run() loop to publish the ZMQ notification on the main thread.
	// ZMQ sockets are not thread-safe; never call pubSocket_ from this thread.
	lastPublishSeq_.store(seq, std::memory_order_relaxed);
	lastPublishTs_.store(ts, std::memory_order_relaxed);
	pendingPublish_.store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// publishFrameNotification()
// ---------------------------------------------------------------------------

void VideoIngestNode::publishFrameNotification(uint64_t seq, uint64_t tsNs)
{
	static thread_local nlohmann::json msg = {
		{"v",    kSchemaVersion},
		{"type", "video_frame"},
		{"shm",  "/oe.vid.ingest"},
		{"seq",  uint64_t{0}},
		{"ts",   uint64_t{0}},
	};

	msg["seq"] = seq;
	msg["ts"]  = tsNs;

	messageRouter_.publish("video_frame", msg);
}

