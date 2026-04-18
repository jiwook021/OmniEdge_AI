#include "ingest/video_ingest_node.hpp"
#include "ingest/gstreamer_pipeline.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_logger.hpp"
#include "common/time_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


namespace {

constexpr uint32_t kVideoSlotCount = kCircularBufferSlotCount;  // 4

/// Persistent storage for the last-applied image_adjust values so the
/// adjustment the user set survives a daemon restart. Matches the frontend
/// localStorage key "oe.imageAdjust.v1" — same schema.
std::filesystem::path imageAdjustStatePath()
{
	const char* home = std::getenv("HOME");
	const std::filesystem::path base = home && *home
		? std::filesystem::path(home) / ".omniedge"
		: std::filesystem::path("/tmp");
	return base / "image_adjust.json";
}

void writeImageAdjustState(float brightness, float contrast, float saturation, float sharpness)
{
	std::error_code ec;
	const auto path = imageAdjustStatePath();
	std::filesystem::create_directories(path.parent_path(), ec);
	nlohmann::json j = {
		{"brightness", brightness},
		{"contrast",   contrast},
		{"saturation", saturation},
		{"sharpness",  sharpness},
	};
	const auto tmp = path;
	std::ofstream out(tmp.string() + ".tmp", std::ios::binary | std::ios::trunc);
	if (!out) return;
	out << j.dump();
	out.close();
	std::filesystem::rename(tmp.string() + ".tmp", path, ec);
}

struct ImageAdjustState { float brightness{0}, contrast{1}, saturation{1}, sharpness{0}; bool loaded{false}; };

ImageAdjustState readImageAdjustState()
{
	ImageAdjustState s;
	const auto path = imageAdjustStatePath();
	std::ifstream in(path, std::ios::binary);
	if (!in) return s;
	try {
		nlohmann::json j; in >> j;
		s.brightness = j.value("brightness", 0.0f);
		s.contrast   = j.value("contrast",   1.0f);
		s.saturation = j.value("saturation", 1.0f);
		s.sharpness  = j.value("sharpness",  0.0f);
		s.loaded = true;
	} catch (...) { /* stale / bad json — treat as no-op neutral */ }
	return s;
}

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
	// Restore the user's previously-applied image adjustment so camera output
	// looks the same across daemon restarts. Frontend mirrors this in
	// localStorage; the backend file is authoritative when both exist.
	const auto s = readImageAdjustState();
	if (s.loaded) {
		brightness_.store(s.brightness, std::memory_order_relaxed);
		contrast_.store(s.contrast,     std::memory_order_relaxed);
		saturation_.store(s.saturation, std::memory_order_relaxed);
		OE_LOG_INFO("image_adjust_restored: brightness={:.2f}, contrast={:.2f}, saturation={:.2f}",
			s.brightness, s.contrast, s.saturation);
	}
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
// configureTransport
// ---------------------------------------------------------------------------

tl::expected<void, std::string> VideoIngestNode::configureTransport()
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
			handleUiCommand(action, msg);
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
// loadInferencer — build GStreamer pipeline object (do NOT start it)
// ---------------------------------------------------------------------------

tl::expected<void, std::string> VideoIngestNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	// Build GStreamer pipeline.
	// Three source modes:
	//   1. File path (.mp4/.mkv/.avi/.webm/.mov/.ts) → filesrc + decodebin
	//   2. V4L2 device exists → v4l2src (MJPG capture)
	//   3. Fallback → videotestsrc (synthetic frames for pipeline validation)
	std::string pipelineStr;

	auto isVideoFile = [](const std::string& path) -> bool {
		const std::filesystem::path p(path);
		const auto ext = p.extension().string();
		return ext == ".mp4" || ext == ".mkv" || ext == ".avi"
			|| ext == ".webm" || ext == ".mov" || ext == ".ts";
	};

	if (isVideoFile(config_.v4l2Device) && std::filesystem::exists(config_.v4l2Device)) {
		// File-based input: demux + decode + scale to configured resolution.
		// Used by integration tests feeding real video through the pipeline.
		// Quote the location value — gst_parse_launch() uses spaces as
		// element delimiters, so unquoted paths with spaces would break.
		pipelineStr = std::format(
			"filesrc location=\"{}\" "
			"! decodebin "
			"! videoconvert "
			"! videoscale "
			"! video/x-raw,format=BGR,width={},height={} "
			"! appsink name=video_sink drop=true max-buffers=2",
			config_.v4l2Device,
			config_.frameWidth, config_.frameHeight);
		OE_LOG_INFO("video_ingest_filesrc: path={}, {}x{}",
			config_.v4l2Device, config_.frameWidth, config_.frameHeight);
	} else if (std::filesystem::exists(config_.v4l2Device)) {
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

void VideoIngestNode::handleUiCommand(UiAction action, const nlohmann::json& msg)
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

	case UiAction::kSetImageAdjust: {
		// Slider ranges (from mode UIs in frontend/conversation|security|beauty):
		//   brightness -100..100 (additive), contrast 0.5..3.0 (multiplicative),
		//   saturation 0.0..2.0, sharpness 0..10 (not yet applied — CPU unsharp
		//   mask is too expensive for the hot path; leave as no-op until GPU
		//   compositor grows one).
		const float b = msg.value("brightness", 0.0f);
		const float c = msg.value("contrast",   1.0f);
		const float s = msg.value("saturation", 1.0f);
		const float sharp = msg.value("sharpness", 0.0f);
		const float bClamped = std::clamp(b, -100.0f, 100.0f);
		const float cClamped = std::clamp(c,    0.0f,   4.0f);
		const float sClamped = std::clamp(s,    0.0f,   2.0f);
		brightness_.store(bClamped, std::memory_order_relaxed);
		contrast_.store(cClamped,   std::memory_order_relaxed);
		saturation_.store(sClamped, std::memory_order_relaxed);
		writeImageAdjustState(bClamped, cClamped, sClamped, sharp);
		OE_LOG_INFO("image_adjust_applied: brightness={:.2f}, contrast={:.2f}, saturation={:.2f}, sharpness={:.2f}",
			bClamped, cClamped, sClamped, sharp);
		break;
	}

	default:
		break;
	}
}

// ---------------------------------------------------------------------------
// applyImageAdjust() — in-place BGR24 CPU adjustment
//
// Neutral defaults (brightness=0, contrast=1, saturation=1) short-circuit
// so the hot path is zero-cost when the sliders are untouched. At 1920×1080
// the scalar loop takes ~4 ms; if that becomes a bottleneck the work can
// migrate to the GPU compositor.
// ---------------------------------------------------------------------------
void VideoIngestNode::applyImageAdjust(uint8_t* bgr, std::size_t pixelCount) const
{
	const float brightness = brightness_.load(std::memory_order_relaxed);
	const float contrast   = contrast_.load(std::memory_order_relaxed);
	const float saturation = saturation_.load(std::memory_order_relaxed);

	constexpr float kEps = 1e-3f;
	const bool skipBrightness = std::fabs(brightness)              < kEps;
	const bool skipContrast   = std::fabs(contrast   - 1.0f)       < kEps;
	const bool skipSaturation = std::fabs(saturation - 1.0f)       < kEps;
	if (skipBrightness && skipContrast && skipSaturation) return;

	for (std::size_t i = 0; i < pixelCount; ++i) {
		uint8_t* p = bgr + i * 3;
		float b = static_cast<float>(p[0]);
		float g = static_cast<float>(p[1]);
		float r = static_cast<float>(p[2]);

		if (!skipSaturation) {
			// BT.601 luma weights; BGR order.
			const float gray = 0.114f * b + 0.587f * g + 0.299f * r;
			b = gray + saturation * (b - gray);
			g = gray + saturation * (g - gray);
			r = gray + saturation * (r - gray);
		}
		if (!skipContrast) {
			b = (b - 128.0f) * contrast + 128.0f;
			g = (g - 128.0f) * contrast + 128.0f;
			r = (r - 128.0f) * contrast + 128.0f;
		}
		if (!skipBrightness) {
			b += brightness;
			g += brightness;
			r += brightness;
		}

		p[0] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
		p[1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
		p[2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
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

	// Apply brightness/contrast/saturation in-place on the SHM slot so every
	// downstream consumer (blur, VLM, ws_bridge preview) sees the adjusted
	// frame. Neutral defaults short-circuit — zero cost when sliders are untouched.
	applyImageAdjust(dst,
		static_cast<std::size_t>(config_.frameWidth) * config_.frameHeight);

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
	static thread_local VideoFrameMsg msg;
	msg.seq = seq;
	msg.ts  = tsNs;
	messageRouter_.publish("video_frame", nlohmann::json(msg));
}
