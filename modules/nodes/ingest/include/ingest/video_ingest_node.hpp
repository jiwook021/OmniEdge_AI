#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "zmq/message_router.hpp"
#include "zmq/port_settings.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/runtime_defaults.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include <tl/expected.hpp>
#include "shm/shm_circular_buffer.hpp"
#include "common/ui_action.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — VideoIngestNode
//
// Captures 1080p BGR24 video from a USB webcam attached via usbipd V4L2,
// writes frames into double-buffered POSIX shared memory, and notifies
// consumers with a tiny ZMQ JSON message.
//
// IPC contracts:
//   SHM producer : /oe.vid.ingest   (1920×1080×3 BGR24 per slot, 2 slots)
//   ZMQ PUB      : ipc:///tmp/omniedge_5555
//   Topic        : "video_frame"
//   Schema       : {"v":1,"type":"video_frame","shm":"/oe.vid.ingest",
//                   "seq":<uint64>,"ts":<mono_ns>}
//
// GStreamer pipeline (V4L2 via usbipd-win):
//     The USB webcam is attached to WSL2 using usbipd-win, appearing
//     as /dev/video0.  Most webcams over usbipd only support MJPG
//     natively — requesting raw video/x-raw via V4L2 S_FMT fails with
//     "Device busy" and causes extreme latency (frames back up in the
//     kernel driver).  Capture MJPG → jpegdec → BGR24:
//
//     v4l2src device=/dev/video0
//     ! image/jpeg,width=1920,height=1080
//     ! jpegdec ! videoconvert
//     ! video/x-raw,format=BGR,width=1920,height=1080
//     ! appsink name=video_sink drop=true max-buffers=2
//
// Double-buffer design:
//   ShmVideoHeader.writeIndex (std::atomic<uint32_t>) lives at offset 0.
//   Producer writes to slot[1 - readIndex], updates seqNumber+timestampNs,
//   then flips writeIndex with release semantics.
//   Consumers re-read writeIndex after memcpy and discard if it changed
//   (stale-read guard).
// ---------------------------------------------------------------------------


class GStreamerPipeline;

/**
 * @brief Ingests 1080p video frames from a TCP GStreamer source.
 *
 * Thread safety:
 *   initialize() and stop() must be called from the same thread.
 *   run() blocks via MessageRouter::run(); stop() is safe to call from a
 *   signal handler (delegates to MessageRouter::stop()).
 */
class VideoIngestNode : public ModuleNodeBase<VideoIngestNode> {
public:
	friend class ModuleNodeBase<VideoIngestNode>;

	/** @brief Runtime configuration populated from YAML. */
	struct Config {
		// GStreamer source
		std::string v4l2Device          = "/dev/video0"; ///< V4L2 device (usbipd-attached webcam)
		uint32_t    frameWidth         = 1920;
		uint32_t    frameHeight        = 1080;

		// ZMQ
		int         pubPort            = kVideoIngest;
		int         wsBridgeSubPort    = kWsBridge;
		int         zmqHeartbeatIvlMs  = kHeartbeatIntervalMs;
		int         zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
		int         zmqHeartbeatTimeoutMs  = kHeartbeatTimeoutMs;
		int         zmqSendHighWaterMark          = kPublisherDataHighWaterMark;

		// Polling
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Module identity (matches YAML key "video_ingest")
		std::string moduleName         = "video_ingest";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/**
	 * @brief Construct with config.  Does NOT initialise GStreamer or ZMQ.
	 */
	explicit VideoIngestNode(const Config& config);

	~VideoIngestNode();

	VideoIngestNode(const VideoIngestNode&)            = delete;
	VideoIngestNode& operator=(const VideoIngestNode&) = delete;

	[[nodiscard]] tl::expected<void, std::string> onConfigure();
	[[nodiscard]] tl::expected<void, std::string> onLoadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

	void onBeforeRun();   // starts GStreamer pipeline
	void onAfterStop();   // stops GStreamer pipeline

private:
	/** Called on the GStreamer streaming thread for each decoded BGR frame. */
	void onVideoFrame(const uint8_t* data, std::size_t size, uint64_t pts);

	/** Dispatch a parsed ui_command (main thread only). */
	void handleUiCommand(UiAction action);

	/** Publish the video_frame ZMQ notification for the frame just written. */
	void publishFrameNotification(uint64_t seq, uint64_t timestampNs);

	Config                              config_;

	// Shared memory — circular buffer producer
	std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shm_;
	std::atomic<uint64_t>               frameSeq_{0};

	// Cross-thread notification: GStreamer callback thread sets these;
	// run() loop on the main thread drains and publishes via ZMQ.
	// pendingPublish_ is the flag; lastSeq_/lastTs_ are the payload.
	std::atomic<bool>                   pendingPublish_{false};
	std::atomic<uint64_t>               lastPublishSeq_{0};
	std::atomic<uint64_t>               lastPublishTs_{0};

	// Consolidated ZMQ networking
	MessageRouter                       messageRouter_;

	// GStreamer pipeline (initialised in initialize())
	std::unique_ptr<GStreamerPipeline>  pipeline_;
};

