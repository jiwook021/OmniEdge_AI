#pragma once

#include <atomic>

#include "face_filter_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/pipeline_types.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "zmq/message_router.hpp"
#include "common/constants/video_constants.hpp"
#include "gpu/cuda_fence.hpp"
#include "common/constants/cv_constants.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// FaceFilterNode — SHM consumer + FaceMesh inference + texture warp + SHM producer
//
// Data flow (JPEG mode — default, for display):
//   /oe.vid.ingest  ->  [read BGRFrame via ShmCircularBuffer]
//   -> [FaceMesh V2 ONNX inference -> 478 landmarks]
//   -> [EMA smooth landmarks across frames]
//   -> [Per-triangle affine warp of filter texture onto face]
//   -> [Alpha-composite filtered overlay onto original]
//   -> [JPEG encode]
//   -> /oe.cv.facefilter.jpeg (SHM double-buffer)
//   -> ZMQ PUB "filtered_frame" on port 5575
//
// Data flow (BGR24 mode — for pipeline chaining):
//   <configurable input SHM>  ->  [read BGRFrame]
//   -> [FaceMesh + warp + composite on GPU]
//   -> /oe.cv.facefilter.bgr (circular-buffer SHM, BGR24)
//   -> ZMQ PUB "filter_bgr_frame" on port 5575
//
// Commands handled (via ui_command topic from WebSocketBridge port 5570):
//   toggle_face_filter  — enable/disable the face filter pipeline
//   select_filter       — switch active filter (e.g. "dog", "cat", "none")
//
// Thread safety: initialize() and stop() are NOT thread-safe with run().
// stop() is safe to call from a signal handler (sets atomic flag only).
// ---------------------------------------------------------------------------


class FaceFilterNode : public ModuleNodeBase<FaceFilterNode> {
public:
	friend class ModuleNodeBase<FaceFilterNode>;

	struct Config {
		// ZMQ
		int  pubPort         = kFaceFilter;
		int  videoSubPort    = kVideoIngest;
		int  wsBridgeSubPort = kWsBridge;
		int  zmqSendHighWaterMark       = kPublisherDataHighWaterMark;
		int  zmqHeartbeatIvlMs  = kHeartbeatIntervalMs;
		int  zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
		int  zmqHeartbeatTimeoutMs  = kHeartbeatTimeoutMs;
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Model
		std::string faceMeshOnnxPath;

		// Filter assets
		std::string filterManifestPath;

		// Frame geometry
		uint32_t inputWidth  = kMaxInputWidth;
		uint32_t inputHeight = kMaxInputHeight;

		// Quality
		int   jpegQuality = kFaceFilterJpegQuality;

		// SHM — input
		std::string inputShmName  = "/oe.vid.ingest";
		std::string inputTopic    = "video_frame";   ///< ZMQ topic that triggers processFrame

		// SHM — output
		std::string outputShmName = "/oe.cv.facefilter.jpeg";   ///< JPEG mode default
		std::string outputBgrShmName = "/oe.cv.facefilter.bgr"; ///< BGR24 mode default
		OutputFormat outputFormat = OutputFormat::kJpeg;
		uint32_t outputSlotCount = kCircularBufferSlotCount; ///< BGR24 circular buffer slots

		// Module identity (used in log messages and module_ready)
		std::string moduleName = "face_filter";
		std::string bgrTopic = "filter_bgr_frame";

		// Initial filter (empty = passthrough until toggled on)
		std::string activeFilterId;

		// Whether face filter is enabled at startup
		bool enabledAtStartup = false;

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/** Constructor — does not allocate GPU or ZMQ resources. */
	explicit FaceFilterNode(const Config& config);

	/** Inject the inference inferencer. Must be called before initialize().
	 *  Enables stub injection in tests without requiring a GPU.
	 */
	void setInferencer(std::unique_ptr<FaceFilterInferencer> inferencer) noexcept {
		inferencer_ = std::move(inferencer);
	}

	FaceFilterNode(const FaceFilterNode&) = delete;
	FaceFilterNode& operator=(const FaceFilterNode&) = delete;

	~FaceFilterNode();

	// -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

	/** Query whether the face filter pipeline is currently enabled. */
	[[nodiscard]] bool isFilterEnabled() const noexcept { return filterEnabled_; }

	/** Query the active filter ID (empty if none). */
	[[nodiscard]] const std::string& activeFilterId() const noexcept { return activeFilterId_; }

private:
	Config                        config_;
	MessageRouter                 messageRouter_;
	uint32_t                      frameSeq_{0};

	// SHM
	std::unique_ptr<ShmMapping> shmIn_;   ///< /oe.vid.ingest       (consumer)
	std::unique_ptr<ShmMapping> shmOut_;  ///< JPEG output (producer, only in kJpeg mode)

	// BGR24 output SHM (only in kBgr24 mode — pipeline chaining)
	std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shmOutBgr_;

	// Inference inferencer
	std::unique_ptr<FaceFilterInferencer> inferencer_;

	// GPU->host synchronization fence — created once, reused every frame.
	CudaFence gpuFence_;

	// Filter state (atomic for thread-safe access from public API / signal handler)
	std::atomic<bool> filterEnabled_{false};
	std::string       activeFilterId_;

	// Processes one BGR24 frame through the filter pipeline
	void processFrame(const nlohmann::json& frameMetadata);

	// Handle a ui_command JSON (toggle_face_filter, select_filter)
	void handleUiCommand(const nlohmann::json& cmd);

	// Publish "filtered_frame" ZMQ notification (JPEG mode)
	void publishFilteredFrame(std::size_t jpegSize, bool filtered);

	// Publish "filter_bgr_frame" ZMQ notification (BGR24 mode)
};

