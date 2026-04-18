#pragma once

#include <atomic>

#include "beauty_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/pipeline_types.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "zmq/message_router.hpp"
#include "common/constants/video_constants.hpp"
#include "common/constants/beauty_constants.hpp"
#include "gpu/cuda_fence.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// BeautyNode — SHM consumer + FaceMesh inference + beauty pipeline + SHM producer
//
// Data flow:
//   /oe.vid.ingest  ->  [read BGRFrame via ShmCircularBuffer]
//   -> [FaceMesh V2 ONNX inference -> 478 landmarks]
//   -> [Skin mask: YCrCb threshold within face region]
//   -> [Bilateral filter on skin region (smoothing)]
//   -> [Regional brightness (dark circles, tone evening)]
//   -> [Thin-plate spline warp (face/feature reshaping)]
//   -> [ISP: warmth, shadow fill, highlight, sharpen]
//   -> [JPEG encode]
//   -> /oe.cv.beauty.jpeg (SHM double-buffer)
//   -> ZMQ PUB "beauty_frame" on port 5579
//
// Commands handled (via ui_command topic from WebSocketBridge port 5570):
//   toggle_beauty       — enable/disable the beauty pipeline
//   set_beauty_skin     — update skin tab slider values
//   set_beauty_shape    — update shape tab slider values
//   set_beauty_light    — update light tab slider values
//   set_beauty_bg       — update background mode/settings
//   set_beauty_preset   — apply a named preset (Natural, Glamour, etc.)
//
// Thread safety: initialize() and stop() are NOT thread-safe with run().
// stop() is safe to call from a signal handler (sets atomic flag only).
// ---------------------------------------------------------------------------


class BeautyNode : public ModuleNodeBase<BeautyNode> {
public:
	friend class ModuleNodeBase<BeautyNode>;

	struct Config {
		// ZMQ
		int  pubPort         = kBeauty;
		int  videoSubPort    = kVideoIngest;
		int  wsBridgeSubPort = kWsBridge;
		int  zmqSendHighWaterMark      = kPublisherDataHighWaterMark;
		int  zmqHeartbeatIvlMs         = kHeartbeatIntervalMs;
		int  zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
		int  zmqHeartbeatTimeoutMs     = kHeartbeatTimeoutMs;
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Model
		std::string faceMeshOnnxPath;

		// Frame geometry
		uint32_t inputWidth  = kMaxInputWidth;
		uint32_t inputHeight = kMaxInputHeight;

		// Quality
		int   jpegQuality = kBeautyJpegQuality;

		// SHM — input
		std::string inputShmName  = std::string(kBeautyShmInput);
		std::string inputTopic    = "video_frame";

		// SHM — output
		std::string outputShmName = std::string(kBeautyShmOutput);
		std::string outputBgrShmName = "/oe.cv.beauty.bgr";
		OutputFormat outputFormat = OutputFormat::kJpeg;
		uint32_t outputSlotCount = kCircularBufferSlotCount;

		// Module identity (used in log messages and module_ready)
		std::string moduleName = std::string(kBeautyModuleName);
		std::string bgrTopic = "beauty_bgr_frame";

		// Whether beauty is enabled at startup
		bool enabledAtStartup = true;

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/** Constructor — does not allocate GPU or ZMQ resources. */
	explicit BeautyNode(const Config& config);

	/** Inject the inference inferencer. Must be called before initialize().
	 *  Enables stub injection in tests without requiring a GPU.
	 */
	void setInferencer(std::unique_ptr<BeautyInferencer> inferencer) noexcept {
		inferencer_ = std::move(inferencer);
	}

	BeautyNode(const BeautyNode&) = delete;
	BeautyNode& operator=(const BeautyNode&) = delete;

	~BeautyNode();

	// -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

	/** Query whether the beauty pipeline is currently enabled. */
	[[nodiscard]] bool isBeautyEnabled() const noexcept { return beautyEnabled_; }

private:
	Config                        config_;
	MessageRouter                 messageRouter_;
	uint32_t                      frameSeq_{0};

	// SHM
	std::unique_ptr<ShmMapping> shmIn_;   ///< input (consumer)
	std::unique_ptr<ShmMapping> shmOut_;  ///< JPEG output (producer, kJpeg mode only)
	std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shmOutBgr_; ///< BGR24 output (kBgr24 mode)

	// Inference inferencer
	std::unique_ptr<BeautyInferencer> inferencer_;

	// GPU->host synchronization fence — created once, reused every frame.
	CudaFence gpuFence_;

	// Beauty state
	std::atomic<bool> beautyEnabled_{true};
	BeautyParams      beautyParams_;
	bool              modelLoaded_{false};  ///< false → pass-through (missing FaceMesh ONNX)

	// Processes one BGR24 frame through the beauty pipeline
	void processFrame(const nlohmann::json& frameMetadata);

	// Handle a ui_command JSON
	void handleUiCommand(const nlohmann::json& cmd);

	// Publish "beauty_frame" ZMQ notification (JPEG mode)
	void publishBeautyFrame(std::size_t jpegSize);

	// Publish "beauty_bgr_frame" ZMQ notification (BGR24 mode)
};
