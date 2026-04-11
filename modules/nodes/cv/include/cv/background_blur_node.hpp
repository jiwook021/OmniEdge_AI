#pragma once

#include "blur_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/runtime_defaults.hpp"
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
// BackgroundBlurNode — SHM consumer + GPU blur pipeline + SHM producer
//
// Data flow:
//   /oe.vid.ingest  →  [read BGRFrame via pinned buffer]
//   → [YOLOv8-seg person mask on GPU]
//   → [Gaussian blur background + composite on GPU]
//   → [nvJPEG encode on GPU]
//   → /oe.cv.blur.jpeg (double-buffer SHM)
//   → ZMQ PUB "blurred_frame" on port 5567
//
// Thread safety: initialize() and stop() are NOT thread-safe with run().
// stop() is safe to call from a signal handler (sets atomic flag only).
// ---------------------------------------------------------------------------


class BackgroundBlurNode : public ModuleNodeBase<BackgroundBlurNode> {
public:
	friend class ModuleNodeBase<BackgroundBlurNode>;

	struct Config {
		// ZMQ
		int  pubPort         = kBgBlur;
		int  videoSubPort    = kVideoIngest;
		int  wsBridgeSubPort = kWsBridge;
		int  zmqSendHighWaterMark       = kPublisherDataHighWaterMark;
		int  zmqHeartbeatIvlMs  = kHeartbeatIntervalMs;
		int  zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
		int  zmqHeartbeatTimeoutMs  = kHeartbeatTimeoutMs;
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Model
		std::string yolov8EnginePath;

		// Frame geometry (populated from video_frame SHM header at runtime
		// if not overridden here)
		uint32_t inputWidth  = kMaxInputWidth;
		uint32_t inputHeight = kMaxInputHeight;

		// Blur quality
		int   blurKernelSize = kDefaultGaussianBlurKernelSize;
		float blurSigma      = kBackgroundBlurSigma;
		int   jpegQuality    = kJpegEncodingQuality;

		// SHM
		std::string inputShmName  = "/oe.vid.ingest";
		std::string outputShmName = "/oe.cv.blur.jpeg";

		// Module identity (used in log messages and module_ready)
		std::string moduleName = "background_blur";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/** Constructor — does not allocate GPU or ZMQ resources. */
	explicit BackgroundBlurNode(const Config& config);

	/** Inject the inference inferencer. Must be called before initialize().
	 *  Enables mock injection in tests without requiring a GPU.
	 */
	void setInferencer(std::unique_ptr<BlurInferencer> inferencer) noexcept {
		inferencer_ = std::move(inferencer);
	}

	BackgroundBlurNode(const BackgroundBlurNode&) = delete;
	BackgroundBlurNode& operator=(const BackgroundBlurNode&) = delete;

	~BackgroundBlurNode();

	// -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
	Config                        config_;
	MessageRouter                 messageRouter_;
	uint32_t                      frameSeq_{0};

	// SHM
	std::unique_ptr<ShmMapping> shmIn_;   ///< /oe.vid.ingest  (consumer)
	std::unique_ptr<ShmMapping> shmOut_;  ///< /oe.cv.blur.jpeg (producer)

	// Inference inferencer
	std::unique_ptr<BlurInferencer> inferencer_;
	bool processErrorLogged_{false};

	// GPU→host synchronization fence — created once, reused every frame.
	// Ensures GPU has finished writing JPEG before flipSlot() exposes it.
	CudaFence gpuFence_;

	// Processes one BGR24 frame: infer → composite → encode → write SHM → pub
	void processFrame(const nlohmann::json& frameMetadata);

	// Handle a ui_command JSON (ISP sliders, bg_mode toggle)
	void handleUiCommand(const nlohmann::json& cmd);

	// Publish "blurred_frame" ZMQ notification
	void publishBlurredFrame(std::size_t jpegSize, bool blurred = true);
};

