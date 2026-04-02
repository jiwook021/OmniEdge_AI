#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "zmq/message_router.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "shm/shm_mapping.hpp"
#include "denoise_inferencer.hpp"
#include "gpu/cuda_fence.hpp"
#include "common/constants/video_denoise_constants.hpp"
#include "zmq/zmq_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — VideoDenoiseNode
//
// Optional video enhancement module that applies BasicVSR++ temporal
// denoising to the raw camera feed. Spawned on-demand when the user
// enables "BasicVSR++" from the web UI; killed when disabled.
//
// IPC contracts:
//   ZMQ SUB (port 5555) — "video_frame" from VideoIngestNode (conflated)
//   ZMQ SUB (port 5570) — "ui_command" from WebSocketBridge
//   ZMQ PUB (port 5568) — "denoised_frame"
//   SHM consumer        — /oe.vid.ingest (BGR24 double-buffered)
//   SHM producer        — /oe.vid.denoise (JPEG double-buffered)
//
// Hot path:
//   video_frame notification → SHM read (BGR24) → ring buffer accumulate
//   → BasicVSR++ ONNX inference (temporal window of N frames)
//   → JPEG encode → SHM write → ZMQ "denoised_frame"
//
// Thread safety:
//   initialize() and stop() must be called on the same thread as run().
//   stop() is safe from a signal-handler thread (atomic flag + PAIR socket).
// ---------------------------------------------------------------------------


/**
 * @brief Video denoising pipeline node: temporal frame window → ONNX → SHM + ZMQ.
 *
 * Instantiated once per process. Inject a custom DenoiseInferencer for unit tests
 * (pass via the two-argument constructor below).
 */
class VideoDenoiseNode : public ModuleNodeBase<VideoDenoiseNode> {
public:
	friend class ModuleNodeBase<VideoDenoiseNode>;

	/**
	 * @brief Runtime configuration — all fields have YAML-friendly defaults.
	 */
	struct Config {
		// Model path — BasicVSR++ ONNX model for temporal video denoising.
		std::string onnxModelPath = "./models/basicvsrpp/basicvsrpp_denoise.onnx";

		// SHM
		std::string shmInput  = "/oe.vid.ingest";
		std::string shmOutput = "/oe.vid.denoise";

		// ZMQ ports
		int subVideoPort    = kVideoIngest;   // 5555
		int subWsBridgePort = kWsBridge;       // 5570
		int pubPort         = kVideoDenoise;   // 5568

		// Temporal window size (number of frames for BasicVSR++)
		uint32_t temporalWindowSize = kTemporalWindowFrameCount;  // 5

		// CUDA
		int cudaStreamPriority = kCudaPriorityVideoDenoise;  // -1

		// Polling
		std::chrono::milliseconds pollTimeout{kPollTimeoutMs};

		// Module identity (for logging)
		std::string moduleName = "video_denoise";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/**
	 * @brief Construct with an injected inferencer.
	 *
	 * @param config   Runtime configuration.
	 * @param inferencer  Concrete denoising inferencer. Must not be nullptr.
	 */
	VideoDenoiseNode(const Config& config,
	                 std::unique_ptr<DenoiseInferencer> inferencer);

	~VideoDenoiseNode();

	VideoDenoiseNode(const VideoDenoiseNode&)            = delete;
	VideoDenoiseNode& operator=(const VideoDenoiseNode&) = delete;

	// -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
	[[nodiscard]] tl::expected<void, std::string> onConfigure();
	[[nodiscard]] tl::expected<void, std::string> onLoadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
	// -----------------------------------------------------------------------
	// Message handlers
	// -----------------------------------------------------------------------

	/** Process one video_frame notification from VideoIngestNode. */
	void handleVideoFrame(const nlohmann::json& msg);

	/**
	 * @brief Read the latest BGR24 frame from input SHM.
	 *
	 * Uses double-buffer stale-read guard (compare writeIndex before/after).
	 *
	 * @return BGR24 frame data, or empty vector on stale read.
	 */
	[[nodiscard]] std::vector<uint8_t> readLatestBgrFrame();

	/**
	 * @brief Run denoising when temporal window is full.
	 *
	 * Calls inferencer_->processFrames() with the current window, writes
	 * JPEG output to /oe.vid.denoise SHM, publishes ZMQ notification.
	 */
	void runDenoise();

	// -----------------------------------------------------------------------
	// Members
	// -----------------------------------------------------------------------

	Config                           config_;

	// Inference inferencer
	std::unique_ptr<DenoiseInferencer>  inferencer_;

	// Temporal frame ring buffer — stores N most recent BGR24 frames
	std::vector<std::vector<uint8_t>> frameWindow_;
	uint32_t                          frameWindowHead_{0};
	uint32_t                          frameWindowCount_{0};

	// SHM mappings
	std::unique_ptr<ShmMapping>      shmInput_;
	std::unique_ptr<ShmMapping>      shmOutput_;

	// Networking — single MessageRouter replaces all ZMQ boilerplate
	MessageRouter                    messageRouter_;

	// Reusable frame pointer array — avoids heap allocation per runDenoise() call
	std::vector<const uint8_t*> framePtrsBuffer_;

	// Sequence counter for SHM header and denoised_frame ZMQ messages
	uint64_t seqNumber_{0};

	// GPU→host synchronization fence — created once, reused every frame.
	CudaFence gpuFence_;
};

