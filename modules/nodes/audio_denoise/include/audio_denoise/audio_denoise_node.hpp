#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "zmq/message_router.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "shm/shm_mapping.hpp"
#include "audio_denoise_inferencer.hpp"
#include "zmq/zmq_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — AudioDenoiseNode
//
// Optional audio enhancement module that applies DTLN real-time denoising
// to the microphone feed. Spawned on-demand when the user enables "DTLN"
// from the web UI; killed when disabled.
//
// IPC contracts:
//   ZMQ SUB (port 5556) — "audio_chunk" from AudioIngestNode (conflated)
//   ZMQ PUB (port 5569) — "denoised_audio"
//   SHM consumer        — /oe.aud.ingest (PCM F32 ring buffer)
//   SHM producer        — /oe.aud.denoise (PCM F32 output)
//
// Hot path:
//   audio_chunk notification → SHM read (F32 PCM) → DTLN two-stage inference
//   → SHM write (denoised PCM) → ZMQ "denoised_audio"
//
// The denoised audio is written to a separate SHM segment and relayed to
// the browser for A/B quality comparison. It does NOT replace the STT input.
//
// Thread safety:
//   initialize() and stop() must be called on the same thread as run().
//   stop() is safe from a signal-handler thread (atomic flag + PAIR socket).
// ---------------------------------------------------------------------------


/**
 * @brief Audio denoising pipeline node: PCM chunks → DTLN → SHM + ZMQ.
 *
 * Instantiated once per process. Inject a custom AudioDenoiseInferencer for
 * unit tests (pass via the two-argument constructor below).
 */
class AudioDenoiseNode : public ModuleNodeBase<AudioDenoiseNode> {
public:
	friend class ModuleNodeBase<AudioDenoiseNode>;

	/**
	 * @brief Runtime configuration — all fields have YAML-friendly defaults.
	 */
	struct Config {
		// Model paths — DTLN two-stage ONNX models (~2.5 MB each).
		std::string model1Path = "./models/dtln/dtln_1.onnx";
		std::string model2Path = "./models/dtln/dtln_2.onnx";

		// SHM
		std::string shmInput  = "/oe.aud.ingest";
		std::string shmOutput = "/oe.aud.denoise";

		// ZMQ ports
		int subAudioPort    = kAudioIngest;    // 5556
		int subWsBridgePort = kWsBridge;        // 5570
		int pubPort         = kAudioDenoise;    // 5569

		// CUDA
		int cudaStreamPriority = kCudaPriorityAudioDenoise;  // 0

		// Polling
		std::chrono::milliseconds pollTimeout{kPollTimeoutMs};

		// Module identity (for logging)
		std::string moduleName = "audio_denoise";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/**
	 * @brief Construct with an injected inferencer.
	 *
	 * @param config   Runtime configuration.
	 * @param inferencer  Concrete audio denoising inferencer. Must not be nullptr.
	 */
	AudioDenoiseNode(const Config& config,
	                 std::unique_ptr<AudioDenoiseInferencer> inferencer);

	~AudioDenoiseNode();

	AudioDenoiseNode(const AudioDenoiseNode&)            = delete;
	AudioDenoiseNode& operator=(const AudioDenoiseNode&) = delete;

	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
	// -----------------------------------------------------------------------
	// Message handlers
	// -----------------------------------------------------------------------

	/** Process one audio_chunk notification from AudioIngestNode. */
	void handleAudioChunk(const nlohmann::json& msg);

	// -----------------------------------------------------------------------
	// Members
	// -----------------------------------------------------------------------

	Config                                config_;

	// Inference inferencer
	std::unique_ptr<AudioDenoiseInferencer>  inferencer_;

	// SHM mappings
	std::unique_ptr<ShmMapping>           shmInput_;
	std::unique_ptr<ShmMapping>           shmOutput_;

	// Networking
	MessageRouter                         messageRouter_;

	// Reusable PCM buffer — avoids heap allocation per audio chunk in hot path
	std::vector<float> inputPcmBuffer_;

	// Sequence counter
	uint64_t seqNumber_{0};
};

