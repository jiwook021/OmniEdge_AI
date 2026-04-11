#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "zmq/message_router.hpp"
#include "zmq/port_settings.hpp"
#include "zmq/zmq_constants.hpp"
#include "zmq/audio_constants.hpp"
#include "common/runtime_defaults.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "shm/shm_circular_buffer.hpp"
#include "common/ui_action.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — AudioIngestNode
//
// Captures 16 kHz PCM audio from a Windows host via TCP, runs Silero VAD
// on every 30 ms chunk, and writes speech-gated chunks to POSIX SHM.
//
// IPC contracts:
//   SHM producer : /oe.aud.ingest   (ring buffer, kAudioShmSlots slots)
//   ZMQ PUB      : ipc:///tmp/omniedge_5556
//   Topics       : "audio_chunk"   — {"v":1,"type":"audio_chunk",
//                                     "shm":"/oe.aud.ingest",
//                                     "slot":<0..N-1>,"samples":<int>,
//                                     "vad":"speech"}
//                  "vad_status"    — {"v":1,"type":"vad_status",
//                                     "speaking":<bool>,
//                                     "silence_ms":<int>}
//
// GStreamer pipeline (Windows host → WSL2):
//   Windows side : wasapi2src ! audioconvert ! tcpserversink
//   WSL2 side    : tcpclientsrc host=<win_ip> port=5001
//                  ! audio/x-raw,format=F32LE,rate=16000,channels=1
//                  ! audioconvert ! audioresample
//                  ! audio/x-raw,format=F32LE,rate=16000,channels=1
//                  ! appsink name=audio_sink drop=true max-buffers=2
//
// VAD behaviour:
//   - Silero VAD runs on CPU per 30 ms chunk (< 1 ms latency).
//   - Only chunks with speechProb >= speechThreshold are written to SHM
//     and produce a ZMQ "audio_chunk" message.
//   - After >= silenceDurationMs of consecutive sub-threshold chunks
//     (following at least one speech chunk), publishes vad_status(speaking=false).
//   - resetVad() must be called when transitioning IDLE → LISTENING.
// ---------------------------------------------------------------------------


class GStreamerPipeline;
class SileroVad;

/**
 * @brief Ingests 16 kHz mono PCM audio with integrated Silero VAD.
 *
 * Thread safety: same contract as VideoIngestNode.
 * run() blocks via MessageRouter::run(); stop() delegates to
 * MessageRouter::stop() and is safe from a signal handler thread.
 */
class AudioIngestNode : public ModuleNodeBase<AudioIngestNode> {
public:
	friend class ModuleNodeBase<AudioIngestNode>;

	/** @brief Runtime configuration populated from YAML. */
	struct Config {
		// GStreamer source
		std::string windowsHostIp       = "172.17.0.1";
		int         audioTcpPort        = 5001;
		uint32_t    sampleRateHz        = kSttInputSampleRateHz;
		uint32_t    chunkSamples        = kVadChunkSampleCount;  ///< 30 ms at 16 kHz

		// VAD — Silero VAD v5 ONNX model; CPU-only stateful GRU.
		// Detects speech onset/offset to gate STT input and prevent
		// Whisper hallucinations on silence. See Architecture § Audio Ingest.
		std::string vadModelPath        = "models/silero_vad.onnx";
		float       vadSpeechThreshold  = 0.5f;
		uint32_t    silenceDurationMs   = kVadSilenceDurationMs;

		// ZMQ
		int         pubPort             = kAudioIngest;
		int         wsBridgeSubPort     = kWsBridge;
		int         zmqHeartbeatIvlMs   = kHeartbeatIntervalMs;
		int         zmqHeartbeatTimeToLiveMs   = kHeartbeatTtlMs;
		int         zmqHeartbeatTimeoutMs   = kHeartbeatTimeoutMs;
		int         zmqSendHighWaterMark           = kPublisherDataHighWaterMark;

		// Polling
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		std::string moduleName          = "audio_ingest";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	explicit AudioIngestNode(const Config& config);
	~AudioIngestNode();

	AudioIngestNode(const AudioIngestNode&)            = delete;
	AudioIngestNode& operator=(const AudioIngestNode&) = delete;

	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

	void onBeforeRun();   // starts GStreamer pipeline
	void onAfterStop();   // stops GStreamer pipeline

private:
	/** Called on GStreamer streaming thread per decoded PCM buffer. */
	void onAudioBuffer(const uint8_t* data, std::size_t size, uint64_t pts);

	/** Write a speech chunk to SHM circular buffer (GStreamer thread). */
	void writeSpeechChunk(const float* samples, uint32_t numSamples);

	/** Publish a single audio_chunk ZMQ notification (main thread only). */
	void publishChunkMsg(uint32_t slot, uint32_t samples, uint64_t ts,
						 uint64_t seq);

	/** Publish a single vad_status ZMQ notification (main thread only). */
	void publishVadMsg(bool speaking, uint32_t silenceMs);

	/** Drain all pending publish messages under lock (main thread only). */
	void drainPendingPublishes();

	/** Dispatch a parsed ui_command (main thread only). Returns true if handled. */
	bool handleUiCommand(UiAction action, const nlohmann::json& cmd);

	/** Reset VAD state and attempt to start the GStreamer pipeline. */
	bool resetVadAndStartPipeline();

	Config                              config_;

	// Pipeline degraded-start tracking.
	// Atomic because the background GStreamer startup thread sets this
	// while the poll thread reads it (see onBeforeRun).
	std::atomic<bool>                   pipelineActive_{false};

	// SHM circular buffer
	std::unique_ptr<ShmCircularBuffer<ShmAudioHeader>> shm_;

	// VAD engine (inference is only called from the GStreamer thread;
	// resetState() is called from the main thread under stateMtx_).
	std::unique_ptr<SileroVad>          vad_;

	// -----------------------------------------------------------------------
	// Mutable state shared between GStreamer streaming thread and main thread.
	// All access must hold stateMtx_.
	// -----------------------------------------------------------------------
	mutable std::mutex                  stateMtx_;
	uint64_t                            chunkSeq_{0};        // guarded by stateMtx_
	bool                                vadSpeaking_{false}; // guarded by stateMtx_
	uint32_t                            silenceSamples_{0};  // guarded by stateMtx_

	// -----------------------------------------------------------------------
	// Pending publish queue — GStreamer thread pushes under stateMtx_;
	// run() drains on the main thread (where ZMQ sends are safe).
	// -----------------------------------------------------------------------
	struct PendingChunkMsg {
		uint32_t slot;
		uint32_t samples;
		uint64_t ts;
		uint64_t seq;
	};
	struct PendingVadMsg {
		bool     speaking;
		uint32_t silenceMs;
	};
	std::vector<PendingChunkMsg>        pendingChunks_;      // guarded by stateMtx_
	std::vector<PendingVadMsg>          pendingVadMsgs_;     // guarded by stateMtx_

	// Consolidated ZMQ networking
	MessageRouter                       messageRouter_;

	// GStreamer
	std::unique_ptr<GStreamerPipeline>   pipeline_;

	// Background thread for GStreamer pipeline startup (tcpclientsrc blocks
	// for 30-60 s). Stored as a member so it can be joined in onAfterStop(),
	// preventing use-after-free if the module is destroyed while the thread
	// is still blocked in pipeline_->start().
	std::thread                         pipelineStartThread_;
};

