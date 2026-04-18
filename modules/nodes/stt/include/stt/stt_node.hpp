#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "zmq/message_router.hpp"
#include "common/runtime_defaults.hpp"
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "shm/shm_mapping.hpp"
#include "zmq/zmq_constants.hpp"
#include "zmq/audio_constants.hpp"
#include "zmq/heartbeat_constants.hpp"
#include "common/constants/whisper_constants.hpp"
#include "stt/hallucination_filter.hpp"
#include "stt_inferencer.hpp"
#include "stt/mel_spectrogram.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — STTNode
//
// Data flow:
//   ZMQ SUB "audio_chunk"   (port 5556, ZMQ_CONFLATE) — data plane
//   ZMQ SUB "vad_status"    (port 5556, no conflate)  — end-of-utterance
//   → read PCM from /oe.aud.ingest SHM ring buffer
//   → accumulate into rolling 30 s audioBuffer_
//   → compute 128-bin log-mel spectrogram (CPU, MelSpectrogram class)
//   → STTInferencer::transcribe(melSpec, nFrames)
//   → HallucinationFilter::isHallucination()
//   → publish "transcription" on ZMQ port 5563
//
// Long-form transcription (30 s sliding window):
//   When audioBuffer_ reaches kChunkSampleCount (480 000 samples),
//   runInference() is called, the buffer is flushed keeping the last
//   kChunkOverlapSeconds * sampleRateHz samples for boundary continuity.
//
//   vad_status(speaking=false) also triggers an early flush so that the
//   final partial utterance is transcribed without waiting for a full 30 s.
//
// ZMQ notes:
//   audio_chunk uses ZMQ_CONFLATE=1 — only the latest chunk matters.
//   vad_status must NOT be conflated — every end-of-utterance event counts.
//   Two separate SUB sockets are used because ZMQ_CONFLATE is per-socket.
//
// Thread safety:
//   initialize() and stop() must not be called concurrently with run().
//   stop() is async-signal-safe (only stores an atomic and sends one PAIR msg).
// ---------------------------------------------------------------------------


/**
 * @brief STT module node: subscribes to audio_chunk, runs speech-to-text.
 *
 * Inherits lifecycle management from ModuleNodeBase<STTNode> (CRTP).
 * Accepts an optional inferencer injection for unit tests (setInferencer()).
 * A inferencer must be set via setInferencer() before initialize() is called;
 * otherwise loadInferencer() returns an error.
 */
class STTNode : public ModuleNodeBase<STTNode> {
	friend class ModuleNodeBase<STTNode>;

public:
	/**
	 * @brief Runtime configuration populated from YAML by main.cpp.
	 *
	 * All defaults reference defaults or focused constant headers.
	 * No bare numeric literals appear in the .cpp file.
	 *
	 * Grouped by concern: networking → model paths → audio → filter → resources.
	 */
	struct Config {
		// -- ZMQ ports --------------------------------------------------------
		int pubPort      = kStt;          ///< PUB for "transcription" (5563)
		int audioSubPort = kAudioIngest;  ///< AudioIngest PUB to subscribe (5556)

		// -- ZMQ socket tuning ------------------------------------------------
		int zmqSendHighWaterMark    = kPublisherControlHighWaterMark;       ///< PUB HWM (100)
		int zmqHeartbeatIvlMs       = kHeartbeatIntervalMs;   ///< Heartbeat interval (1000 ms)
		int zmqHeartbeatTimeToLiveMs = kHeartbeatTtlMs;     ///< Heartbeat TTL (10000 ms)
		int zmqHeartbeatTimeoutMs   = kHeartbeatTimeoutMs;  ///< Heartbeat timeout (30000 ms)
		std::chrono::milliseconds pollTimeout{kPollTimeoutMs};

		// -- TRT engine paths (required — no default) -------------------------
		std::string encoderEngineDir;  ///< Directory containing encoder .plan/.engine
		std::string decoderEngineDir;  ///< Directory containing decoder .plan/.engine
		std::string tokenizerDir;      ///< Directory containing vocab.json

		// -- Audio / mel spectrogram ------------------------------------------
		uint32_t sampleRateHz   = kSttInputSampleRateHz;     ///< 16 000 Hz
		uint32_t numMelBins     = kMelBinCount;     ///< 128 mel bins
		uint32_t fftWindowSize  = kFftWindowSizeSamples;  ///< 400-sample Hann window
		uint32_t hopLength      = kHopLengthSamples;      ///< 160-sample hop (10 ms)
		uint32_t chunkSamples   = kChunkSampleCount;   ///< 480 000 (30 s window)
		uint32_t overlapSamples = kChunkOverlapSeconds
		                          * kSttInputSampleRateHz;   ///< 16 000 (1 s overlap)

		// -- Hallucination filter thresholds ----------------------------------
		float noSpeechProbThreshold = kSttNoSpeechProbThreshold;  ///< Reject if > 0.6
		float minAvgLogprob         = kSttMinAvgLogprob;          ///< Reject if < -1.0
		int   maxConsecutiveRepeats = kSttMaxConsecutiveRepeats;  ///< Reject on 3x repeat

		// -- Shared memory ----------------------------------------------------
		std::string inputShmName = "/oe.aud.ingest";  ///< POSIX SHM segment (consumer mode)

		// -- CUDA -------------------------------------------------------------
		int cudaStreamPriority = kCudaPriorityStt;  ///< -5 (highest tier)

		// -- Module identity --------------------------------------------------
		std::string moduleName = "stt";  ///< Matches INI key for logging

		/// Factory validator — returns errors for all invalid fields at once.
		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/**
	 * @brief Construct with config. Does NOT acquire any resources.
	 */
	explicit STTNode(const Config& config);

	/**
	 * @brief Inject a custom inference inferencer (optional; primarily for tests).
	 *
	 * Must be called before initialize(). If not called, loadInferencer()
	 * returns an error.
	 */
	void setInferencer(std::unique_ptr<STTInferencer> inferencer) noexcept;

	// Non-copyable — owns CUDA context and MessageRouter
	STTNode(const STTNode&)            = delete;
	STTNode& operator=(const STTNode&) = delete;

	~STTNode();

	// -----------------------------------------------------------------
	// ModuleNodeBase CRTP hooks
	// -----------------------------------------------------------------

	/// Configure resources: logger, CUDA stream, SHM, ZMQ, mel, filter, buffer.
	[[nodiscard]] tl::expected<void, std::string> configureTransport();

	/// Load the TRT inferencer (encoder + decoder engines).
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();

	/// Return the MessageRouter reference required by the base class.
	[[nodiscard]] MessageRouter& router() noexcept { return *messageRouter_; }

	/// Return the module identity string required by the base class.
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
	Config config_;

	// -----------------------------------------------------------------------
	// Networking — MessageRouter (owns ZMQ context, PUB, SUB, interrupt)
	// -----------------------------------------------------------------------
	std::unique_ptr<MessageRouter> messageRouter_;

	// -----------------------------------------------------------------------
	// Processing pipeline (created in configureTransport(), used in run())
	// -----------------------------------------------------------------------
	std::unique_ptr<ShmMapping>          audioInputShm_;        ///< /oe.aud.ingest consumer
	std::unique_ptr<MelSpectrogram>      melProcessor_;         ///< CPU log-mel spectrogram
	std::unique_ptr<STTInferencer>         inferencer_;              ///< TRT encoder+decoder (or mock)
	std::unique_ptr<HallucinationFilter> hallucinationFilter_;  ///< Post-decode reject filter

	cudaStream_t cudaStream_{nullptr};  ///< High-priority stream (priority -5)

	// -----------------------------------------------------------------------
	// Audio accumulator
	// -----------------------------------------------------------------------
	/// Rolling PCM buffer.  Capacity = chunkSamples + overlapSamples.
	std::vector<float> audioBuffer_;

	// -----------------------------------------------------------------------
	// Optional CRTP hook — flush remaining audio after the poll loop exits.
	// -----------------------------------------------------------------------
	void onAfterStop();

	// -----------------------------------------------------------------------
	// Private helpers
	// -----------------------------------------------------------------------

	/**
	 * @brief Read one audio slot from SHM and append to audioBuffer_.
	 *
	 * @param chunkMetadata  Parsed JSON from the "audio_chunk" message.
	 *                       Must contain "slot" (int) and "samples" (int) fields.
	 */
	void processAudioChunk(const nlohmann::json& chunkMetadata);

	/**
	 * @brief Handle a vad_status message — flush audio on end-of-utterance.
	 *
	 * @param msg  Parsed JSON containing "speaking" (bool) field.
	 */
	void handleVadStatus(const nlohmann::json& msg);

	/**
	 * @brief Compute mel spec, call inferencer, apply filter, publish.
	 *
	 * Consumes audioBuffer_: trims to overlapSamples after inference.
	 */
	void runInference();

	/**
	 * @brief Publish a validated TranscribeResult as a "transcription" ZMQ msg.
	 *
	 * @param result     Transcription to publish.
	 * @param latencyMs  End-to-end latency from first chunk to publish (ms).
	 */
	void publishTranscription(const TranscribeResult& result, double latencyMs);
};

