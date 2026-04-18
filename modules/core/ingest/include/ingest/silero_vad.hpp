#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tl/expected.hpp>

#include "zmq/audio_constants.hpp"
#include "common/constants/ingest_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — Silero VAD v5 Wrapper (CPU, ONNX Runtime)
//
// Wraps the stateful Silero VAD v5 ONNX model running on the CPU execution
// provider.  The model processes 30 ms PCM float32 chunks at 16 kHz mono
// (480 samples per chunk) and outputs a speech probability in [0, 1].
//
// Statefulness: Silero VAD carries internal GRU hidden state across chunks.
// The state is reset by calling resetState() at the start of each PTT session
// (IDLE → LISTENING transition).  Do NOT reset between chunks mid-utterance.
//
// Model file: silero_vad.onnx (from https://github.com/snakers4/silero-vad)
// Expected inputs:
//   input:   [1, 576]    float32  — 64 context + 512 data samples (16 kHz)
//   state:   [2, 1, 128] float32  — GRU state (reset = zeros)
//   sr:      []           int64    — sample rate (16000)
// Expected output:
//   output:  [1, 1]      float32  — speech probability
//   stateN:  [2, 1, 128] float32  — updated state
// ---------------------------------------------------------------------------

// Forward declarations to avoid including OnnxRuntime headers in this header
namespace Ort { class Session; class Env; class SessionOptions; }


/**
 * @brief Silero VAD v5 ONNX inference wrapper (CPU-only, stateful).
 *
 * Non-copyable; owns the ONNX Runtime session and GRU state tensors.
 */
class SileroVad {
public:
	/** @brief Runtime configuration. */
	struct Config {
		// Silero VAD v5 ONNX — lightweight CPU-only voice activity detector.
		// Runs stateful GRU inference on 30 ms audio chunks (480 samples at 16 kHz)
		// to classify speech vs. silence. No GPU VRAM required.
		std::string modelPath           = "models/silero_vad.onnx";
		float       speechThreshold     = 0.5f;   ///< probability ≥ threshold → speech
		uint32_t    sampleRateHz        = kSttInputSampleRateHz;
		uint32_t    chunkSamples        = kVadChunkSampleCount;  ///< 30 ms at 16 kHz
	};

	/**
	 * @brief Load the ONNX model and allocate state tensors.
	 * @throws std::runtime_error if the model file is missing or fails to load.
	 */
	explicit SileroVad(const Config& config);

	~SileroVad();

	SileroVad(const SileroVad&)            = delete;
	SileroVad& operator=(const SileroVad&) = delete;

	/**
	 * @brief Classify a single 30 ms PCM chunk.
	 *
	 * Updates internal GRU state.  Input must be exactly cfg_.chunkSamples
	 * float32 samples at cfg_.sampleRateHz.
	 *
	 * @param pcm  Non-owning view of float32 PCM samples.
	 * @return     Speech probability in [0.0, 1.0], or error string.
	 */
	[[nodiscard]] tl::expected<float, std::string>
	classify(std::span<const float> pcm);

	/**
	 * @brief Reset GRU hidden state to zeros.
	 *
	 * Must be called at the start of each PTT session (IDLE → LISTENING).
	 * Do NOT call between chunks mid-utterance.
	 */
	void resetState();

	/** @brief true if the last classify() result exceeded speechThreshold. */
	[[nodiscard]] bool isSpeech(float probability) const noexcept
	{
		return probability >= config_.speechThreshold;
	}

private:
	Config config_;

	// ONNX Runtime objects — PImpl to avoid exposing ORT headers in this file
	struct OrtState;
	std::unique_ptr<OrtState> ort_;

	// GRU unified state tensor [2, 1, 128] — 256 float32
	std::vector<float> state_;

	// Context window: last 64 samples from previous chunk (16 kHz).
	// Prepended to each new chunk before inference, as required by Silero VAD v5.
	std::vector<float> context_;

	// Pre-allocated input buffer [576 floats] — reused across classify() calls
	// to keep it warm in L1/L2 cache at 33 Hz invocation frequency.
	std::array<float, kSileroVadTotalInputSamples> inputBuf_{};
};

