#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — AudioDenoiseInferencer: pure virtual audio denoising interface
//
// All audio denoising inferencers implement this interface, enabling mock
// injection for tests and inferencer swapping with a single YAML field change.
//
// DTLN (Dual-signal Transformation LSTM Network) uses two ONNX models:
//   Stage 1: operates in STFT domain (magnitude masking)
//   Stage 2: operates in time domain (waveform refinement)
//
// Implementations:
//   OnnxDtlnInferencer        — ONNX Runtime + CUDA EP (production)
//   MockAudioDenoiseInferencer — returns pass-through PCM (unit tests)
// ---------------------------------------------------------------------------


/**
 * @brief Pure interface for real-time audio denoising inferencers.
 *
 * The inferencer processes audio in fixed-size frames (512 samples at 16 kHz).
 * It maintains internal LSTM hidden states between calls for temporal context.
 *
 * Thread safety: implementations are NOT thread-safe.
 *  Callers must serialize access.
 */
class AudioDenoiseInferencer {
public:
	virtual ~AudioDenoiseInferencer() = default;

	/**
	 * @brief Load both DTLN ONNX model stages.
	 *
	 * Must be called once before processFrame().
	 *
	 * @param model1Path  Path to dtln_1.onnx (STFT domain stage).
	 * @param model2Path  Path to dtln_2.onnx (time domain stage).
	 * @return            Void on success, error string on failure.
	 */
	[[nodiscard]] virtual tl::expected<void, std::string> loadModel(
		const std::string& model1Path,
		const std::string& model2Path) = 0;

	/**
	 * @brief Denoise one frame of PCM audio.
	 *
	 * Processes a fixed-size frame (512 samples) through the two-stage DTLN
	 * pipeline. Internal LSTM hidden states are carried across calls.
	 *
	 * @param pcmInput  Input PCM samples (F32, 16 kHz mono).
	 * @return          Denoised PCM samples, or error string.
	 */
	[[nodiscard]] virtual tl::expected<std::vector<float>, std::string> processFrame(
		std::span<const float> pcmInput) = 0;

	/**
	 * @brief Reset internal LSTM hidden states.
	 *
	 * Call when starting a new audio stream or after silence gaps.
	 */
	virtual void resetState() = 0;

	/**
	 * @brief Unload model weights and free GPU memory.
	 */
	virtual void unloadModel() = 0;

	/**
	 * @brief Current VRAM usage in bytes.
	 */
	[[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;

	/**
	 * @brief Inferencer identifier string.
	 */
	[[nodiscard]] virtual std::string name() const = 0;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/** @brief Production inferencer: ONNX Runtime with CUDA EP. */
[[nodiscard]] std::unique_ptr<AudioDenoiseInferencer> createOnnxDtlnInferencer();

