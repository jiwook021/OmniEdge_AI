#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — STTInferencer: pure virtual interface for Whisper inferencers
//
// Design note: the interface takes a pre-computed log-mel spectrogram
// (not raw PCM). Mel computation happens in STTNode::runInference(),
// using the MelSpectrogram helper class, so it is tested independently of
// TensorRT and is exercised in CPU-only CI.
//
// Concrete implementations:
//   TrtWhisperInferencer — TensorRT encoder+decoder (production)
//   MockSTTInferencer    — canned result (unit tests, no GPU required)
// ---------------------------------------------------------------------------


/**
 * @brief Structured result from a single Whisper transcription pass.
 *
 * All fields are populated by the inferencer. HallucinationFilter reads
 * noSpeechProb and avgLogprob to decide whether to discard the result.
 */
struct TranscribeResult {
	std::string text;               ///< Decoded UTF-8 transcription text
	std::string language  = "en";  ///< ISO 639-1 language code (e.g. "en", "ko")
	float noSpeechProb    = 0.0f;  ///< P(no_speech) from decoder token 0  [0, 1]
	float avgLogprob      = 0.0f;  ///< Mean per-token log-probability (≤ 0; higher is better)
};

/**
 * @brief Pure interface for speech-to-text inference inferencers.
 *
 * Thread safety: none. Created and used exclusively from the
 * STTNode's run() thread.
 */
class STTInferencer {
public:
	virtual ~STTInferencer() = default;

	/**
	 * @brief Load TensorRT engines and allocate GPU resources.
	 *
	 * Called once from STTNode::initialize().
	 * @throws std::runtime_error if engines are missing or TRT init fails.
	 *
	 * @param encoderEngineDir  Directory containing the encoder engine file.
	 * @param decoderEngineDir  Directory containing the decoder engine file.
	 * @param tokenizerDir      Directory containing the Whisper tokenizer assets.
	 */
	virtual void loadModel(const std::string& encoderEngineDir,
	                       const std::string& decoderEngineDir,
	                       const std::string& tokenizerDir) = 0;

	/**
	 * @brief Run encoder + greedy decoder on a pre-computed log-mel spectrogram.
	 *
	 * The caller (STTNode) is responsible for computing the mel
	 * spectrogram before invoking this method. The input is a flattened
	 * row-major [nMels × nFrames] float32 array.
	 *
	 * @param melSpectrogram  [nMels × nFrames] row-major float32 mel features.
	 * @param nFrames         Number of time frames in the spectrogram.
	 * @return                TranscribeResult on success; error string on failure.
	 *                        Never throws — all errors are returned via expected.
	 */
	[[nodiscard]] virtual tl::expected<TranscribeResult, std::string>
	transcribe(std::span<const float> melSpectrogram, uint32_t nFrames) = 0;

	/**
	 * @brief Release all TRT contexts and free GPU memory.
	 *
	 * Safe to call multiple times (idempotent). Must not throw.
	 */
	virtual void unloadModel() noexcept = 0;

	/**
	 * @brief Current GPU memory used by loaded engines, in bytes.
	 *
	 * Returns 0 if the model is not loaded. Used by the daemon's
	 * VramTracker for eviction accounting.
	 */
	[[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// Factory functions — defined in their respective .cpp files
// ---------------------------------------------------------------------------

/** @brief Create the production TensorRT Whisper encoder-decoder inferencer. */
[[nodiscard]] std::unique_ptr<STTInferencer> createTrtWhisperInferencer();

/** @brief Create a no-GPU mock inferencer for unit tests. */
[[nodiscard]] std::unique_ptr<STTInferencer> createMockSTTInferencer();

