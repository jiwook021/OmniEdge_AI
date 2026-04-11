#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — TTSInferencer: pure virtual TTS inference interface
//
// All TTS inferencers implement this interface, enabling mock injection for tests
// and inferencer swapping (Kokoro → VITS2) with a single YAML field change.
//
// Implementations:
//   OnnxKokoroInferencer   — ONNX Runtime + CUDA EP (production)
//   MockTTSInferencer      — returns synthetic PCM (unit tests)
// ---------------------------------------------------------------------------


/**
 * @brief Pure interface for text-to-speech synthesis inferencers.
 *
 * Thread safety: implementations are NOT thread-safe.
 *  Callers must serialize access.
 */
class TTSInferencer {
public:
	virtual ~TTSInferencer() = default;

	/**
	 * @brief Load ONNX model and initialise inference session.
	 *
	 * Must be called once before synthesize().
	 *
	 * @param onnxModelPath  Path to kokoro-v1_0-int8.onnx (or equivalent).
	 * @param voiceDir       Directory containing .npy voice style tensors.
	 * @return               Void on success, error string on failure.
	 */
	[[nodiscard]] virtual tl::expected<void, std::string> loadModel(
		const std::string& onnxModelPath,
		const std::string& voiceDir) = 0;

	/**
	 * @brief Synthesize speech from a UTF-8 text string.
	 *
	 * Performs G2P tokenisation internally. The returned vector contains
	 * F32 PCM samples at 24,000 Hz mono. Values are in [-1.0, 1.0].
	 *
	 * @param text       One sentence of UTF-8 text to synthesize.
	 * @param voiceName  Voice preset name, e.g. "af_heart".
	 * @param speed      Speaking rate multiplier (1.0 = natural speed).
	 * @return           F32 PCM samples, or error string on failure.
	 */
	[[nodiscard]] virtual tl::expected<std::vector<float>, std::string> synthesize(
		std::string_view text,
		std::string_view voiceName,
		float            speed) = 0;

	/**
	 * @brief Unload model weights and free GPU memory.
	 *
	 * Safe to call even if loadModel() was never called.
	 */
	virtual void unloadModel() noexcept = 0;

	/**
	 * @brief Current VRAM usage in bytes.
	 *
	 * Used by OmniEdgeDaemon for eviction accounting.
	 * Returns 0 if the model is not loaded.
	 */
	[[nodiscard]] virtual std::size_t currentVramUsageBytes() const = 0;

	/**
	 * @brief Inferencer identifier string matching the YAML `inferencer:` field.
	 *
	 * e.g. "onnx-kokoro"
	 */
	[[nodiscard]] virtual std::string name() const = 0;
};

// ---------------------------------------------------------------------------
// Factory functions — one per concrete inferencer
// ---------------------------------------------------------------------------

/** @brief Production inferencer: ONNX Runtime with CUDA Execution Provider. */
[[nodiscard]] std::unique_ptr<TTSInferencer> createOnnxKokoroInferencer();

/** @brief Test-only mock inferencer: returns silent PCM at the requested length. */
[[nodiscard]] std::unique_ptr<TTSInferencer> createMockTTSInferencer();

