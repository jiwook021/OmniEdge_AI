#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "tts_inferencer.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxKokoroInferencer
//
// ONNX Runtime inferencer for Kokoro v1.0 INT8 TTS.
//
// Inference stack:
//   Text → espeak-ng G2P (C API, no subprocess) → int64_t phoneme token IDs
//   + float32 voice style tensor [1, 256] loaded from .npy
//   + float32 speed scalar
//   → Ort::Session (CUDAExecutionProvider — no CPU fallback)
//   → float32 PCM [1, num_samples] at 24,000 Hz mono
//
// VRAM: ~100 MB for the INT8 ONNX model on CUDA EP.
//
// Phoneme vocabulary:
//   Loaded from a companion JSON file (<voiceDir>/../phoneme_vocab.json) or
//   from the built-in English IPA mapping if no file is present.
//   The vocabulary must match the token IDs the ONNX model was exported with.
//
// Thread safety: NOT thread-safe.  TTSNode serialises all calls.
// ---------------------------------------------------------------------------


/**
 * @brief ONNX Runtime Kokoro v1.0 INT8 TTS inferencer.
 *
 * Uses CUDA Execution Provider for GPU inference.  CUDA is required —
 * loadModel() returns an error if CUDA EP is unavailable.
 */
class OnnxKokoroInferencer : public TTSInferencer {
public:
	/**
	 * @brief Construct with default ORT environment (INFO log level).
	 *
	 * Does NOT load the model.  Call loadModel() before synthesize().
	 */
	OnnxKokoroInferencer();
	~OnnxKokoroInferencer() override;

	OnnxKokoroInferencer(const OnnxKokoroInferencer&)            = delete;
	OnnxKokoroInferencer& operator=(const OnnxKokoroInferencer&) = delete;

	[[nodiscard]] tl::expected<void, std::string> loadModel(
		const std::string& onnxModelPath,
		const std::string& voiceDir) override;

	[[nodiscard]] tl::expected<std::vector<float>, std::string> synthesize(
		std::string_view text,
		std::string_view voiceName,
		float            speed) override;

	void unloadModel() noexcept override;

	[[nodiscard]] std::size_t currentVramUsageBytes() const override;

	[[nodiscard]] std::string name() const override { return "onnx-kokoro"; }

private:
	// -----------------------------------------------------------------------
	// ORT session state
	// -----------------------------------------------------------------------

	Ort::Env                           ortEnv_;
	Ort::SessionOptions                sessionOpts_;
	std::unique_ptr<Ort::Session>      session_;
	Ort::AllocatorWithDefaultOptions   allocator_;

	std::string voiceDir_;
	bool        cudaEpActive_{false};  ///< true once CUDA EP registered (always true after loadModel)
	bool        modelLoaded_{false};

	/// VRAM budget reported to daemon.  Set from kVramKokoroTtsMb on load.
	std::size_t vramUsageBytes_{0};

	// -----------------------------------------------------------------------
	// Voice style cache
	// Voice name → [256] float32 loaded from <voiceDir>/<name>.npy
	// -----------------------------------------------------------------------

	std::unordered_map<std::string, std::vector<float>> voiceCache_;

	// -----------------------------------------------------------------------
	// Phoneme vocabulary: UTF-32 codepoint string → Kokoro token ID
	// Built from the built-in English IPA table or loaded from JSON.
	// -----------------------------------------------------------------------

	std::unordered_map<std::u32string, int64_t> phonemeVocab_;

	// -----------------------------------------------------------------------
	// Private helpers
	// -----------------------------------------------------------------------

	/**
	 * @brief Populate phonemeVocab_ from the built-in English IPA table.
	 *
	 * Called during loadModel() when no companion vocab JSON is found.
	 */
	void loadBuiltinVocab();

	/**
	 * @brief Load phonemeVocab_ from <modelDir>/phoneme_vocab.json.
	 *
	 * @return true on success, false if the file is absent (caller falls back
	 *         to loadBuiltinVocab()).
	 */
	[[nodiscard]] bool loadVocabJson(const std::string& modelDir);

	/// Kokoro style embedding dimension.
	static constexpr std::size_t kStyleDim = 256;

	/**
	 * @brief Retrieve (and cache) the voice style tensor for voiceName.
	 *
	 * Loads <voiceDir_>/<voiceName>.bin (raw float32) or .npy on first
	 * access.  Voice packs are shaped [N, 256]; the row at index
	 * min(tokenCount, N-1) is returned.
	 *
	 * @return Span over the cached 256-element float32 row, or error.
	 */
	[[nodiscard]] tl::expected<std::span<const float>, std::string>
	getVoiceStyle(std::string_view voiceName, std::size_t tokenCount);

	/**
	 * @brief Convert UTF-8 text to Kokoro phoneme token IDs via espeak-ng.
	 *
	 * Uses espeak_TextToPhonemes() (C API, never a subprocess) to obtain
	 * IPA phoneme strings word-by-word, then maps each UTF-32 codepoint
	 * (or multi-codepoint digraph) to a token ID from phonemeVocab_.
	 *
	 * Unknown codepoints are silently skipped with a warn log.
	 *
	 * @param text  UTF-8 input sentence.
	 * @return      Ordered token ID list ready for ONNX input tensor.
	 */
	[[nodiscard]] tl::expected<std::vector<int64_t>, std::string>
	textToPhonemeTokens(std::string_view text) const;

	/**
	 * @brief Parse a float32 .npy file into a vector.
	 *
	 * Supports NumPy format v1.0 and v2.0 with C-order float32 arrays.
	 * The shape is not validated here; callers check the returned size.
	 *
	 * @param path  Filesystem path to the .npy file.
	 * @return      Float values or error string.
	 */
	[[nodiscard]] static tl::expected<std::vector<float>, std::string>
	loadNpy(const std::string& path);

	/**
	 * @brief Load raw little-endian float32 data from a .bin file.
	 *
	 * @param path  Filesystem path to the .bin file.
	 * @return      Float values or error string.
	 */
	[[nodiscard]] static tl::expected<std::vector<float>, std::string>
	loadBinFloat32(const std::string& path);

	/**
	 * @brief Convert a UTF-8 string to UTF-32.
	 *
	 * Used when mapping espeak-ng IPA output (UTF-8) to the phonemeVocab_
	 * keys (std::u32string for stable codepoint comparison).
	 */
	[[nodiscard]] static std::u32string utf8ToUtf32(std::string_view utf8);
};

