#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "stt_inferencer.hpp"
#include "common/constants/whisper_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — TrtWhisperInferencer
//
// Runs Whisper V3 Turbo inference via the TRT-LLM Executor API using
// encoder-decoder engines built by trtllm-build.
//
// Flow:
//   1. loadModel() creates a tle::Executor with kENCODER_DECODER model type
//   2. transcribe() transposes mel [nMels, nFrames] → [nFrames, nMels],
//      creates a tle::Request with encoderInputFeatures + decoder prompt tokens,
//      and streams decoded token IDs back
//   3. Token IDs are decoded to UTF-8 text via the loaded vocabulary
//
// Encoder input:  log-mel spectrogram [nFrames, 128] float32 (transposed)
// Decoder input:  prompt tokens [SOT, <lang>, TRANSCRIBE, NO_TIMESTAMPS]
// Decoder output: text token IDs (greedy, beam_width=1)
//
// Tokenizer:
//   Loads vocab.json + added_tokens.json from the tokenizer directory.
//   Token IDs → UTF-8 fragments are looked up in a flat vector.
//
// VRAM accounting:
//   currentVramUsageBytes() returns the delta measured at loadModel() time.
// ---------------------------------------------------------------------------


/**
 * @brief TRT-LLM Executor-based Whisper encoder-decoder inferencer.
 *
 * Uses the TRT-LLM C++ Executor API to run both encoder and decoder,
 * which properly handles paged KV cache, cross-attention, and all
 * internal TRT-LLM tensor management.
 *
 * Thread safety: none. Owned and used exclusively by STTNode.
 */
class TrtWhisperInferencer : public STTInferencer {
public:
	/**
	 * @brief Runtime configuration for the inferencer.
	 *
	 * All numeric defaults come from stt/whisper_constants.hpp or common/runtime_defaults.hpp.
	 */
	struct Config {
		Config() = default;

		// Whisper model parameters
		uint32_t numMelBins             = kMelBinCount;         ///< 128
		uint32_t numFrames              = kFramesPerChunk;          ///< 3000
		uint32_t maxDecodeTokens        = kMaxDecoderTokens;          ///< 448

		// Whisper special token IDs
		int tokenStartOfTranscript      = kTokenStartOfTranscript;
		int tokenEndOfText              = kTokenEndOfText;
		int tokenTranscribe             = kTokenTranscribe;
		int tokenNoTimestamps           = kTokenNoTimestamps;
		int tokenEnglish                = kTokenEnglish;
		int vocabularySize              = kVocabularySize;

		// KV cache fraction for the Executor (must be high enough for
		// cross-attention KV cache; Python Whisper example uses 0.9)
		float kvCacheFreeGpuFraction    = 0.9f;
	};

	explicit TrtWhisperInferencer(const Config& config);
	TrtWhisperInferencer();
	~TrtWhisperInferencer() override;

	TrtWhisperInferencer(const TrtWhisperInferencer&)            = delete;
	TrtWhisperInferencer& operator=(const TrtWhisperInferencer&) = delete;

	// STTInferencer overrides
	void loadModel(const std::string& encoderEngineDir,
	               const std::string& decoderEngineDir,
	               const std::string& tokenizerDir) override;

	[[nodiscard]] tl::expected<TranscribeResult, std::string>
	transcribe(std::span<const float> melSpectrogram, uint32_t numFrames) override;

	void   unloadModel() noexcept override;
	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

private:
	// Pimpl — hides TRT-LLM Executor headers from consumers
	struct Impl;
	std::unique_ptr<Impl> pimpl_;

	Config config_;

	// Vocabulary for token-ID → UTF-8 fragment decoding
	std::vector<std::string> vocabulary_;

	// VRAM used by the Executor (measured at loadModel time)
	std::size_t vramUsageBytes_ = 0;

	// ── Helper methods ──────────────────────────────────────────────────

	/**
	 * @brief Load Whisper vocabulary from JSON files in the tokenizer directory.
	 */
	void loadVocabulary(const std::string& tokenizerDir);
};

