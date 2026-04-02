#pragma once

#include <string>

#include "common/runtime_defaults.hpp"
#include "stt_inferencer.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — HallucinationFilter
//
// Whisper frequently generates confident-sounding phantom text from silence
// or background noise (e.g. "Thank you for watching!" or "Subscribe now!").
// Three independent tests catch the most common failure modes:
//
//   1. no_speech_prob  — P(no speech) from decoder token 0.  Threshold: 0.6.
//   2. avg_logprob     — Mean per-token log-probability. Below -1.0 → noise.
//   3. repeat_count    — Same text N times in a row → stuck hallucination.
//
// The filter is stateful (tracks the last transcription for repeat detection)
// and must not be shared across threads.
// ---------------------------------------------------------------------------


/**
 * @brief Stateful post-decode hallucination filter for Whisper outputs.
 *
 * Constructed with threshold config; call isHallucination() after each
 * transcription call.  Call reset() at session boundaries (PTT press) so
 * the repeat counter does not carry over between utterances.
 *
 * Thread safety: none. Use from a single thread.
 */
class HallucinationFilter {
public:
	/**
	 * @brief Threshold configuration (overridable via YAML).
	 *
	 * Default values come from defaults and must not be hardcoded
	 * as bare literals in the implementation file.
	 */
	struct Config {
		/// Discard if P(no_speech) from decoder token 0 exceeds this value.
		float noSpeechProbThreshold = kSttNoSpeechProbThreshold;

		/// Discard if mean per-token log-prob is below this value.
		float minAvgLogprob = kSttMinAvgLogprob;

		/// Discard after the same text repeats this many consecutive times.
		int maxConsecutiveRepeats = kSttMaxConsecutiveRepeats;
	};

	/**
	 * @brief Construct with default filter thresholds from defaults.
	 */
	HallucinationFilter();

	/**
	 * @brief Construct with the given filter thresholds.
	 * @param cfg  Threshold values.
	 */
	explicit HallucinationFilter(const Config& cfg);

	/**
	 * @brief Test whether a transcription result is a hallucination.
	 *
	 * Updates internal repeat-count state.  Returns true (discard) on
	 * any of the three failure conditions; false (keep) otherwise.
	 *
	 * @param result  Transcription to evaluate.
	 * @return true if the result should be discarded; false if it should
	 *         be published.
	 */
	[[nodiscard]] bool isHallucination(const TranscribeResult& result) noexcept;

	/**
	 * @brief Reset the repeat counter and last-text cache.
	 *
	 * Call at every session boundary (e.g. PTT pressed, new conversation
	 * started) so a stuck repeat does not bleed into the next utterance.
	 */
	void reset() noexcept;

private:
	Config      cfg_;
	std::string lastText_;    ///< Last accepted or seen transcription text
	int         repeatCount_; ///< Consecutive count of identical transcriptions
};

