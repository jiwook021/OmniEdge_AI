#include "stt/hallucination_filter.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

HallucinationFilter::HallucinationFilter()
	: HallucinationFilter(Config{})
{
}

HallucinationFilter::HallucinationFilter(const Config& cfg)
	: cfg_(cfg)
	, repeatCount_(0)
{
}

// ---------------------------------------------------------------------------
// isHallucination()
// ---------------------------------------------------------------------------

bool HallucinationFilter::isHallucination(const TranscribeResult& result)
{
	// Test 1: no-speech probability — decoder token 0 P(no_speech).
	// High values indicate the model itself believes there is no speech.
	if (result.noSpeechProb > cfg_.noSpeechProbThreshold) {
		OE_LOG_DEBUG("hallucination_no_speech: no_speech_prob={}, threshold={}",
		                     result.noSpeechProb, cfg_.noSpeechProbThreshold);
		return true;
	}

	// Test 2: average log-probability.
	// Values far below 0 indicate low model confidence across all tokens —
	// characteristic of decoding background noise as speech.
	if (result.avgLogprob < cfg_.minAvgLogprob) {
		OE_LOG_DEBUG("hallucination_low_logprob: avg_logprob={}, threshold={}",
		                     result.avgLogprob, cfg_.minAvgLogprob);
		return true;
	}

	// Test 3: consecutive repeat detection.
	// Whisper tends to output the same hallucinated phrase on every chunk
	// when given consistent background noise.  Suppress after N repeats.
	if (result.text == lastText_) {
		++repeatCount_;
	} else {
		// New text — reset repeat counter and update last-seen text
		repeatCount_ = 1;  // count this occurrence as the first
		lastText_    = result.text;
	}

	if (repeatCount_ >= cfg_.maxConsecutiveRepeats) {
		OE_LOG_WARN("hallucination_repeat: text={}, repeat_count={}, threshold={}",
		                    result.text, repeatCount_, cfg_.maxConsecutiveRepeats);
		return true;
	}

	return false;
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

void HallucinationFilter::reset() noexcept
{
	lastText_.clear();
	repeatCount_ = 0;
}

