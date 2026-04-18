#include <gtest/gtest.h>

#include "stt/hallucination_filter.hpp"
#include "common/runtime_defaults.hpp"

// ---------------------------------------------------------------------------
// HallucinationFilter tests — pure CPU, no GPU or ZMQ required.
// Tests cover all three rejection conditions and the reset() API.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Helper: build a TranscribeResult with controlled fields
// ---------------------------------------------------------------------------
static TranscribeResult makeResult(const std::string& text,
                                   float noSpeechProb,
                                   float avgLogprob)
{
	TranscribeResult r;
	r.text         = text;
	r.noSpeechProb = noSpeechProb;
	r.avgLogprob   = avgLogprob;
	r.language     = "en";
	return r;
}

// ---------------------------------------------------------------------------
// Test fixture with default-threshold filter
// ---------------------------------------------------------------------------
class HallucinationFilterTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		// Use default thresholds from oe_defaults (0.6, -1.0, 3)
		filter_ = HallucinationFilter{};
	}

	HallucinationFilter filter_;
};

// ---------------------------------------------------------------------------
// Condition 1: noSpeechProb threshold
// ---------------------------------------------------------------------------

TEST_F(HallucinationFilterTest, AcceptsWhenNoSpeechProbBelowThreshold)
{
	const auto r = makeResult("hello", 0.1f, -0.3f);
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, RejectsWhenNoSpeechProbAboveThreshold)
{
	const auto r = makeResult("phantom", 0.9f, -0.3f);
	EXPECT_TRUE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, RejectsWhenNoSpeechProbAtThreshold)
{
	// noSpeechProb == 0.6 → use strict > threshold → should NOT be rejected
	const auto r = makeResult("borderline",
	                           kSttNoSpeechProbThreshold,
	                           -0.3f);
	// The filter uses: result.noSpeechProb > threshold (strict greater-than)
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, RejectsJustAboveNoSpeechThreshold)
{
	const auto r = makeResult("noise",
	                           kSttNoSpeechProbThreshold + 0.01f,
	                           -0.3f);
	EXPECT_TRUE(filter_.isHallucination(r));
}

// ---------------------------------------------------------------------------
// Condition 2: avgLogprob threshold
// ---------------------------------------------------------------------------

TEST_F(HallucinationFilterTest, AcceptsWhenAvgLogprobAboveThreshold)
{
	const auto r = makeResult("hello", 0.1f, -0.5f);
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, RejectsWhenAvgLogprobBelowThreshold)
{
	const auto r = makeResult("noise", 0.1f, -2.0f);
	EXPECT_TRUE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, AcceptsWhenAvgLogprobAtThreshold)
{
	// avgLogprob == -1.0 → condition is avgLogprob < threshold → should NOT reject
	const auto r = makeResult("borderline", 0.1f,
	                           kSttMinAvgLogprob);
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, RejectsJustBelowAvgLogprobThreshold)
{
	const auto r = makeResult("noise", 0.1f,
	                           kSttMinAvgLogprob - 0.01f);
	EXPECT_TRUE(filter_.isHallucination(r));
}

// ---------------------------------------------------------------------------
// Condition 3: consecutive repeat detection
// ---------------------------------------------------------------------------

TEST_F(HallucinationFilterTest, FirstOccurrenceIsNotFiltered)
{
	const auto r = makeResult("subscribe now", 0.05f, -0.4f);
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, SecondOccurrenceIsNotFiltered)
{
	const auto r = makeResult("subscribe now", 0.05f, -0.4f);
	(void)filter_.isHallucination(r);  // first
	EXPECT_FALSE(filter_.isHallucination(r));  // second
}

TEST_F(HallucinationFilterTest, ThirdOccurrenceIsFiltered)
{
	// 3 == maxConsecutiveRepeats → filtered on the 3rd occurrence
	const auto r = makeResult("thank you for watching", 0.05f, -0.4f);
	(void)filter_.isHallucination(r);    // 1st — repeatCount = 1
	(void)filter_.isHallucination(r);    // 2nd — repeatCount = 2
	EXPECT_TRUE(filter_.isHallucination(r));  // 3rd — repeatCount = 3 → reject
}

TEST_F(HallucinationFilterTest, DifferentTextResetsRepeatCounter)
{
	const auto r1 = makeResult("subscribe now", 0.05f, -0.4f);
	const auto r2 = makeResult("different text", 0.05f, -0.4f);

	(void)filter_.isHallucination(r1);  // 1st occurrence of r1
	(void)filter_.isHallucination(r1);  // 2nd occurrence of r1
	// Now switch to different text — repeat counter should reset
	EXPECT_FALSE(filter_.isHallucination(r2));  // 1st of r2 — must pass
}

TEST_F(HallucinationFilterTest, RepeatCountExactlyAtThreshold)
{
	// With maxConsecutiveRepeats = 3, the 3rd occurrence is rejected.
	// The 4th occurrence of the same text must also be rejected.
	const auto r = makeResult("looping hallucination", 0.05f, -0.4f);
	EXPECT_FALSE(filter_.isHallucination(r));  // 1
	EXPECT_FALSE(filter_.isHallucination(r));  // 2
	EXPECT_TRUE(filter_.isHallucination(r));   // 3 — at threshold → reject
	EXPECT_TRUE(filter_.isHallucination(r));   // 4 — still same text → reject
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------

TEST_F(HallucinationFilterTest, ResetClearsRepeatCount)
{
	const auto r = makeResult("subscribe now", 0.05f, -0.4f);
	(void)filter_.isHallucination(r);  // 1st
	(void)filter_.isHallucination(r);  // 2nd

	filter_.reset();

	// After reset, the 1st occurrence of the same text must pass
	EXPECT_FALSE(filter_.isHallucination(r));
}

TEST_F(HallucinationFilterTest, ResetAfterRejectionAllowsNextUtterance)
{
	const auto r = makeResult("phantom text", 0.05f, -0.4f);
	(void)filter_.isHallucination(r);
	(void)filter_.isHallucination(r);
	(void)filter_.isHallucination(r);  // now stuck in reject

	filter_.reset();  // new session begins

	// Next occurrence should be accepted regardless
	EXPECT_FALSE(filter_.isHallucination(r));
}

// ---------------------------------------------------------------------------
// Custom threshold configuration
// ---------------------------------------------------------------------------

TEST(HallucinationFilterCustomThresholdTest, CustomNoSpeechThreshold)
{
	HallucinationFilter::Config cfg;
	cfg.noSpeechProbThreshold = 0.9f;  // much more permissive
	HallucinationFilter filter(cfg);

	// 0.7 is below the custom 0.9 threshold — should pass
	const auto r = makeResult("speech", 0.7f, -0.3f);
	EXPECT_FALSE(filter.isHallucination(r));
}

TEST(HallucinationFilterCustomThresholdTest, CustomRepeatLimit)
{
	HallucinationFilter::Config cfg;
	cfg.maxConsecutiveRepeats = 1;  // reject after first occurrence
	HallucinationFilter filter(cfg);

	const auto r = makeResult("noise phrase", 0.05f, -0.4f);
	// 1st occurrence hits the limit of 1
	EXPECT_TRUE(filter.isHallucination(r));
}

