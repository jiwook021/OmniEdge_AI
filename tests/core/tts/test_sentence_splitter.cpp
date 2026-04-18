#include <gtest/gtest.h>

#include "tts/sentence_splitter.hpp"

// ---------------------------------------------------------------------------
// Unit tests for SentenceSplitter and splitIntoSentences.
//
// These tests have no GPU or ZMQ dependency — they run in CI without a
// CUDA device.  Tag: CPU-only (no LABELS gpu).
// ---------------------------------------------------------------------------


// ===========================================================================
// SentenceSplitter — streaming mode
// ===========================================================================

class SentenceSplitterTest : public ::testing::Test {
protected:
	SentenceSplitter splitter_;
};

// ---------------------------------------------------------------------------
// appendToken: basic period boundary
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, AppendTokenDetectsPeriod)
{
	// Tokens feed into the buffer.  The sentence completes when we append
	// the whitespace that follows the period.
	EXPECT_EQ(splitter_.appendToken("Hello"), std::nullopt);
	EXPECT_EQ(splitter_.appendToken(" world"), std::nullopt);
	// The period is part of a token; the space after triggers the boundary.
	auto result = splitter_.appendToken(". ");
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, "Hello world.");
}

TEST_F(SentenceSplitterTest, AppendTokenDetectsExclamation)
{
	(void)splitter_.appendToken("Hello");
	auto result = splitter_.appendToken("! ");
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, "Hello!");
}

TEST_F(SentenceSplitterTest, AppendTokenDetectsQuestion)
{
	(void)splitter_.appendToken("How are you");
	auto result = splitter_.appendToken("? ");
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, "How are you?");
}

// ---------------------------------------------------------------------------
// appendToken: boundary at end-of-buffer (no trailing whitespace)
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, AppendTokenBoundaryAtBufferEnd)
{
	// The period is the last character — treated as end-of-buffer boundary.
	(void)splitter_.appendToken("Done");
	auto result = splitter_.appendToken(".");
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, "Done.");
}

// ---------------------------------------------------------------------------
// appendToken: no boundary yet
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, AppendTokenNoCompletionYet)
{
	auto r1 = splitter_.appendToken("This is");
	auto r2 = splitter_.appendToken(" an incomplete");
	EXPECT_EQ(r1, std::nullopt);
	EXPECT_EQ(r2, std::nullopt);
	EXPECT_EQ(splitter_.pending(), "This is an incomplete");
}

// ---------------------------------------------------------------------------
// appendToken: ellipsis is NOT treated as a sentence boundary
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, AppendTokenEllipsisSkipped)
{
	auto r = splitter_.appendToken("Wait... and then ");
	// No boundary should have been detected — ellipsis skipped.
	EXPECT_EQ(r, std::nullopt);
	EXPECT_FALSE(splitter_.pending().empty());
}

// ---------------------------------------------------------------------------
// flush: returns pending text
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, FlushReturnsPendingText)
{
	(void)splitter_.appendToken("Last sentence");
	auto tail = splitter_.flush();
	ASSERT_TRUE(tail.has_value());
	EXPECT_EQ(*tail, "Last sentence");
	// Buffer should be empty now.
	EXPECT_EQ(splitter_.flush(), std::nullopt);
}

TEST_F(SentenceSplitterTest, FlushOnEmptyBufferReturnsNullopt)
{
	EXPECT_EQ(splitter_.flush(), std::nullopt);
}

// ---------------------------------------------------------------------------
// flush: trims leading/trailing whitespace
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, FlushTrimsWhitespace)
{
	(void)splitter_.appendToken("  spaces  ");
	auto tail = splitter_.flush();
	ASSERT_TRUE(tail.has_value());
	EXPECT_EQ(*tail, "spaces");
}

// ---------------------------------------------------------------------------
// reset: clears buffer
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, ResetClearsBuffer)
{
	(void)splitter_.appendToken("partial sentence");
	splitter_.reset();
	EXPECT_TRUE(splitter_.pending().empty());
	EXPECT_EQ(splitter_.flush(), std::nullopt);
}

// ---------------------------------------------------------------------------
// Multiple sentences in a single token
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, MultiSentenceToken)
{
	// Only the first sentence is returned per appendToken call.
	auto r1 = splitter_.appendToken("First. Second. Third.");
	ASSERT_TRUE(r1.has_value());
	EXPECT_EQ(*r1, "First.");
	// Remainder is in the buffer.
	EXPECT_FALSE(splitter_.pending().empty());
}

// ---------------------------------------------------------------------------
// pending() reflects uncommitted buffer content
// ---------------------------------------------------------------------------

TEST_F(SentenceSplitterTest, PendingReflectsBuffer)
{
	(void)splitter_.appendToken("partial");
	EXPECT_EQ(splitter_.pending(), "partial");
}

// ===========================================================================
// splitIntoSentences — batch mode
// ===========================================================================

TEST(SplitIntoSentencesTest, SingleSentence)
{
	auto result = splitIntoSentences("Hello world.");
	ASSERT_EQ(result.size(), 1u);
	EXPECT_EQ(result[0], "Hello world.");
}

TEST(SplitIntoSentencesTest, TwoSentences)
{
	auto result = splitIntoSentences("Hello. World.");
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0], "Hello.");
	EXPECT_EQ(result[1], "World.");
}

TEST(SplitIntoSentencesTest, MixedPunctuation)
{
	auto result = splitIntoSentences("Really? Yes! Of course.");
	ASSERT_EQ(result.size(), 3u);
	EXPECT_EQ(result[0], "Really?");
	EXPECT_EQ(result[1], "Yes!");
	EXPECT_EQ(result[2], "Of course.");
}

TEST(SplitIntoSentencesTest, TrailingTextWithoutPunctuationIsFlushed)
{
	auto result = splitIntoSentences("First. Incomplete");
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0], "First.");
	EXPECT_EQ(result[1], "Incomplete");
}

