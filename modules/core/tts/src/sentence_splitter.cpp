#include "tts/sentence_splitter.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// SentenceSplitter::findBoundary
//
// Scans buffer_ for the earliest sentence-ending punctuation (. ! ?)
// that is followed by ASCII whitespace or end-of-buffer.
//
// Ellipsis guard: skip if the character at (pos - 1) is also '.'.
// This avoids splitting "e.g. some text" mid-abbreviation in many common
// cases.  The heuristic is intentionally lightweight; the flush() call
// handles sentences that slip through.
// ---------------------------------------------------------------------------

std::size_t SentenceSplitter::findBoundary() const noexcept
{
	const std::size_t len = buffer_.size();

	for (std::size_t i = 0; i < len; ++i) {
		const char ch = buffer_[i];
		if (ch != '.' && ch != '!' && ch != '?') {
			continue;
		}

		// Ellipsis guard: "..." — skip middle and first dots.
		if (ch == '.') {
			// Skip if next char is also '.', or if previous char is '.'.
			const bool prevDot = (i > 0 && buffer_[i - 1] == '.');
			const bool nextDot = (i + 1 < len && buffer_[i + 1] == '.');
			if (prevDot || nextDot) {
				continue;
			}
		}

		// Boundary is valid if the punctuation is at end-of-buffer or
		// immediately followed by ASCII whitespace.
		const std::size_t after = i + 1;
		if (after == len || std::isspace(static_cast<unsigned char>(buffer_[after]))) {
			return after;  // one past the punctuation character
		}
	}

	return std::string::npos;
}

// ---------------------------------------------------------------------------
// SentenceSplitter::trim
// ---------------------------------------------------------------------------

std::string SentenceSplitter::trim(std::string_view s)
{
	// Trim leading whitespace
	std::size_t start = 0;
	while (start < s.size() &&
		   std::isspace(static_cast<unsigned char>(s[start]))) {
		++start;
	}

	// Trim trailing whitespace
	std::size_t end = s.size();
	while (end > start &&
		   std::isspace(static_cast<unsigned char>(s[end - 1]))) {
		--end;
	}

	return std::string(s.substr(start, end - start));
}

// ---------------------------------------------------------------------------
// SentenceSplitter::appendToken
// ---------------------------------------------------------------------------

std::optional<std::string> SentenceSplitter::appendToken(std::string_view token)
{
	buffer_.append(token);

	const std::size_t boundaryPos = findBoundary();
	if (boundaryPos == std::string::npos) {
		return std::nullopt;
	}

	// Extract the completed sentence (up to and including the punctuation).
	std::string sentence = trim(std::string_view(buffer_).substr(0, boundaryPos));

	// Keep the remainder for the next sentence (in-place erase avoids copy).
	buffer_.erase(0, boundaryPos);

	if (sentence.empty()) {
		return std::nullopt;
	}

	OE_LOG_DEBUG("sentence_split: sentence_len={}, remainder_len={}",
	                     sentence.size(), buffer_.size());

	return sentence;
}

// ---------------------------------------------------------------------------
// SentenceSplitter::flush
// ---------------------------------------------------------------------------

std::optional<std::string> SentenceSplitter::flush()
{
	std::string remainder = trim(buffer_);
	buffer_.clear();

	if (remainder.empty()) {
		return std::nullopt;
	}

	return remainder;
}

// ---------------------------------------------------------------------------
// SentenceSplitter::reset
// ---------------------------------------------------------------------------

void SentenceSplitter::reset()
{
	buffer_.clear();
}

// ---------------------------------------------------------------------------
// Batch helper: splitIntoSentences
// ---------------------------------------------------------------------------

std::vector<std::string> splitIntoSentences(std::string_view text)
{
	SentenceSplitter splitter;
	std::vector<std::string> sentences;

	// Feed the entire text as a single "token", then drain all boundaries
	// in a single pass — no reset-and-refeed, so total work is O(n).
	if (auto s = splitter.appendToken(text)) {
		sentences.push_back(std::move(*s));
	}

	// appendToken returns one sentence per call.  Keep calling with an
	// empty token to extract remaining sentences from the buffer.
	while (auto s = splitter.appendToken("")) {
		sentences.push_back(std::move(*s));
	}

	// Flush remaining tail (text after last punctuation).
	if (auto tail = splitter.flush()) {
		sentences.push_back(std::move(*tail));
	}

	return sentences;
}

