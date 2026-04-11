#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// OmniEdge_AI — SentenceSplitter
//
// Streaming sentence-boundary detector for TTS sentence-by-sentence synthesis.
//
// Usage pattern (ZMQ poll loop):
//
//   SentenceSplitter splitter;
//
//   // On each llm_response token:
//   if (auto sentence = splitter.appendToken(token)) {
//       inferencer_.synthesize(*sentence, voice_, speed_);
//   }
//
//   // On finished:true:
//   if (auto tail = splitter.flush()) {
//       inferencer_.synthesize(*tail, voice_, speed_);
//   }
//
// Boundary rule: a sentence ends at '.', '!', or '?' when followed by
// whitespace or end-of-input (not inside an abbreviation like "e.g.").
// This is intentionally simple — edge cases (Mr., Dr., ellipsis ...) that
// slip through are handled at the flush() call on finished:true.
//
// Thread safety: none. External locking required for concurrent access.
// ---------------------------------------------------------------------------


/**
 * @brief Streaming sentence-boundary detector for streaming token input.
 *
 * Accumulates tokens appended one by one and detects sentence boundaries
 * at '.', '!', '?' followed by whitespace or end-of-stream.
 */
class SentenceSplitter {
public:
	SentenceSplitter() = default;

	/**
	 * @brief Append one LLM token to the internal buffer.
	 *
	 * Returns a completed sentence if a boundary was detected.
	 * The returned string is trimmed of leading/trailing whitespace.
	 *
	 * @param token  Token text emitted by the LLM (may contain spaces).
	 * @return       Completed sentence, or std::nullopt if no boundary yet.
	 */
	[[nodiscard]] std::optional<std::string> appendToken(std::string_view token);

	/**
	 * @brief Flush remaining accumulated text as a final sentence.
	 *
	 * Call when the LLM signals finished:true.  Returns the remaining
	 * buffer stripped of leading/trailing whitespace.
	 *
	 * @return  Remaining text if non-empty, std::nullopt otherwise.
	 */
	[[nodiscard]] std::optional<std::string> flush();

	/**
	 * @brief Discard the current accumulation buffer.
	 *
	 * Call when the TTS pipeline is interrupted (daemon flush_tts event).
	 */
	void reset();

	/**
	 * @brief View the pending (unsent) accumulation buffer.
	 *
	 * Useful for diagnostics and tests. Does not modify state.
	 */
	[[nodiscard]] const std::string& pending() const noexcept { return buffer_; }

private:
	std::string buffer_;

	/**
	 * @brief Scan `buffer_` for the first sentence boundary.
	 *
	 * A boundary is: (`.` | `!` | `?`) followed by ASCII whitespace or
	 * end-of-buffer, provided it is not part of an ellipsis ("...").
	 *
	 * @return Index one past the boundary punctuation, or std::string::npos
	 *         if no boundary is found.
	 */
	[[nodiscard]] std::size_t findBoundary() const noexcept;

	/**
	 * @brief Trim leading and trailing ASCII whitespace in-place.
	 */
	[[nodiscard]] static std::string trim(std::string_view s);
};

// ---------------------------------------------------------------------------
// Stateless batch helper
// ---------------------------------------------------------------------------

/**
 * @brief Split a complete text block into sentences.
 *
 * Suitable for batch processing; SentenceSplitter handles streaming.
 * Empty strings are excluded from the result.
 *
 * @param text  UTF-8 input text (may contain multiple sentences).
 * @return      Ordered vector of trimmed sentence strings.
 */
[[nodiscard]] std::vector<std::string> splitIntoSentences(std::string_view text);

