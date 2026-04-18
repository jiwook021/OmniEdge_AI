#pragma once

// conversation_constants.hpp -- Shared constants for the OmniEdge conversation module.
//
// The conversation module handles STT+LLM+TTS in a single process.

#include <string_view>


// ---------------------------------------------------------------------------
// ZMQ topic strings
// ---------------------------------------------------------------------------

/// ZMQ topic for incoming prompts published by the daemon.
inline constexpr std::string_view kZmqTopicConversationPrompt   = "conversation_prompt";

/// ZMQ topic for incoming UI commands from WebSocketBridge.
// NOTE: also defined in ws_bridge_constants.hpp — keep in sync.
inline constexpr std::string_view kZmqTopicUiCommand            = "ui_command";

/// ZMQ topic for outgoing per-token conversation responses.
/// Must match the daemon's subscription topic ("llm_response").
inline constexpr std::string_view kZmqTopicConversationResponse = "llm_response";

/// ZMQ in-process (inproc://) address used for interrupt signalling.
inline constexpr std::string_view kZmqInprocInterruptAddress    = "inproc://conversation_interrupt";

// ---------------------------------------------------------------------------
// Module identity
// ---------------------------------------------------------------------------

/// Module name used for logging and INI config key matching.
inline constexpr std::string_view kModuleName = "conversation";

// ---------------------------------------------------------------------------
// Sentence segmentation
// ---------------------------------------------------------------------------

/// Characters treated as sentence-ending punctuation.  When the previous
/// token ends with one of these, a sentence_boundary flag is raised on the
/// next token so the TTS pipeline can start speaking.
inline constexpr std::string_view kSentenceEndPunctuation = ".?!";

// ---------------------------------------------------------------------------
// ZMQ message schema version (v2 for conversation protocol)
// ---------------------------------------------------------------------------

inline constexpr int kConversationSchemaVersion = 2;

// ---------------------------------------------------------------------------
// Conversation model identity
//
// Canonical dispatch enum used by main.cpp, the daemon, and tests. String
// IDs below are the wire-level keys (INI `[conversation_model]`, ZMQ model
// selection messages, CLI --model flag). Always parse/serialize via the
// helpers — do not branch on raw strings.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <optional>

enum class ConversationModelId : std::uint8_t {
    kGemmaE2B,   ///< google/gemma-4-E2B-it (2B effective)
    kGemmaE4B,   ///< google/gemma-4-E4B-it (4B effective, default)
};

inline constexpr std::string_view kModelGemmaE2b = "gemma_e2b";
inline constexpr std::string_view kModelGemmaE4b = "gemma_e4b";

/// Wire-level string for a model id. Stable across process boundaries.
[[nodiscard]] constexpr std::string_view toString(ConversationModelId id) noexcept
{
    switch (id) {
    case ConversationModelId::kGemmaE2B: return kModelGemmaE2b;
    case ConversationModelId::kGemmaE4B: return kModelGemmaE4b;
    }
    return {};
}

/// Parse a wire-level model id; returns nullopt for unknown values.
[[nodiscard]] constexpr std::optional<ConversationModelId>
parseConversationModelId(std::string_view s) noexcept
{
    if (s == kModelGemmaE2b) return ConversationModelId::kGemmaE2B;
    if (s == kModelGemmaE4b) return ConversationModelId::kGemmaE4B;
    return std::nullopt;
}

/// Default model when INI/frontend omits the selection.
inline constexpr ConversationModelId kDefaultConversationModel = ConversationModelId::kGemmaE4B;

// ---------------------------------------------------------------------------
// Video conversation mode
// ---------------------------------------------------------------------------

/// ZMQ topic for video conversation toggle commands from the daemon.
inline constexpr std::string_view kZmqTopicVideoConversation = "video_conversation";

/// Default video SHM segment name (shared with video ingest).
inline constexpr std::string_view kDefaultVideoShmName = "/oe.vid.ingest";

