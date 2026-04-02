// conversation_constants.hpp -- Shared constants for the OmniEdge conversation module.
//
// Replaces llm_constants.hpp for the unified conversation architecture (v4.0).
// The conversation module handles STT+LLM+TTS in a single process.

#pragma once

#include <string_view>


// ---------------------------------------------------------------------------
// ZMQ topic strings
// ---------------------------------------------------------------------------

/// ZMQ topic for incoming prompts published by the daemon.
inline constexpr std::string_view kZmqTopicConversationPrompt   = "conversation_prompt";

/// ZMQ topic for incoming UI commands from WebSocketBridge.
inline constexpr std::string_view kZmqTopicUiCommand            = "ui_command";

/// ZMQ topic for outgoing per-token conversation responses.
inline constexpr std::string_view kZmqTopicConversationResponse = "conversation_response";

/// ZMQ in-process (inproc://) address used for interrupt signalling.
inline constexpr std::string_view kZmqInprocInterruptAddress    = "inproc://conversation_interrupt";

// ---------------------------------------------------------------------------
// Module identity
// ---------------------------------------------------------------------------

/// Module name used for logging and YAML config key matching.
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
// Model variant identifiers (used in YAML config and CLI --model flag)
// ---------------------------------------------------------------------------

inline constexpr std::string_view kModelQwenOmni7b = "qwen_omni_7b";
inline constexpr std::string_view kModelQwenOmni3b = "qwen_omni_3b";
inline constexpr std::string_view kModelGemmaE4b   = "gemma_e4b";

