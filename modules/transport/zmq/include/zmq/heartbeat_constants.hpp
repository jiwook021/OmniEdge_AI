#pragma once

#include <string_view>

#include "common/runtime_defaults.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — Module Heartbeat / Watchdog / Lifecycle Constants
//
// Defines the heartbeat interval published by each module and the silence
// threshold used by the daemon watchdog to detect dead modules, plus
// lifecycle-related ZMQ topic strings.
//
// Timing values are sourced from runtime_defaults.hpp ( namespace)
// to ensure a single source of truth across the codebase.
// ---------------------------------------------------------------------------


/// How often a running module publishes a heartbeat on its PUB socket (ms).
inline constexpr int kIntervalMs = kHeartbeatIntervalMs;

/// Silence threshold: daemon considers a module dead after this gap (ms).
inline constexpr int kTimeoutMs = kHeartbeatTimeoutMs;

// ---------------------------------------------------------------------------
// Lifecycle ZMQ topic strings
// ---------------------------------------------------------------------------

/// Topic for periodic module heartbeats (all nodes → Daemon watchdog).
inline constexpr std::string_view kZmqTopicHeartbeat    = "heartbeat";

/// Topic published by each module once initialization succeeds.
inline constexpr std::string_view kZmqTopicModuleReady  = "module_ready";

/// Topic for daemon → bridge per-module health updates.
inline constexpr std::string_view kZmqTopicModuleStatus = "module_status";

/// Topic published once daemon + conversation_model are both ready.
/// Frontend gates chat input / mode switches on receipt of this message.
inline constexpr std::string_view kZmqTopicOrchestratorReady = "orchestrator_ready";

/// Topic for TTS sentence chunks routed from Daemon to TTSNode.
inline constexpr std::string_view kZmqTopicTtsSentence  = "tts_sentence";

