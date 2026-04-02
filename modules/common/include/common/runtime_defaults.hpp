#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Runtime Default Values
//
// Fallback defaults for module Config structs. Each constant here is
// overridden at runtime by omniedge_config.yaml when the key is present.
//
// Module-specific compile-time constants (model dimensions, token IDs, etc.)
// live in per-module *_constants.hpp headers — NOT here.
//
// Cross-cutting infrastructure constants live in their own headers:
//   - zmq_transport/port_settings.hpp    — ZMQ port assignments
//   - gpu/cuda_priority.hpp              — CUDA stream priorities
//   - vram/vram_thresholds.hpp           — VRAM budgets and tier thresholds
//   - zmq_transport/zmq_constants.hpp    — ZMQ high-water marks, poll timeout
//   - zmq_transport/heartbeat_constants.hpp — heartbeat interval/timeout
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-cutting infrastructure constants live in their own headers.
// Include them directly where needed:
//   - zmq/port_settings.hpp        — ZMQ port assignments ( namespace)
//   - gpu/cuda_priority.hpp        — CUDA stream priorities
//   - vram/vram_thresholds.hpp     — VRAM budgets and tier thresholds ()
//   - zmq/zmq_constants.hpp        — ZMQ high-water marks, poll timeout
//   - zmq/heartbeat_constants.hpp  — heartbeat interval/timeout
//
// This header does NOT pull them in transitively — each module includes
// only what it needs, avoiding dependency leaks across the layer DAG.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// ZMQ message schema version — every JSON payload must carry "v": kSchemaVersion.
// Moved here from port_settings.hpp so non-ZMQ modules can reference it
// without depending on the zmq_transport header directly.
// ---------------------------------------------------------------------------
inline constexpr int kSchemaVersion = 1;


// ---------------------------------------------------------------------------
// Module lifecycle timing (used by daemon watchdog and module launchers)
// ---------------------------------------------------------------------------


/// ZMQ poll timeout — governs worst-case stop() latency.
inline constexpr std::chrono::milliseconds kZmqPollTimeout{50};

/// Silence duration (ms) before AudioIngestNode declares end-of-utterance.
inline constexpr std::uint32_t kVadSilenceDurationMs = 800;

/// Base delay for exponential restart backoff (base × 2^attempt).
inline constexpr unsigned kRestartBaseBackoffMs = 500;

/// Maximum restart backoff cap (prevents unbounded wait).
inline constexpr unsigned kRestartMaxBackoffMs = 30000;

/// Maximum attempts to verify a process exited after SIGKILL during eviction.
/// Total verification window = kEvictionVerifyMaxAttempts × kEvictionVerifyPollInterval.
inline constexpr int kEvictionVerifyMaxAttempts = 10;

/// Interval between kill(pid, 0) polls during eviction verification.
inline constexpr std::chrono::milliseconds kEvictionVerifyPollInterval{200};

// ---------------------------------------------------------------------------
// Daemon watchdog & module management
// ---------------------------------------------------------------------------

/// How often the daemon watchdog checks child process health (ms).
inline constexpr int kWatchdogPollMs = 1000;

/// Maximum time to wait for a module to publish module_ready after spawn (s).
inline constexpr int kModuleReadyTimeoutS = 30;

/// Maximum restarts before permanently disabling a module for this session.
inline constexpr int kMaxModuleRestarts = 5;

/// SIGTERM → SIGKILL grace period when stopping a module (s).
inline constexpr int kStopGracePeriodS = 5;

/// Timeout waiting for VRAM to become available before restarting a module (ms).
inline constexpr int kVramWaitTimeoutMs = 3000;

// ---------------------------------------------------------------------------
// Heartbeat (module PUB → Daemon watchdog)
// ---------------------------------------------------------------------------

/// How often a running module publishes a heartbeat on its PUB socket (ms).
inline constexpr int kHeartbeatIntervalMs = 1000;

/// Silence threshold: daemon considers a module dead after this gap (ms).
/// Set high enough to cover TRT-LLM engine initialization (20-30s for 7B models).
inline constexpr int kHeartbeatTimeoutMs = 30000;

/// Heartbeat TTL — maximum age before a heartbeat is considered stale (ms).
inline constexpr int kHeartbeatTtlMs = 10000;

// ---------------------------------------------------------------------------
// Subprocess management (SubprocessManager)
// ---------------------------------------------------------------------------

/// Maximum time to wait for a subprocess health probe to return true (s).
inline constexpr int kSubprocessStartupTimeoutS = 30;

/// SIGTERM → SIGKILL grace period for subprocess shutdown (s).
inline constexpr int kSubprocessGracePeriodS = 5;

// ---------------------------------------------------------------------------
// Crash protection (IniConfig / INI-overridable)
// ---------------------------------------------------------------------------

/// Max segfault restarts before permanently disabling a module.
inline constexpr int kMaxSegfaultRestarts = 1;

/// How often the memory watchdog checks /proc/<pid>/statm (ms).
inline constexpr int kMemoryCheckIntervalMs = 2000;

/// Kill a module after this many consecutive failed health checks.
inline constexpr int kMaxUnresponsiveChecks = 10;

// ---------------------------------------------------------------------------
// VRAM watchdog (IniConfig / INI-overridable)
// ---------------------------------------------------------------------------

/// Hard cap — never exceed this total VRAM usage (MiB).
/// Must match kVramBudgetMiB in vram_thresholds.hpp (11 GB usable of 12 GB).
inline constexpr std::size_t kVramWatchdogMaxMb = 11264;

/// How often the VRAM watchdog probes cudaMemGetInfo (ms).
inline constexpr int kVramWatchdogCheckMs = 2000;

/// Log a warning when used VRAM exceeds this (MiB).
inline constexpr std::size_t kVramWatchdogWarningMb = 10000;

/// Start evicting lowest-priority modules when used VRAM exceeds this (MiB).
inline constexpr std::size_t kVramWatchdogCriticalMb = 10500;

// ---------------------------------------------------------------------------
// VRAM gate (pre-spawn check with exponential backoff)
// ---------------------------------------------------------------------------

/// Maximum time to wait for VRAM after evicting a lower-priority module (ms).
inline constexpr int kVramGateMaxWaitMs = 5000;

/// Initial poll interval when checking for freed VRAM (ms).
inline constexpr int kVramGatePollInitialMs = 50;

/// Maximum poll interval (exponential backoff cap) (ms).
inline constexpr int kVramGatePollMaxMs = 500;

/// Safety margin subtracted from available VRAM to prevent fragmentation (MiB).
inline constexpr std::size_t kVramGateSafetyMarginMiB = 100;


// ---------------------------------------------------------------------------
// Runtime defaults for module Config structs
//
// These are YAML/INI-overridable defaults, not compile-time model constants.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// LLM generation parameters (LLMNode / ConversationNode)
// ---------------------------------------------------------------------------

/// Maximum number of new tokens the LLM generates per request.
inline constexpr int kLlmMaxNewTokens = 512;

/// Maximum context window size in tokens (prompt + generated).
inline constexpr int kLlmMaxContextTokens = 4096;

/// Sampling temperature — higher = more creative, lower = more deterministic.
inline constexpr float kLlmTemperature = 0.7f;

/// Nucleus (top-p) sampling probability mass cutoff.
inline constexpr float kLlmTopP = 0.9f;

// ---------------------------------------------------------------------------
// Generation defaults (INI-overridable fallbacks for GenerationParams)
//
// Used when the frontend ZMQ message omits generation_params fields.
// Overridden at runtime by [generation] section in omniedge.ini.
// ---------------------------------------------------------------------------

/// Default sampling temperature for conversation generation.
inline constexpr float kGenerationDefaultTemperature = 0.7f;

/// Default nucleus (top-p) threshold for conversation generation.
inline constexpr float kGenerationDefaultTopP = 0.9f;

/// Default maximum tokens to generate per conversation turn.
inline constexpr int kGenerationDefaultMaxTokens = 2048;

/// Maximum time to wait for LLM generation to complete (s).
inline constexpr int kLlmGenerationTimeoutS = 60;

/// Maximum time for a Stable Diffusion image transform inference (s).
inline constexpr int kImageTransformTimeoutS = 120;

// ---------------------------------------------------------------------------
// Prompt assembly (PromptAssembler)
// ---------------------------------------------------------------------------

/// Maximum total context window in tokens for prompt assembly.
inline constexpr int kPromptMaxContextTokens = 4096;

/// Token budget reserved for the system prompt.
inline constexpr int kPromptSystemTokens = 200;

/// Token budget for dynamic context (face identity, scene description).
inline constexpr int kPromptDynamicContextTokens = 100;

/// Maximum tokens per user conversational turn.
inline constexpr int kPromptMaxUserTurnTokens = 500;

/// Headroom reserved for generation output tokens.
inline constexpr int kPromptGenerationHeadroom = 500;

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------

/// How often session state is auto-saved to disk (s).
inline constexpr int kSessionSaveIntervalS = 30;

/// Maximum staleness before a restored session is considered expired (s).
inline constexpr int kSessionMaxStalenessS = 60;

// ---------------------------------------------------------------------------
// STT hallucination filter thresholds (STTNode)
// ---------------------------------------------------------------------------

/// Reject transcription segments where no-speech probability exceeds this.
inline constexpr float kSttNoSpeechProbThreshold = 0.6f;

/// Reject segments with average log-probability below this floor.
inline constexpr float kSttMinAvgLogprob = -1.0f;

/// Reject segments containing more than N consecutive repeated tokens.
inline constexpr int kSttMaxConsecutiveRepeats = 3;

// ---------------------------------------------------------------------------
// Computer vision defaults (BackgroundBlurNode, FaceRecognitionNode)
// ---------------------------------------------------------------------------

/// Face recognition cosine similarity threshold (higher = stricter matching).
inline constexpr float kFaceRecognitionThreshold = 0.45f;

/// YOLOv8-seg segmentation confidence floor.
inline constexpr float kSegmentationConfidenceThreshold = 0.5f;

/// JPEG encoding quality for blurred frame output (0-100).
inline constexpr int kJpegEncodingQuality = 85;

/// Gaussian blur sigma for background blur effect.
inline constexpr float kBackgroundBlurSigma = 25.0f;

/// Process every Nth video frame for face recognition (skip others).
inline constexpr uint32_t kFaceDetectionFrameSubsample = 3;

// ---------------------------------------------------------------------------
// Inferencer loading retry
// ---------------------------------------------------------------------------

/// Retry delay when a inferencer fails to load (ms). Used by onLoadInferencer() fallback loops.
inline constexpr int kInferencerRetryDelayMs = 500;

// ---------------------------------------------------------------------------
// WebSocket / HTTP server
// ---------------------------------------------------------------------------

/// uWebSockets HTTP + WebSocket listen port.
inline constexpr int kWsHttpPort = 9001;

