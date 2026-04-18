#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Error-code taxonomy
//
// Every user-visible log line that reports an anomaly, warning, or error
// carries a stable identifier of the form:
//
//     OE-<MOD>-<SEV><NNN>
//
// where:
//   <MOD>  three-to-eight-letter module prefix (CONV, STT, TTS, CV, DAEMON,
//          ZMQ, SHM, INGEST, WS, GPU, VRAM, ORCH, COMMON)
//   <SEV>  single digit severity band:
//            1 = notable INFO (lifecycle milestone)
//            2 = INFO with numeric payload (metrics, clamps, completions)
//            3 = WARN (degradation, graceful fallback taken)
//            4 = ERROR (request failed, but process continues)
//            5 = CRITICAL (subsystem initialization failed)
//   <NNN>  zero-padded three-digit ordinal, unique within (MOD, SEV).
//
// Codes are only defined here once they actually appear at a call site.
// Do not pre-register codes for planned features — register them when the
// emit-site lands.  Conversely, every string matching the pattern above
// that appears in a log call must have a constant in this header.
//
// Usage:
//
//     #include "common/error_codes.hpp"
//     OE_LOG_WARN("{}: transcription_error: {}", oe_err::kConvTranscriptionError,
//                 result.error());
//
// The README ("Error codes" subsection) mirrors this taxonomy for operators
// who grep the log file.  Keep the two in sync; there is no separate
// `docs/error_codes.md`.
// ---------------------------------------------------------------------------

#include <string_view>

namespace oe_err {

// --- COMMON (shared helpers) ----------------------------------------------
inline constexpr std::string_view kCommonEventBusHandlerThrew = "OE-COMMON-4001";

// --- DAEMON (orchestrator) -------------------------------------------------
inline constexpr std::string_view kDaemonOnnxEpCreateFailed    = "OE-DAEMON-5001";
inline constexpr std::string_view kDaemonScreenCaptureExecFail = "OE-DAEMON-4005";

// --- VAD (voice activity detection, ingest-side) --------------------------
inline constexpr std::string_view kVadInferenceFailed = "OE-VAD-4001";

// --- STT (speech-to-text) -------------------------------------------------
inline constexpr std::string_view kSttAddedTokensParseFailed = "OE-STT-3001";
inline constexpr std::string_view kSttVocabularyEmpty        = "OE-STT-5003";

// --- CONV (conversation node) ---------------------------------------------
inline constexpr std::string_view kConvTtsSidecar              = "OE-CONV-1004";
inline constexpr std::string_view kConvGenParamsClamped        = "OE-CONV-2004";
inline constexpr std::string_view kConvGenerationComplete      = "OE-CONV-2005";
inline constexpr std::string_view kConvPromptMissingText       = "OE-CONV-3002";
inline constexpr std::string_view kConvGenParamsMissing        = "OE-CONV-3003";
inline constexpr std::string_view kConvAudioShmInitFailed      = "OE-CONV-3010";
inline constexpr std::string_view kConvVideoShmInitFailed      = "OE-CONV-3011";
inline constexpr std::string_view kConvVideoFrameUnavailable   = "OE-CONV-3012";
inline constexpr std::string_view kConvVideoOnTextOnlyModel    = "OE-CONV-3013";
inline constexpr std::string_view kConvGenerationError         = "OE-CONV-4003";
inline constexpr std::string_view kConvTranscriptionError      = "OE-CONV-4010";

} // namespace oe_err
