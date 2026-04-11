#pragma once

#include <chrono>
#include <cstdint>

// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Profiler Compile-Time Constants
//
// Probe retry policy, device selection defaults, and GPU profiler tuning.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// GPU probe retry policy
// ---------------------------------------------------------------------------

/// Maximum number of CUDA device probe attempts before propagating the error.
/// Three attempts covers transient driver-busy after WSL2 cold start.
inline constexpr int kProbeMaxAttempts = 3;

/// Delay between probe retry attempts (milliseconds).
/// 500 ms is long enough for the CUDA driver to recover from a transient
/// failure but short enough to keep daemon startup under 2 s worst-case.
inline constexpr std::chrono::milliseconds kProbeRetryDelay{500};

