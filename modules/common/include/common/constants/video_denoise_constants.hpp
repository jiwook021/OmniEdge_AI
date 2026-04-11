#pragma once

#include <cstddef>
#include <cstdint>

#include "zmq/jpeg_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — BasicVSR++ Video Denoising Compile-Time Constants
//
// Model-intrinsic values for the BasicVSR++ temporal video denoiser.
// Changing these requires re-exporting / re-building the ONNX model.
// ---------------------------------------------------------------------------


/// Number of temporal frames in the BasicVSR++ sliding window.
/// The model processes N consecutive frames to exploit temporal coherence.
inline constexpr uint32_t kTemporalWindowFrameCount = 5;

/// Frame pacing interval for the /denoise_video WebSocket channel (milliseconds).
/// The WS Bridge reads /oe.vid.denoise at this fixed cadence (20 fps).
inline constexpr int kDenoiseFramePacingMs = 50;

/// Maximum allowed inference time (milliseconds) before the denoise node
/// falls back to raw-frame passthrough. BasicVSR++ is temporal and
/// naturally slower; 100 ms gives it 2× headroom over the pacing interval.
inline constexpr int kDenoiseDeadlineMs = 100;

