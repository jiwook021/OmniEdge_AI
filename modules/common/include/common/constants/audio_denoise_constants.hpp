#pragma once

#include <cstddef>
#include <cstdint>

#include "zmq/audio_constants.hpp"
#include "zmq/jpeg_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — DTLN Audio Denoising Compile-Time Constants
//
// Model-intrinsic values for the Dual-signal Transformation LSTM Network
// (DTLN) two-stage audio denoiser.  Changing these requires re-exporting
// the ONNX models.
// ---------------------------------------------------------------------------


/// Samples per DTLN processing frame (stage-1 input window).
inline constexpr uint32_t kProcessingFrameSizeSamples = 512;

/// Hop size between consecutive processing frames (overlap-add stride).
inline constexpr uint32_t kHopSizeSamples = 128;

/// FFT size for the stage-1 Short-Time Fourier Transform.
inline constexpr uint32_t kFftSize = 512;

/// Block length for the overlap-add reconstruction.
inline constexpr uint32_t kOverlapAddBlockLength = 512;

// ---------------------------------------------------------------------------
// SHM layout constants
// ---------------------------------------------------------------------------

/// Total input SHM segment size.
/// = kAudioShmDataOffset + kRingBufferSlotCount(4) * kAudioShmSlotByteSize.
inline constexpr std::size_t kInputAudioShmSegmentByteSize =
    kAudioShmDataOffset + 4u * kAudioShmSlotByteSize;  // 4 ring-buffer slots

/// Maximum float32 samples in one denoised output SHM segment.
/// 2 seconds at 16 kHz = 32,000 samples.
inline constexpr std::size_t kOutputShmMaxSampleCount = 32'000;

/// Total output SHM segment size.
/// = sizeof(ShmAudioHeader) + kOutputShmMaxSampleCount * sizeof(float)
/// = 24 + 32000 * 4 = 128'024.
inline constexpr std::size_t kOutputAudioShmSegmentByteSize =
    24 + kOutputShmMaxSampleCount * sizeof(float);  // ShmAudioHeader(24) + PCM

