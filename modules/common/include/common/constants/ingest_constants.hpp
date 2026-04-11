#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Ingest module constants
//
// Shared-memory layout offsets and model-dimension constants used by
// VideoIngestNode, AudioIngestNode, and SileroVad.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <string_view>

#include "zmq/jpeg_constants.hpp"


// ---------------------------------------------------------------------------
// ZMQ topic strings produced by ingest modules
// ---------------------------------------------------------------------------

/// Topic for raw audio chunks (AudioIngestNode → STTNode, AudioDenoiseNode).
inline constexpr std::string_view kZmqTopicAudioChunk = "audio_chunk";

/// Topic for VAD speech/silence status (AudioIngestNode → STTNode, Daemon).
inline constexpr std::string_view kZmqTopicVadStatus  = "vad_status";

/// Offset of the audio control word within the SHM segment.
inline constexpr std::size_t kAudioShmControlOffset    = 0;

// ---------------------------------------------------------------------------
// Silero VAD GRU hidden-state dimensions
//
//   h: [2, 1, 64]  =>  2 * 1 * 64  =  128 floats
//   c: [2, 1, 64]  =>  2 * 1 * 64  =  128 floats
// ---------------------------------------------------------------------------

/// Number of floats in the Silero VAD hidden state (h).
inline constexpr std::size_t kSileroVadHiddenStateSize = 2 * 1 * 64;   // [2,1,64]

/// Number of floats in the Silero VAD cell state (c).
inline constexpr std::size_t kSileroVadCellStateSize   = 2 * 1 * 64;   // [2,1,64]

/// Silero VAD v5 ONNX model expects exactly 512 float32 samples per input
/// tensor at 16 kHz (regardless of the VAD chunk size from the audio pipeline).
inline constexpr std::size_t kSileroVadModelInputSamples = 512;

