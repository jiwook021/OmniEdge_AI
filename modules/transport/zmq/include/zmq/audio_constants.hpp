#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Audio Pipeline Compile-Time Constants
//
// Defines sample rates, VAD chunk sizing, and shared-memory ring-buffer
// layout used by AudioIngestNode (producer) and STTNode (consumer).
//
// These constants are compile-time invariants. For runtime configuration,
// see common/runtime_defaults.hpp and omniedge_config.yaml.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Sample rates
// ---------------------------------------------------------------------------

/// STT input sample rate — Whisper requires 16 kHz mono float32.
inline constexpr uint32_t kSttInputSampleRateHz = 16'000;

/// TTS output sample rate — Kokoro outputs 24 kHz mono float32.
inline constexpr uint32_t kTtsOutputSampleRateHz = 24'000;

// ---------------------------------------------------------------------------
// Voice Activity Detection (VAD) chunk sizing
// ---------------------------------------------------------------------------

/// VAD chunk duration in milliseconds (Silero VAD minimum = 30 ms).
inline constexpr uint32_t kVadChunkDurationMs = 30;

/// Derived: samples per VAD chunk at the STT sample rate.
/// 16000 Hz × 30 ms / 1000 = 480 samples.
inline constexpr uint32_t kVadChunkSampleCount =
    kSttInputSampleRateHz * kVadChunkDurationMs / 1000;  // = 480

// ---------------------------------------------------------------------------
// Shared-memory ring-buffer layout (/oe.aud.ingest)
// ---------------------------------------------------------------------------

/// Number of ring-buffer slots in the audio SHM segment.
/// 4 slots at 30 ms each = 120 ms of audio buffering before stale reads.
inline constexpr std::size_t kRingBufferSlotCount = 4;

/// Byte size of one audio SHM slot for kVadChunkSampleCount samples.
/// Padded to 8-byte alignment so that ShmAudioHeader casts are aligned.
/// ShmAudioHeader = 24 bytes (uint32 sampleRateHz + uint32 numSamples + uint64 seqNumber + uint64 timestampNs).
inline constexpr std::size_t kAudioShmSlotByteSize =
    ((24 + kVadChunkSampleCount * sizeof(float)) + 7) & ~std::size_t{7};

