#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "zmq/audio_constants.hpp"
#include "zmq/jpeg_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — Whisper V3 Turbo Compile-Time Constants
//
// All numeric parameters for the Whisper speech-to-text model:
// mel spectrogram dimensions, chunk/window sizing, special token IDs,
// and encoder/decoder tensor shapes.
//
// These are model-intrinsic values — changing them requires re-exporting
// and re-building the TensorRT engines.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ZMQ topic strings produced by the STT module
// ---------------------------------------------------------------------------

/// Topic for transcription results (STTNode → Daemon).
inline constexpr std::string_view kZmqTopicTranscription = "transcription";

// ---------------------------------------------------------------------------
// Mel spectrogram parameters
// ---------------------------------------------------------------------------

/// Number of mel frequency bins. V3 uses 128 (V2 and earlier used 80).
inline constexpr uint32_t kMelBinCount = 128;

/// FFT analysis window: 25 ms @ 16 kHz = 400 samples.
inline constexpr uint32_t kFftWindowSizeSamples = 400;

/// Hop length between successive STFT frames: 10 ms @ 16 kHz = 160 samples.
inline constexpr uint32_t kHopLengthSamples = 160;

/// Mel filterbank lower frequency bound (Hz).
inline constexpr float kMelFrequencyMinHz = 0.0f;

/// Mel filterbank upper frequency bound (Hz).
inline constexpr float kMelFrequencyMaxHz = 8000.0f;

// ---------------------------------------------------------------------------
// 30-second sliding window
// ---------------------------------------------------------------------------

/// Standard long-form chunk duration in seconds (30 s sliding window).
inline constexpr uint32_t kChunkDurationSeconds = 30;

/// Derived: total audio samples per chunk (30 s × 16 000 Hz = 480 000).
/// NOTE: uses the literal 16000 to avoid a cross-include dependency on
/// audio_constants.hpp.  Must stay in sync with kSttInputSampleRateHz.
inline constexpr uint32_t kChunkSampleCount = kChunkDurationSeconds * 16'000;  // 480 000

/// Derived: number of STFT time frames per 30 s chunk.
/// 480 000 / 160 = 3 000.
inline constexpr uint32_t kFramesPerChunk =
    kChunkSampleCount / kHopLengthSamples;  // 3 000

/// Encoder output frames after two Conv1D stride-2 layers: 3000 / 2 = 1500.
inline constexpr uint32_t kEncoderOutputFrameCount = 1500;

/// Maximum decoder output tokens (BPE subwords) per 30 s chunk.
inline constexpr uint32_t kMaxDecoderTokens = 448;

/// Overlap between successive 30 s windows in seconds.
/// Prevents word loss at chunk boundaries.
inline constexpr uint32_t kChunkOverlapSeconds = 1;

// ---------------------------------------------------------------------------
// Special token IDs (multilingual model V3, unchanged across versions)
// ---------------------------------------------------------------------------

inline constexpr int kTokenStartOfTranscript = 50258;  ///< <|startoftranscript|>
inline constexpr int kTokenEndOfText         = 50257;  ///< <|endoftext|>
inline constexpr int kTokenTranscribe        = 50359;  ///< <|transcribe|>
inline constexpr int kTokenNoTimestamps      = 50363;  ///< <|notimestamps|>
inline constexpr int kTokenEnglish           = 50259;  ///< <|en|>
inline constexpr int kVocabularySize         = 51866;  ///< multilingual vocab size

// ---------------------------------------------------------------------------
// SHM ring-buffer layout (must match ShmAudioHeader in shm_mapping.hpp)
//
// These constants define the shared-memory segment layout used by
// AudioIngestNode (producer) and STTNode (consumer).
// We use literal values for sizeof(ShmAudioHeader) and  constants
// to avoid a cross-module include dependency on zmq_transport headers.
// ---------------------------------------------------------------------------

/// Total SHM segment size (must match the producer's audioShmSize()).
inline constexpr std::size_t kAudioShmSegmentByteSize =
    kAudioShmDataOffset + kRingBufferSlotCount * kAudioShmSlotByteSize;

// ---------------------------------------------------------------------------
// BPE byte-pair encoding markers (UTF-8 multi-byte prefix sequences)
// ---------------------------------------------------------------------------

/// First byte of a two-byte BPE token that encodes a special character.
inline constexpr uint8_t kBpePrefixByte  = 0xC4;

/// Second byte indicating a BPE-encoded space character (U+0120 → 0xC4 0xA0).
inline constexpr uint8_t kBpeSpaceByte   = 0xA0;

/// Second byte indicating a BPE-encoded newline character (U+010A → 0xC4 0x8A).
inline constexpr uint8_t kBpeNewlineByte = 0x8A;

// ---------------------------------------------------------------------------
// Subprocess configuration
// ---------------------------------------------------------------------------

/// Timeout in seconds for the Python whisper_transcribe.py subprocess.
/// Set high (120 s) to accommodate first-run TRT-LLM model loading.
inline constexpr int kSubprocessTimeoutSeconds = 120;

/// Estimated VRAM usage for the Whisper encoder+decoder TRT engines (~2.4 GB).
inline constexpr std::size_t kEstimatedVramBytes = 2400ULL * 1024 * 1024;

/// Maximum decoder output tokens per subprocess invocation.
inline constexpr int kSubprocessMaxTokens = 96;

