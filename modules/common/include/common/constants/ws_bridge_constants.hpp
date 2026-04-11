#pragma once

#include <cstddef>
#include <string_view>

#include "zmq/jpeg_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — WebSocket Bridge Compile-Time Constants
//
// ZMQ topic strings used by WebSocketBridgeNode for upstream subscriptions
// and downstream publishing, plus SHM segment sizes for JPEG, audio, and
// TTS shared-memory regions consumed by the bridge.
//
// Struct-size literals are derived from the POD headers defined in
// zmq_transport/shm_mapping.hpp and kept as literal values to avoid
// cross-module header dependencies in downstream consumers.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ZMQ topic strings
// ---------------------------------------------------------------------------

// NOTE: also defined in conversation_constants.hpp — keep in sync.
inline constexpr std::string_view kZmqTopicUiCommand      = "ui_command";
inline constexpr std::string_view kZmqTopicBlurredFrame    = "blurred_frame";
inline constexpr std::string_view kZmqTopicTtsAudio        = "tts_audio";
inline constexpr std::string_view kZmqTopicLlmResponse     = "llm_response";
inline constexpr std::string_view kZmqTopicModuleStatus    = "module_status";
inline constexpr std::string_view kZmqTopicFaceRegistered  = "face_registered";
inline constexpr std::string_view kZmqTopicVideoFrame      = "video_frame";
inline constexpr std::string_view kZmqTopicDenoisedFrame   = "denoised_frame";
inline constexpr std::string_view kZmqTopicDenoisedAudio   = "denoised_audio";

// ---------------------------------------------------------------------------
// SHM segment sizes
// ---------------------------------------------------------------------------

/// Maximum float32 samples in the denoised-audio SHM segment.
/// 2 seconds at 16 kHz = 32,000 samples.
inline constexpr std::size_t kDenoiseAudioShmMaxSampleCount = 32'000;

/// Denoised-audio SHM segment size.
/// Layout: [ShmAudioHeader (24 B)] [float32 PCM samples]
/// = sizeof(ShmAudioHeader) + kDenoiseAudioShmMaxSampleCount * sizeof(float)
inline constexpr std::size_t kShmAudioHeaderSize = 24;  // validated by static_assert in shm_mapping.hpp
inline constexpr std::size_t kDenoiseAudioShmSegmentByteSize =
    kShmAudioHeaderSize + kDenoiseAudioShmMaxSampleCount * sizeof(float);

/// Maximum float32 samples in the TTS SHM segment.
/// 10 seconds at 24 kHz = 240,000 samples.
inline constexpr std::size_t kTtsShmMaxSampleCount = 240'000;

/// TTS SHM segment size.
/// Layout: [ShmTtsHeader (24 B)] [float32 PCM samples]
inline constexpr std::size_t kShmTtsHeaderSize = 24;  // validated by static_assert in shm_tts_header.hpp
inline constexpr std::size_t kTtsShmSegmentByteSize =
    kShmTtsHeaderSize + kTtsShmMaxSampleCount * sizeof(float);

// ---------------------------------------------------------------------------
// Image upload protocol
// ---------------------------------------------------------------------------

/// Binary frame on /chat for image upload:
///   [4 bytes: JSON header length, little-endian uint32]
///   [N bytes: UTF-8 JSON with action metadata]
///   [remaining bytes: raw JPEG/PNG image data]
inline constexpr uint32_t kImageUploadHeaderSize = 4;

/// Maximum accepted image upload size (10 MiB).
inline constexpr std::size_t kMaxImageUploadBytes = 10u * 1024u * 1024u;

/// Temporary directory for uploaded images.
inline constexpr std::string_view kUploadTempDir = "/tmp";

