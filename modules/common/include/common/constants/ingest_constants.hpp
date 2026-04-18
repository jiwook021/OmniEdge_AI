#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Ingest module constants
//
// Shared-memory layout offsets and model-dimension constants used by
// VideoIngestNode, AudioIngestNode, and SileroVad.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "zmq/jpeg_constants.hpp"


// ---------------------------------------------------------------------------
// ZMQ topic strings produced by ingest modules
// ---------------------------------------------------------------------------

/// Topic for raw audio chunks (AudioIngestNode → STTNode, AudioDenoiseNode).
inline constexpr std::string_view kZmqTopicAudioChunk = "audio_chunk";

/// Topic for VAD speech/silence status (AudioIngestNode → STTNode, Daemon).
inline constexpr std::string_view kZmqTopicVadStatus  = "vad_status";

/// Topic for screen frame notifications (ScreenIngestNode → Daemon, WS Bridge).
inline constexpr std::string_view kZmqTopicScreenFrame  = "screen_frame";

/// Topic for screen capture health status (ScreenIngestNode → Daemon, Frontend).
inline constexpr std::string_view kZmqTopicScreenHealth = "screen_health";

/// Topic for one-shot snapshot responses (ScreenIngestNode/VideoIngestNode →
/// MCP server or any subscriber that issued a `{"action":"snapshot"}` command).
///// JSON payload: {source, width, height, jpeg_len, jpeg_b64}. The JPEG payload
/// is base64-encoded into the JSON body so the existing JSON-only ZmqPublisher
/// API can carry it without a multipart extension.
inline constexpr std::string_view kZmqTopicSnapshotResponse = "snapshot_response";

/// POSIX shared-memory name for the screen ingest circular buffer.
inline constexpr std::string_view kScreenShmName = "/oe.screen.ingest";

/// Offset of the audio control word within the SHM segment.
inline constexpr std::size_t kAudioShmControlOffset    = 0;

// ---------------------------------------------------------------------------
// TCP ports — Windows host → WSL2 ingest endpoints
// INI keys in [audio_ingest] / [screen_ingest] override these at runtime.
// ---------------------------------------------------------------------------

/// AudioIngestNode TCP listener — receives 16 kHz PCM from Windows host.
inline constexpr int kAudioIngestTcpPort  = 5001;

/// ScreenIngestNode TCP listener — receives DXGI-captured JPEG frames.
inline constexpr int kScreenIngestTcpPort = 5002;

// ---------------------------------------------------------------------------
// Silero VAD GRU state dimensions (unified state tensor)
//
//   state: [2, 1, 128]  =>  2 * 1 * 128  =  256 floats
// ---------------------------------------------------------------------------

/// Silero VAD unified GRU state tensor dimensions [2, 1, 128].
inline constexpr int64_t kSileroVadStateDim0 = 2;
inline constexpr int64_t kSileroVadStateDim1 = 1;
inline constexpr int64_t kSileroVadStateDim2 = 128;

/// Number of floats in the Silero VAD unified state tensor [2,1,128].
inline constexpr std::size_t kSileroVadStateSize = kSileroVadStateDim0
    * kSileroVadStateDim1 * kSileroVadStateDim2;   // 256

/// Silero VAD v5 ONNX model expects 512 data samples per chunk at 16 kHz.
inline constexpr std::size_t kSileroVadModelInputSamples = 512;

/// Context window prepended to each chunk (64 for 16 kHz, 32 for 8 kHz).
/// The model requires this context for accurate classification.
inline constexpr std::size_t kSileroVadContextSize16k = 64;

/// Total tensor size fed to the model: context + chunk samples.
inline constexpr std::size_t kSileroVadTotalInputSamples =
    kSileroVadContextSize16k + kSileroVadModelInputSamples;  // 576

