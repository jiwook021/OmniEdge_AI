#pragma once

#include <chrono>
#include <cstddef>

// ---------------------------------------------------------------------------
// OmniEdge_AI — JPEG Shared-Memory Protocol Constants
//
// Defines the JPEG double-buffer SHM layout shared between BackgroundBlurNode
// (producer, writes compressed JPEG frames to /oe.cv.blur.jpeg) and
// WebSocketBridgeNode (consumer, reads and relays to the browser).
//
// These constants form the IPC contract. Changing them requires rebuilding
// both the CV module and the WebSocket bridge.
// ---------------------------------------------------------------------------


/// Maximum JPEG bytes per double-buffer slot in /oe.cv.blur.jpeg.
/// At quality 85, a 1080p JPEG typically occupies 200–450 KB. 1 MB gives
/// ample headroom for pathological high-detail frames.
inline constexpr std::size_t kMaxJpegBytesPerSlot = 1u * 1024u * 1024u;

/// JPEG double-buffer SHM segment size (single source of truth).
/// Layout: [ShmJpegControl (64 B)] [slot 0 JPEG] [slot 1 JPEG]
/// = sizeof(ShmJpegControl) + 2 * kMaxJpegBytesPerSlot
inline constexpr std::size_t kJpegShmSegmentByteSize =
    64 + 2u * kMaxJpegBytesPerSlot;  // ShmJpegControl(64) + 2 * 1 MiB

/// If no blurred_frame arrives within this window, the bridge falls back to
/// raw video_frame relay.  200 ms ≈ 6 frames at 30 fps — generous headroom
/// for GPU inference jitter without flashing raw frames during normal operation.
inline constexpr std::chrono::milliseconds kBlurredFrameFallbackTimeout{200};

// ---------------------------------------------------------------------------
// Video SHM layout offsets (shared between ingest, CV, and denoise modules)
// ---------------------------------------------------------------------------

/// Byte offset of the ShmVideoControl region within the video ingest SHM.
/// = sizeof(ShmVideoHeader) = 64.
inline constexpr std::size_t kVideoShmControlOffset = 64;

/// Byte offset of the first BGR24 frame slot within the video ingest SHM.
/// = sizeof(ShmVideoHeader) + sizeof(ShmCircularControl) = 64 + 128 = 192.
inline constexpr std::size_t kVideoShmFirstSlotOffset = 192;

// ---------------------------------------------------------------------------
// Audio SHM layout offsets (shared between ingest, STT, and denoise modules)
// ---------------------------------------------------------------------------

/// Byte offset from the start of the audio SHM segment to the first data slot.
/// Layout: [ShmAudioHeader padded to 64 B] [ShmCircularControl 128 B] [slots...]
inline constexpr std::size_t kAudioShmDataOffset = 192;

