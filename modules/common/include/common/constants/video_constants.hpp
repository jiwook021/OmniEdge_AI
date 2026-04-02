#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Video Pipeline Compile-Time Constants
//
// Defines the video resolution limits, pixel format, and shared-memory
// circular-buffer layout used by VideoIngestNode (producer),
// BackgroundBlurNode, FaceRecognitionNode, and WebSocketBridgeNode
// (consumers).
//
// These constants are compile-time invariants. For runtime configuration,
// see common/runtime_defaults.hpp and omniedge_config.yaml.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Resolution and pixel format
// ---------------------------------------------------------------------------

/// Maximum supported input width in pixels (Full HD).
inline constexpr uint32_t kMaxInputWidth = 1920;

/// Maximum supported input height in pixels (Full HD).
inline constexpr uint32_t kMaxInputHeight = 1080;

/// Bytes per pixel for BGR24 — the only pixel format in OmniEdge_AI.
inline constexpr uint32_t kBgr24BytesPerPixel = 3;

/// Maximum BGR24 frame size in bytes: 1920 × 1080 × 3 = 6,220,800.
inline constexpr std::size_t kMaxBgr24FrameBytes =
    static_cast<std::size_t>(kMaxInputWidth) * kMaxInputHeight * kBgr24BytesPerPixel;

// ---------------------------------------------------------------------------
// Shared-memory circular-buffer layout (/oe.vid.ingest)
// ---------------------------------------------------------------------------

/// Number of circular buffer slots for the video SHM segment.
/// 4 slots at 30 fps = 133 ms of buffering before stale reads.
inline constexpr std::size_t kCircularBufferSlotCount = 4;

