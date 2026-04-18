#pragma once

// beauty_constants.hpp -- Shared constants for the OmniEdge beauty module.
//
// Real-time face beautification: FaceMesh V2 landmark detection +
// skin smoothing (bilateral filter) + shape warping (TPS) + ISP lighting.

#include <cstddef>
#include <cstdint>
#include <string_view>


// ---------------------------------------------------------------------------
// ZMQ topic strings
// ---------------------------------------------------------------------------

/// ZMQ topic for outgoing beauty frames (JPEG in SHM).
inline constexpr std::string_view kZmqTopicBeautyFrame = "beauty_frame";

// ---------------------------------------------------------------------------
// Module identity
// ---------------------------------------------------------------------------

/// Module name used for logging and YAML config key matching.
inline constexpr std::string_view kBeautyModuleName = "beauty";

// ---------------------------------------------------------------------------
// SHM segment names
// ---------------------------------------------------------------------------

/// Input SHM: raw BGR24 frames from VideoIngestNode.
inline constexpr std::string_view kBeautyShmInput  = "/oe.vid.ingest";

/// Output SHM: beauty-processed JPEG frames.
inline constexpr std::string_view kBeautyShmOutput = "/oe.cv.beauty.jpeg";

// ---------------------------------------------------------------------------
// FaceMesh V2 model parameters
// ---------------------------------------------------------------------------

/// FaceMesh V2 input resolution (square).
inline constexpr uint32_t kBeautyFaceMeshInputSize = 192;

/// Number of FaceMesh V2 landmarks (478 3D points).
inline constexpr uint32_t kBeautyLandmarkCount = 478;

// ---------------------------------------------------------------------------
// Bilateral filter defaults
// ---------------------------------------------------------------------------

/// Spatial sigma for the bilateral filter (pixels).
/// Higher = smoother skin but more edge blur.
inline constexpr float kBeautyBilateralSigmaSpatial = 7.0f;

/// Color sigma for the bilateral filter (0-255 range).
/// Higher = less edge preservation.
inline constexpr float kBeautyBilateralSigmaColor = 30.0f;

/// Filter kernel radius in pixels (half-width of the window).
inline constexpr int kBeautyBilateralKernelRadius = 5;

// ---------------------------------------------------------------------------
// Output encoding
// ---------------------------------------------------------------------------

/// JPEG quality for the beauty output (0-100).
inline constexpr int kBeautyJpegQuality = 85;

// ---------------------------------------------------------------------------
// Frame pacing
// ---------------------------------------------------------------------------

/// Target frame interval in milliseconds (30 fps).
inline constexpr int kBeautyFramePacingMs = 33;

// ---------------------------------------------------------------------------
// YCrCb skin mask thresholds
// ---------------------------------------------------------------------------

/// Cr channel range for skin detection.
inline constexpr float kSkinCrMin = 133.0f;
inline constexpr float kSkinCrMax = 173.0f;

/// Cb channel range for skin detection.
inline constexpr float kSkinCbMin =  77.0f;
inline constexpr float kSkinCbMax = 127.0f;
