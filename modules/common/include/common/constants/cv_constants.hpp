#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "zmq/jpeg_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — Computer Vision Module Compile-Time Constants
//
// CV-specific constants for BackgroundBlurNode: MediaPipe Selfie Seg input,
// Gaussian blur kernel, and JPEG encoding quality.
//
// JPEG SHM protocol constants shared with WebSocketBridgeNode are in
// common/include/common/jpeg_constants.hpp (jpeg_protocol).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ZMQ topic strings produced by CV modules
// ---------------------------------------------------------------------------

/// Topic for face identity results (FaceRecognitionNode → Daemon).
inline constexpr std::string_view kZmqTopicIdentity = "identity";

/// Default Gaussian blur kernel size for background blur (must be odd).
/// Overridden by omniedge_config.yaml → background_blur.blur_kernel_size.
inline constexpr int kDefaultGaussianBlurKernelSize = 51;

/// Default JPEG quality for BackgroundBlurNode blurred-frame output.
/// Overridden by omniedge_config.yaml → background_blur.jpeg_quality.
inline constexpr int kDefaultJpegQuality = 85;

/// Frame pacing interval for the /video WebSocket channel (milliseconds).
/// The WS Bridge reads /oe.cv.blur.jpeg at this fixed cadence (20 fps).
inline constexpr int kBlurFramePacingMs = 50;

/// Maximum allowed pipeline time (milliseconds) before the blur node falls
/// back to raw-frame passthrough. Matches the pacing interval so the
/// display never stalls waiting for GPU inference.
inline constexpr int kBlurDeadlineMs = 50;

/// Estimated VRAM usage for face detection + recognition engines.
/// Default: SCRFD-10G + AuraFace v1 / glintr100 (~350 MiB).
inline constexpr std::size_t kFaceRecogEstimatedVramBytes = 350ULL * 1024 * 1024;

// ---------------------------------------------------------------------------
// FaceFilter AR constants
// ---------------------------------------------------------------------------

/// FaceMesh V2 ONNX input resolution (square, normalised).
inline constexpr uint32_t kFaceMeshInputResolution = 192;

/// Number of 3D landmarks output by FaceMesh V2.
inline constexpr uint32_t kFaceMeshLandmarkCount = 478;

/// Coordinate dimensions per FaceMesh landmark (x, y, z).
inline constexpr uint32_t kFaceMeshCoordinateDims = 3;

/// Dimension of the facial transformation matrix (4x4 homogeneous).
inline constexpr uint32_t kFacialTransformMatrixDim = 4;

/// Number of triangles in the MediaPipe canonical face mesh triangulation.
/// Used for per-triangle affine warp of filter textures.
inline constexpr uint32_t kFaceMeshTriangleCount = 468;

/// EMA smoothing factor for landmark temporal filtering (0 = no smoothing, 1 = full).
/// Higher values produce smoother landmarks but increase latency.
inline constexpr float kLandmarkEmaSmoothingAlpha = 0.6f;

/// JPEG quality for face-filtered frame output.
inline constexpr int kFaceFilterJpegQuality = 85;

/// Frame pacing interval for the /facefilter WebSocket channel (milliseconds).
inline constexpr int kFaceFilterFramePacingMs = 50;

/// Maximum pipeline time before falling back to passthrough (milliseconds).
inline constexpr int kFaceFilterDeadlineMs = 50;

/// ZMQ topic string for filtered frame notifications.
inline constexpr std::string_view kZmqTopicFilteredFrame = "filtered_frame";

/// Estimated VRAM usage for FaceMesh + texture buffers (~100 MiB).
inline constexpr std::size_t kFaceFilterEstimatedVramBytes = 100ULL * 1024 * 1024;

// ---------------------------------------------------------------------------
// SAM2 (Segment Anything Model 2) constants
// ---------------------------------------------------------------------------

/// Background compositing mode for SAM2 segmentation output.
/// Controls how the background (non-mask) region is rendered.
enum class Sam2BackgroundMode : uint8_t {
    kOverlay = 0,   ///< Blue tint overlay on mask region (debug/preview)
    kBlur    = 1,   ///< Gaussian blur background, sharp foreground (virtual meeting)
    kWhite   = 2,   ///< Solid white background
    kColor   = 3,   ///< User-specified solid color background
    kNone    = 4,   ///< Black background (person cutout)
    kImage   = 5,   ///< User-uploaded background image (JPG/PNG)
};

/// Parse a background mode string to enum (case-insensitive not needed — exact match).
[[nodiscard]] inline constexpr Sam2BackgroundMode parseSam2BgMode(std::string_view s) noexcept {
    if (s == "blur")    return Sam2BackgroundMode::kBlur;
    if (s == "white")   return Sam2BackgroundMode::kWhite;
    if (s == "color")   return Sam2BackgroundMode::kColor;
    if (s == "none")    return Sam2BackgroundMode::kNone;
    if (s == "image")   return Sam2BackgroundMode::kImage;
    return Sam2BackgroundMode::kOverlay;  // default fallback
}

/// Convert background mode enum to string.
[[nodiscard]] inline constexpr std::string_view sam2BgModeName(Sam2BackgroundMode m) noexcept {
    switch (m) {
    case Sam2BackgroundMode::kOverlay: return "overlay";
    case Sam2BackgroundMode::kBlur:    return "blur";
    case Sam2BackgroundMode::kWhite:   return "white";
    case Sam2BackgroundMode::kColor:   return "color";
    case Sam2BackgroundMode::kNone:    return "none";
    case Sam2BackgroundMode::kImage:   return "image";
    }
    return "overlay";
}

/// Default background color for kColor mode (RGB, white).
inline constexpr uint8_t kSam2DefaultBgColorR = 255;
inline constexpr uint8_t kSam2DefaultBgColorG = 255;
inline constexpr uint8_t kSam2DefaultBgColorB = 255;

/// Gaussian blur kernel size for SAM2 blur-background mode (must be odd).
inline constexpr int kSam2BlurKernelSize = 51;

/// Gaussian blur sigma for SAM2 blur-background mode.
inline constexpr float kSam2BlurSigma = 25.0f;

/// SAM2 image encoder input resolution (square, resized before encoding).
inline constexpr uint32_t kSam2EncoderInputResolution = 1024;

/// SAM2 mask decoder output resolution before upscaling to original size.
inline constexpr uint32_t kSam2MaskDecoderResolution = 256;

/// Logit threshold for converting SAM2 mask logits to binary mask.
/// Pixels with logit > threshold are foreground (255), else background (0).
inline constexpr float kSam2MaskThreshold = 0.0f;

/// Minimum IoU score to accept a SAM2 segmentation result as valid.
inline constexpr float kSam2MinIouScore = 0.50f;

/// Minimum stability score to accept a SAM2 mask.
inline constexpr float kSam2MinStabilityScore = 0.80f;

/// JPEG quality for SAM2 overlay output (mask composited onto original image).
inline constexpr int kSam2JpegQuality = 90;

/// ZMQ topic string for segmentation mask results.
inline constexpr std::string_view kZmqTopicSegmentationMask = "segmentation_mask";

/// ZMQ topic string for SAM2 prompt requests (from daemon/UI).
inline constexpr std::string_view kZmqTopicSam2Prompt = "sam2_prompt";

/// Estimated VRAM usage for SAM2 Tiny encoder + decoder (~800 MiB).
inline constexpr std::size_t kSam2EstimatedVramBytes = 800ULL * 1024 * 1024;

/// Maximum single-shot pipeline time before timeout (ms).
inline constexpr int kSam2InferenceTimeoutMs = 5000;

/// Frame pacing for SAM2 continuous tracking mode (milliseconds).
/// At 33 ms the WS bridge and GPU pipeline sustain 30 fps output.
inline constexpr int kSam2FramePacingMs = 33;

/// Maximum allowed pipeline time per frame in continuous tracking mode (ms).
/// If exceeded, the frame is skipped to prevent pipeline stalls.
inline constexpr int kSam2ContinuousDeadlineMs = 33;

/// Maximum size of a user-uploaded background image (bytes).
/// Reuses the general upload limit (10 MiB).
inline constexpr std::size_t kSam2MaxBgImageBytes = 10ULL * 1024 * 1024;

/// Overlay alpha for mask visualisation (0.0 = transparent, 1.0 = opaque).
inline constexpr float kSam2MaskOverlayAlpha = 0.45f;

/// Overlay colour for positive mask region (R, G, B).
inline constexpr uint8_t kSam2MaskOverlayR = 30;
inline constexpr uint8_t kSam2MaskOverlayG = 144;
inline constexpr uint8_t kSam2MaskOverlayB = 255;

