#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Security Camera Mode Compile-Time Constants
//
// Detection pipeline: YOLOX-Nano ONNX → bounding-box NMS →
// annotated JPEG output + NVENC MP4 recording + JSON Lines event log.
//
// All values are compile-time fallback defaults.  Runtime values come from
// config/omniedge_config.yaml (security_camera section) and config/omniedge.ini.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ZMQ topic strings produced by SecurityCameraNode
// ---------------------------------------------------------------------------

/// Per-frame detection results (bounding boxes, classes, confidences).
inline constexpr std::string_view kZmqTopicSecurityDetection = "security_detection";

/// High-level detection events (cooldown-filtered, one per notable sighting).
inline constexpr std::string_view kZmqTopicSecurityEvent = "security_event";

/// Recording pipeline status (recording flag, current file, duration).
inline constexpr std::string_view kZmqTopicSecurityRecordingStatus = "security_recording_status";

/// VLM analysis response tokens (streamed from on-demand conversation model).
inline constexpr std::string_view kZmqTopicSecurityVlmAnalysis = "security_vlm_analysis";

// ---------------------------------------------------------------------------
// Detection pipeline tuning
// ---------------------------------------------------------------------------

/// YOLO inference interval (milliseconds).  5 fps is sufficient for security
/// monitoring and reduces GPU utilisation by 83% vs. 30 fps.
inline constexpr int kSecurityDetectionIntervalMs = 200;

/// Minimum confidence to accept a YOLO detection (0.0 – 1.0).
inline constexpr float kSecurityConfidenceThreshold = 0.5f;

/// Minimum interval between publishing duplicate events for the same class
/// (milliseconds).  Prevents notification flood.
inline constexpr int kSecurityEventCooldownMs = 5000;

/// YOLO classes of interest for security mode.
/// Index mapping: 0 = person (COCO), 24 = backpack, 28 = suitcase.
/// Package detection is approximated by suitcase/backpack classes.
inline constexpr int kCocoClassPerson    = 0;
inline constexpr int kCocoClassBackpack  = 24;
inline constexpr int kCocoClassSuitcase  = 28;

// ---------------------------------------------------------------------------
// Recording pipeline
// ---------------------------------------------------------------------------

/// Default MP4 segment duration (minutes).  splitmuxsink rotates files at
/// this interval for bounded file sizes and easier scrubbing.
inline constexpr int kSecuritySegmentDurationMin = 30;

/// Recording frame rate (fps).  Lower than camera capture (30) to reduce
/// file size during 24/7 operation.
inline constexpr int kSecurityRecordingFps = 20;

/// H.264 bitrate for NVENC recording (bits per second).
/// 5 Mbps at 1080p/20fps produces ~1.1 GB per 30-min segment.
inline constexpr int kSecurityRecordingBitrate = 5'000'000;

/// Maximum recording retention (days).  Recordings older than this are
/// automatically deleted to prevent disk overflow during 24/7 operation.
inline constexpr int kSecurityRecordingRetentionDays = 7;

// ---------------------------------------------------------------------------
// Annotated JPEG output
// ---------------------------------------------------------------------------

/// JPEG quality for annotated detection overlay frames (0–100).
inline constexpr int kSecurityJpegQuality = 85;

/// Frame pacing interval for the /security_video WebSocket channel (ms).
/// 200 ms matches the detection interval (5 fps).
inline constexpr int kSecurityFramePacingMs = 200;

/// Bounding box line thickness (pixels) for detection overlay.
inline constexpr int kSecurityBboxThickness = 3;

/// Corner bracket length as a fraction of the shorter bbox dimension.
inline constexpr float kSecurityCornerLengthFraction = 0.15f;

/// Minimum corner bracket length (pixels) to keep visible on small boxes.
inline constexpr int kSecurityCornerLengthMinPx = 12;

/// Confidence bar height (pixels) drawn below the bounding box.
inline constexpr int kSecurityConfidenceBarHeight = 4;

/// Bounding box colour for "person" class (BGR: green).
inline constexpr uint8_t kSecurityBboxPersonB = 0;
inline constexpr uint8_t kSecurityBboxPersonG = 255;
inline constexpr uint8_t kSecurityBboxPersonR = 0;

/// Bounding box colour for "package" class (BGR: blue).
inline constexpr uint8_t kSecurityBboxPackageB = 255;
inline constexpr uint8_t kSecurityBboxPackageG = 165;
inline constexpr uint8_t kSecurityBboxPackageR = 0;

/// Bounding box colour for "car" class (BGR: cyan).
inline constexpr uint8_t kSecurityBboxCarB = 255;
inline constexpr uint8_t kSecurityBboxCarG = 255;
inline constexpr uint8_t kSecurityBboxCarR = 0;

/// Bounding box colour for "truck" / "bus" class (BGR: orange).
inline constexpr uint8_t kSecurityBboxTruckB = 0;
inline constexpr uint8_t kSecurityBboxTruckG = 140;
inline constexpr uint8_t kSecurityBboxTruckR = 255;

/// Bounding box colour for "motorcycle" / "bicycle" class (BGR: magenta).
inline constexpr uint8_t kSecurityBboxBikeB = 255;
inline constexpr uint8_t kSecurityBboxBikeG = 0;
inline constexpr uint8_t kSecurityBboxBikeR = 255;

/// ROI polygon overlay colour (BGR: cyan).
inline constexpr uint8_t kSecurityRoiB = 200;
inline constexpr uint8_t kSecurityRoiG = 200;
inline constexpr uint8_t kSecurityRoiR = 0;

/// ROI polygon line thickness (pixels).
inline constexpr int kSecurityRoiThickness = 2;

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------

/// Maximum event log file size before rotation (megabytes).
inline constexpr int kSecurityEventLogMaxSizeMB = 100;

/// Event log file name (relative to OE_LOG_DIR).
inline constexpr std::string_view kSecurityEventLogFileName = "security_events.jsonl";

// ---------------------------------------------------------------------------
// SHM names
// ---------------------------------------------------------------------------

/// Annotated JPEG output shared memory segment.
inline constexpr std::string_view kSecurityShmOutput = "/oe.cv.security.jpeg";

/// Size of the JPEG output segment: [uint32 jpegSize][JPEG bytes].
/// 4 MiB comfortably holds annotated 1920x1080 JPEGs at quality 85.
inline constexpr std::size_t kSecurityJpegShmSize = 4ULL * 1024 * 1024;

// ---------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------

/// Estimated VRAM usage for security camera YOLOX-Nano ONNX (~150 MiB).
inline constexpr std::size_t kSecurityCameraEstimatedVramBytes = 150ULL * 1024 * 1024;
