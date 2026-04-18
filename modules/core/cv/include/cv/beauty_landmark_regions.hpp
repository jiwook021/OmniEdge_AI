#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — FaceMesh V2 Landmark Region Maps
//
// Constexpr arrays mapping FaceMesh V2 478-landmark indices to semantically
// meaningful face regions.  Used by beauty CUDA kernels:
//
//   kFaceOval      — convex hull boundary for skin mask
//   kLeftEye       — eye contour (enlargement warp)
//   kRightEye      — eye contour (enlargement warp)
//   kUnderEyeLeft  — dark circle treatment region
//   kUnderEyeRight — dark circle treatment region
//   kNoseBridge    — nose narrowing control points
//   kJawline       — jawline reshaping control points
//   kLips          — lip contour (excluded from skin filter)
//
// Reference: MediaPipe FaceMesh V2 478-point canonical face model.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>


namespace oe::beauty {

// ---------------------------------------------------------------------------
// Face oval — 36 landmarks forming the outermost face contour.
// Used to compute the convex hull bounding the skin mask region.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kFaceOval[] = {
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58,  132, 93,  234, 127,
    162, 21,  54,  103, 67,  109
};
inline constexpr std::size_t kFaceOvalCount = sizeof(kFaceOval) / sizeof(kFaceOval[0]);

// ---------------------------------------------------------------------------
// Left eye contour — 16 landmarks around the left eye.
// Used for eye enlargement warp and skin filter exclusion.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kLeftEye[] = {
    33,  7,   163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
};
inline constexpr std::size_t kLeftEyeCount = sizeof(kLeftEye) / sizeof(kLeftEye[0]);

// ---------------------------------------------------------------------------
// Right eye contour — 16 landmarks around the right eye.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kRightEye[] = {
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
};
inline constexpr std::size_t kRightEyeCount = sizeof(kRightEye) / sizeof(kRightEye[0]);

// ---------------------------------------------------------------------------
// Left iris center — single landmark for eye center computation.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kLeftIrisCenter = 468;

// ---------------------------------------------------------------------------
// Right iris center — single landmark for eye center computation.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kRightIrisCenter = 473;

// ---------------------------------------------------------------------------
// Under-eye left — 6 landmarks defining the dark circle region below left eye.
// Used for regional brightness adjustment (dark circle removal).
// ---------------------------------------------------------------------------
inline constexpr uint16_t kUnderEyeLeft[] = {
    111, 117, 118, 119, 120, 121
};
inline constexpr std::size_t kUnderEyeLeftCount = sizeof(kUnderEyeLeft) / sizeof(kUnderEyeLeft[0]);

// ---------------------------------------------------------------------------
// Under-eye right — 6 landmarks defining the dark circle region below right eye.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kUnderEyeRight[] = {
    340, 346, 347, 348, 349, 350
};
inline constexpr std::size_t kUnderEyeRightCount = sizeof(kUnderEyeRight) / sizeof(kUnderEyeRight[0]);

// ---------------------------------------------------------------------------
// Nose bridge — 9 landmarks along the nose centerline.
// Used for nose narrowing (TPS warp).
// ---------------------------------------------------------------------------
inline constexpr uint16_t kNoseBridge[] = {
    6,   197, 195, 5,   4,   1,   19,  94,  2
};
inline constexpr std::size_t kNoseBridgeCount = sizeof(kNoseBridge) / sizeof(kNoseBridge[0]);

// ---------------------------------------------------------------------------
// Jawline — 17 landmarks along the lower face contour.
// Used for face slimming and jaw reshaping (TPS warp).
// ---------------------------------------------------------------------------
inline constexpr uint16_t kJawline[] = {
    234, 93,  132, 58,  172, 136, 150, 149, 176,
    148, 152, 377, 400, 378, 379, 365, 397
};
inline constexpr std::size_t kJawlineCount = sizeof(kJawline) / sizeof(kJawline[0]);

// ---------------------------------------------------------------------------
// Lips (outer contour) — 20 landmarks around the outer lip boundary.
// Excluded from skin smoothing filter to preserve lip definition.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kLips[] = {
    61,  146, 91,  181, 84,  17,  314, 405, 321, 375,
    291, 409, 270, 269, 267, 0,   37,  39,  40,  185
};
inline constexpr std::size_t kLipsCount = sizeof(kLips) / sizeof(kLips[0]);

// ---------------------------------------------------------------------------
// Eyebrows (combined left+right) — excluded from skin filter.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kLeftEyebrow[] = {
    70,  63,  105, 66,  107, 55,  65,  52,  53,  46
};
inline constexpr std::size_t kLeftEyebrowCount = sizeof(kLeftEyebrow) / sizeof(kLeftEyebrow[0]);

inline constexpr uint16_t kRightEyebrow[] = {
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276
};
inline constexpr std::size_t kRightEyebrowCount = sizeof(kRightEyebrow) / sizeof(kRightEyebrow[0]);

// ---------------------------------------------------------------------------
// Face center reference point (nose tip) — landmark 1.
// Used as reference for radial displacements in face slimming.
// ---------------------------------------------------------------------------
inline constexpr uint16_t kFaceCenter = 1;

} // namespace oe::beauty
