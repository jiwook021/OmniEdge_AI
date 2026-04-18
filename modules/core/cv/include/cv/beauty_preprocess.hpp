#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Beauty Pipeline CUDA Kernel Declarations
//
// All kernels operate on raw device pointers (no cv::cuda::GpuMat dependency).
// BGR24 frames are packed uint8 arrays: [B0 G0 R0 B1 G1 R1 ...], row-major.
//
// Kernel inventory:
//   launchResizeNormForFaceMesh  — fused bilinear resize + BGR→RGB + normalize
//   launchComputeSkinMask        — YCrCb skin detection within face bounding box
//   launchBilateralFilterMasked  — edge-preserving smooth within skin mask
//   launchRegionBrightnessAdjust — additive brightness within a region mask
//   launchBeautySharpen          — 3x3 unsharp mask on raw BGR pointers
//   launchBcsAdjust              — brightness/contrast/saturation in-place
//   launchWarmthAdjust           — color temperature shift within face mask
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>


namespace oe::beauty {

// ---------------------------------------------------------------------------
// Fused preprocessing: bilinear resize BGR→RGB + normalize [0,1] to FP32
//
// Input:  d_bgrSrc  — device BGR24 uint8 [srcH × srcW × 3]
// Output: d_rgbDst  — device FP32 NCHW [1, 3, dstH, dstW], range [0,1]
// ---------------------------------------------------------------------------
void launchResizeNormForFaceMesh(
    const uint8_t* d_bgrSrc, int srcW, int srcH,
    float*         d_rgbDst, int dstW, int dstH,
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Skin mask: BGR→YCrCb threshold within face bounding box
//
// For each pixel within the bounding box defined by (bboxX, bboxY, bboxW, bboxH),
// converts to YCrCb and checks Cr/Cb against skin-tone thresholds.
// Pixels outside the bounding box get mask = 0.
//
// Input:  d_bgr      — device BGR24 uint8 [h × w × 3]
// Output: d_mask     — device float [h × w], values in {0.0, 1.0}
//
// The bounding box is computed from the FaceMesh landmarks on the host side
// and passed in to avoid uploading all 478 landmarks to the kernel.
// ---------------------------------------------------------------------------
void launchComputeSkinMask(
    const uint8_t* d_bgr,
    float*         d_mask,
    int            w, int h,
    int            bboxX, int bboxY, int bboxW, int bboxH,
    float          crMin, float crMax,
    float          cbMin, float cbMax,
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Masked bilateral filter — edge-preserving skin smoothing
//
// Applies bilateral filter only where d_mask > 0.5. Pixels with mask <= 0.5
// are copied unchanged from input to output.
//
// The filter window is (2*radius+1) × (2*radius+1). Processing is restricted
// to the bounding box region for performance (~30% of full frame).
//
// Input:  d_input    — device BGR24 uint8 [h × w × 3]
//         d_mask     — device float [h × w]
// Output: d_output   — device BGR24 uint8 [h × w × 3]
// ---------------------------------------------------------------------------
void launchBilateralFilterMasked(
    const uint8_t* d_input,
    uint8_t*       d_output,
    const float*   d_mask,
    int            w, int h,
    int            bboxX, int bboxY, int bboxW, int bboxH,
    float          sigmaSpatial,
    float          sigmaColor,
    int            radius,
    float          strength,      // 0-100 slider value (0 = no smoothing)
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Region brightness adjustment — additive brightness within a mask region
//
// Brightens pixels where d_regionMask > 0.5. Used for:
//   - Dark circle removal (under-eye region masks)
//   - Tone evening within skin mask
//
// Operates in-place on d_bgr.
// ---------------------------------------------------------------------------
void launchRegionBrightnessAdjust(
    uint8_t*       d_bgr,
    const float*   d_regionMask,
    int            w, int h,
    float          amount,        // brightness offset (-100 to +100)
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Sharpening: 3x3 unsharp mask on raw BGR24 pointers
//
// Same algorithm as gpu_compositor's ispSharpenKernel but without GpuMat.
// Reads from d_src, writes to d_dst. Both must be [h × w × 3] uint8.
// ---------------------------------------------------------------------------
void launchBeautySharpen(
    const uint8_t* d_src,
    uint8_t*       d_dst,
    int            w, int h,
    float          amount,        // 0 = no change, 10 = very sharp
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// BCS: brightness + contrast + saturation in one pass (in-place)
//
// Same algorithm as gpu_compositor's ispBcsKernel but without GpuMat.
// ---------------------------------------------------------------------------
void launchBcsAdjust(
    uint8_t*     d_bgr,
    int          w, int h,
    float        brightness,      // additive (-100 to +100)
    float        contrast,        // multiplicative (0.5 to 3.0, 1.0 = no change)
    float        saturation,      // 0 = grayscale, 1 = no change, 2 = double
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Warmth: color temperature shift within face mask (in-place)
//
// Adds warmth (positive = warm/orange, negative = cool/blue) only where
// d_faceMask > 0.5. Background pixels are untouched.
// ---------------------------------------------------------------------------
void launchWarmthAdjust(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          warmth,        // -100 to +100
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Shadow fill: brighten low-luminance pixels within face mask (in-place)
//
// Selectively brightens shadow areas (luma < threshold) within the face.
// ---------------------------------------------------------------------------
void launchShadowFill(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          amount,        // 0-100 slider value
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Highlight boost: subtle specular boost on bright face pixels (in-place)
// ---------------------------------------------------------------------------
void launchHighlightBoost(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          amount,        // 0-100 slider value
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Build under-eye region mask from landmark coordinates.
//
// Given 2D landmark coordinates (already projected to image space), fills
// d_regionMask with 1.0 inside an elliptical region under each eye.
// The mask should be pre-zeroed by the caller.
//
// landmarkX/Y arrays contain the 6 under-eye landmark points for one eye.
// ---------------------------------------------------------------------------
void launchBuildUnderEyeMask(
    float*         d_regionMask,
    int            w, int h,
    const float*   landmarkX,     // host array, 6 points
    const float*   landmarkY,     // host array, 6 points
    int            numPoints,
    float          dilate,        // pixels to expand the region
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Thin-Plate Spline warp — face reshaping within bounding box
//
// For each output pixel in the face region, computes the TPS-warped source
// coordinate and samples with bilinear interpolation.  Pixels outside the
// bounding box are copied unchanged.
//
// The TPS mapping for a point (x,y) is:
//   f(x,y) = a_0 + a_x*x + a_y*y + sum_i(w_i * U(||(x,y) - p_i||))
// where U(r) = r^2 * log(r) is the TPS basis function.
//
// @param d_input     Input BGR24 [h × w × 3]
// @param d_output    Output BGR24 [h × w × 3]
// @param d_srcX      Source control point X coords [numPoints] (device)
// @param d_srcY      Source control point Y coords [numPoints] (device)
// @param d_weightsX  TPS weights for X: [numPoints + 3] (device)
// @param d_weightsY  TPS weights for Y: [numPoints + 3] (device)
// @param numPoints   Number of control points (e.g. 68)
// @param bboxX/Y/W/H Face bounding box (warp only applies within)
// ---------------------------------------------------------------------------
void launchTpsWarp(
    const uint8_t* d_input,
    uint8_t*       d_output,
    const float*   d_srcX,
    const float*   d_srcY,
    const float*   d_weightsX,
    const float*   d_weightsY,
    int            numPoints,
    int            w, int h,
    int            bboxX, int bboxY, int bboxW, int bboxH,
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Background compositing: solid colour behind person mask
//
// For each pixel: output = (mask > 0.5) ? foreground : (bgB, bgG, bgR)
// The mask is a float [h x w] person segmentation mask (1.0 = person).
// ---------------------------------------------------------------------------
void launchCompositeSolidBg(
    const uint8_t* d_foreground,
    uint8_t*       d_output,
    const float*   d_mask,
    int            w, int h,
    uint8_t        bgB, uint8_t bgG, uint8_t bgR,
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// Background compositing: replace background with user image
//
// For each pixel: output = (mask > 0.5) ? foreground : bgImage
// Both d_foreground and d_bgImage are BGR24 [h x w x 3].
// The caller must resize d_bgImage to match frame dimensions.
// ---------------------------------------------------------------------------
void launchCompositeBgImage(
    const uint8_t* d_foreground,
    uint8_t*       d_output,
    const float*   d_mask,
    const uint8_t* d_bgImage,
    int            w, int h,
    cudaStream_t   stream);

} // namespace oe::beauty
