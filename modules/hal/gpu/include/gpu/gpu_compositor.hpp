#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Compositor for Background Blur
//
// Provides two CUDA operations that complete the blur pipeline after
// YoloSegEngine delivers the 104×104 person mask:
//
//   1. upscaleMask()  — bilinear upscale from 104×104 → 1920×1080 float mask
//   2. composite()    — per-pixel: output = mask>0.5 ? original : blurred
//
// Both kernels run on the caller-supplied CUDA stream.
// cv::cuda::GaussianBlur (OpenCV CUDA) handles the Gaussian blur step;
// that call is made from BackgroundBlurNode to keep this header GPU-only.
//
// Memory ownership: all device pointers are caller-owned.
// ---------------------------------------------------------------------------


/**
 * @brief Bilinear upscale a float mask from (srcW×srcH) to (dstW×dstH).
 *
 * @param d_src     Device ptr to source mask [srcH × srcW], row-major float.
 * @param srcW      Source width (160).
 * @param srcH      Source height (160).
 * @param d_dst     Device ptr to destination mask [dstH × dstW].  Caller allocates.
 * @param dstW      Destination width  (1920).
 * @param dstH      Destination height (1080).
 * @param stream    CUDA stream.
 */
void upscaleMask(const float* d_src,
                 int           srcW, int srcH,
                 float*        d_dst,
                 int           dstW, int dstH,
                 cudaStream_t  stream);

/**
 * @brief Composite foreground (sharp) over background (blurred) using a mask.
 *
 * For each pixel at (x, y):
 *   output(x,y) = mask(x,y) > 0.5 ? original(x,y) : blurred(x,y)
 *
 * All GpuMat arguments must be CV_8UC3 (BGR24), same size (1920×1080).
 * The mask must be CV_32FC1, same size.
 *
 * @param original  Source BGR frame (sharp foreground).
 * @param blurred   Gaussian-blurred version of original.
 * @param mask      Float mask in [0,1] — 1.0 = foreground, 0.0 = background.
 * @param output    Output BGR frame (caller-allocated, same size).
 * @param stream    CUDA stream.
 */
void composite(const cv::cuda::GpuMat& original,
               const cv::cuda::GpuMat& blurred,
               const cv::cuda::GpuMat& mask,
               cv::cuda::GpuMat&       output,
               cudaStream_t            stream);

// ---------------------------------------------------------------------------
// ISP Adjustment Kernels
// ---------------------------------------------------------------------------

/**
 * @brief Apply brightness, contrast, and saturation in a single GPU pass.
 *
 * Per-pixel: pixel = clamp(pixel * contrast + brightness), then
 * interpolate toward luma by saturation factor.
 * Operates in-place on the GpuMat (must be CV_8UC3).
 *
 * @param frame       BGR24 GpuMat, modified in-place.
 * @param brightness  Additive offset (−100 to +100).
 * @param contrast    Multiplicative scale (0.5 to 3.0).  1.0 = no change.
 * @param saturation  Saturation factor (0.0 = grayscale, 1.0 = no change, 2.0 = double).
 * @param stream      CUDA stream.
 */
void applyIspBcs(cv::cuda::GpuMat& frame,
                 float brightness, float contrast, float saturation,
                 cudaStream_t stream);

/**
 * @brief Apply sharpening via 3×3 unsharp mask on the GPU.
 *
 * Reads from @p src, writes to @p dst.  Both must be CV_8UC3, same size.
 * For each pixel: dst = src + amount * (src − avg3x3(src)).
 * Edge pixels (border row/col) are copied unchanged.
 *
 * @param src     Input BGR GpuMat (unmodified).
 * @param dst     Output BGR GpuMat (caller-allocated, same size).
 * @param amount  Sharpening strength (0 = no change, 10 = very sharp).
 * @param stream  CUDA stream.
 */
void applyIspSharpen(const cv::cuda::GpuMat& src,
                     cv::cuda::GpuMat&       dst,
                     float                   amount,
                     cudaStream_t            stream);

