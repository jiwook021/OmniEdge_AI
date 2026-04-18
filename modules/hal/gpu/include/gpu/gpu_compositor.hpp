#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "gpu/cuda_graph.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Compositor for Background Blur
//
// Provides two CUDA operations that complete the blur pipeline after
// the selfie-segmentation inferencer delivers a low-resolution person mask:
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

/**
 * @brief Composite foreground over a solid-color background using a mask.
 *
 * For each pixel at (x, y):
 *   output(x,y) = mask(x,y) > 0.5 ? original(x,y) : solidColor
 *
 * @param original  Source BGR frame (sharp foreground).
 * @param mask      Float mask in [0,1] — 1.0 = foreground, 0.0 = background.
 * @param output    Output BGR frame (caller-allocated, same size).
 * @param bgR       Background red   component [0, 255].
 * @param bgG       Background green component [0, 255].
 * @param bgB       Background blue  component [0, 255].
 * @param stream    CUDA stream.
 */
void compositeSolidBg(const cv::cuda::GpuMat& original,
                      const cv::cuda::GpuMat& mask,
                      cv::cuda::GpuMat&       output,
                      uint8_t bgR, uint8_t bgG, uint8_t bgB,
                      cudaStream_t stream);

/**
 * @brief Composite foreground over a user-uploaded background image using a mask.
 *
 * For each pixel at (x, y):
 *   output(x,y) = mask(x,y) > 0.5 ? original(x,y) : bgImage(x,y)
 *
 * The background image must be pre-uploaded to a GpuMat of the same size
 * as the original frame (caller resizes if necessary).
 *
 * @param original  Source BGR frame (sharp foreground).
 * @param mask      Float mask in [0,1] — 1.0 = foreground, 0.0 = background.
 * @param bgImage   User-uploaded background image (BGR, same size as original).
 * @param output    Output BGR frame (caller-allocated, same size).
 * @param stream    CUDA stream.
 */
void compositeBgImage(const cv::cuda::GpuMat& original,
                      const cv::cuda::GpuMat& mask,
                      const cv::cuda::GpuMat& bgImage,
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

/**
 * @brief Separable 2-pass Gaussian blur on a BGR24 GpuMat.
 *
 * Writes dst = GaussianBlur(src, kernelRadius, sigma). Uses @p scratch as the
 * horizontal-pass intermediate (same size/type as src/dst). All three GpuMats
 * must be CV_8UC3 and the same size; dst and scratch may alias neither src
 * nor each other.
 *
 * @param src            Input BGR frame.
 * @param scratch        Intermediate BGR buffer (caller-allocated, same size/type).
 * @param dst            Output BGR frame (caller-allocated, same size/type).
 * @param kernelRadius   Half-width in pixels (kernel size = 2*radius+1). Clamped to 15.
 * @param sigma          Gaussian sigma in pixels. Must be > 0.
 * @param stream         CUDA stream.
 */
void launchGaussianBlur(const cv::cuda::GpuMat& src,
                        cv::cuda::GpuMat&       scratch,
                        cv::cuda::GpuMat&       dst,
                        int                     kernelRadius,
                        float                   sigma,
                        cudaStream_t            stream);

// ---------------------------------------------------------------------------
// Fused upscale + composite kernels
//
// Combine bilinear mask upscale and pixel compositing in a single kernel,
// eliminating the intermediate full-resolution mask buffer (~8 MB at 1080p).
// Each output pixel computes its interpolated mask value on-the-fly.
// ---------------------------------------------------------------------------

/**
 * @brief Fused upscale + blur composite: upscale mask and blend in one pass.
 *
 * Equivalent to upscaleMask() + composite(), but without the intermediate
 * full-resolution mask allocation.  Saves ~8 MB global memory traffic.
 *
 * @param original    Source BGR frame (sharp foreground).  CV_8UC3.
 * @param blurred     Gaussian-blurred version of original.  CV_8UC3.
 * @param d_maskSrc   Device ptr to low-res mask [maskH × maskW], float.
 * @param maskW       Low-res mask width  (e.g. 104 or 160).
 * @param maskH       Low-res mask height (e.g. 104 or 160).
 * @param output      Output BGR frame.  CV_8UC3, same size as original.
 * @param stream      CUDA stream.
 */
void upscaleAndComposite(const cv::cuda::GpuMat& original,
                         const cv::cuda::GpuMat& blurred,
                         const float*            d_maskSrc,
                         int                     maskW,
                         int                     maskH,
                         cv::cuda::GpuMat&       output,
                         cudaStream_t            stream);

/**
 * @brief Fused upscale + solid-background composite.
 *
 * Equivalent to upscaleMask() + compositeSolidBg(), fused into one kernel.
 */
void upscaleAndCompositeSolidBg(const cv::cuda::GpuMat& original,
                                const float*            d_maskSrc,
                                int                     maskW,
                                int                     maskH,
                                cv::cuda::GpuMat&       output,
                                uint8_t bgR, uint8_t bgG, uint8_t bgB,
                                cudaStream_t            stream);

/**
 * @brief Fused upscale + background-image composite.
 *
 * Equivalent to upscaleMask() + compositeBgImage(), fused into one kernel.
 */
void upscaleAndCompositeBgImage(const cv::cuda::GpuMat& original,
                                const float*            d_maskSrc,
                                int                     maskW,
                                int                     maskH,
                                const cv::cuda::GpuMat& bgImage,
                                cv::cuda::GpuMat&       output,
                                cudaStream_t            stream);

// ---------------------------------------------------------------------------
// CompositorGraph — CUDA graph capture/replay for the compositor pipeline
//
// Captures the sequence: fused upscale+composite → ISP BCS → ISP sharpen
// into a CUDA graph on the first frame, then replays it on subsequent frames.
// Eliminates 3 kernel launch dispatches per frame (~20-30 us saved).
//
// The graph is invalidated (and re-captured on next frame) when:
//   - ISP parameters change (kernel arguments are baked into the graph)
//   - The caller explicitly calls invalidate()
//
// Usage in the production blur inferencer:
//
//   CompositorGraph cgraph;
//
//   // In processFrame():
//   if (!cgraph.captured()) {
//       cgraph.beginCapture(stream);
//       upscaleAndComposite(original, blurred, d_mask, maskW, maskH, output, stream);
//       applyIspBcs(output, b, c, s, stream);
//       applyIspSharpen(output, sharpened, amount, stream);
//       cgraph.endCapture(stream);
//   } else {
//       cgraph.replay(stream);
//   }
// ---------------------------------------------------------------------------
class CompositorGraph {
public:
    CompositorGraph() = default;

    [[nodiscard]] bool captured() const noexcept { return graph_.captured(); }

    void beginCapture(cudaStream_t stream) { graph_.beginCapture(stream); }
    void endCapture(cudaStream_t stream)   { graph_.endCapture(stream); }
    void replay(cudaStream_t stream)       { graph_.replay(stream); }

    /// Invalidate the cached graph (e.g. when ISP parameters change).
    /// The next call to captured() will return false, triggering re-capture.
    void invalidate() noexcept { graph_.reset(); }

private:
    CudaGraphInstance graph_;
};

