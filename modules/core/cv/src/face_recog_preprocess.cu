// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Face Recognition Preprocessing Kernels
//
// Fused CUDA kernels that replace CPU-side OpenCV preprocessing for the
// face recognition pipeline.  All operations run on the caller-supplied
// CUDA stream, writing directly into pre-allocated GPU buffers (suitable
// for ONNX IO Binding).
//
// Kernels:
//   letterboxResizeNormKernel — bilinear resize + grey pad + BGR→RGB + normalize → NCHW float
//   alignCropNormKernel       — affine warp + BGR→RGB + normalize → NCHW float
// ---------------------------------------------------------------------------

#include "cv/face_recog_preprocess.hpp"

#include <cuda_runtime.h>

#include "gpu/oe_cuda_check.hpp"


namespace {

/// CUDA thread block dimension for 2D image kernels.
constexpr int kBlockDim = 16;

/// Standard grey letterbox padding pixel value (mid-range of [0, 255]).
constexpr float kGreyPadPixel = 128.0f;

/// Pixel value range maximum (8-bit unsigned).
constexpr float kPixelRangeMax = 255.0f;


// ---------------------------------------------------------------------------
// letterboxResizeNormKernel
//
// Input:  BGR24 device image (srcW × srcH, 3 bytes/pixel, row-major)
// Output: NCHW float32 [1, 3, dstSize, dstSize] — RGB, normalized
//
// The output is letterboxed with grey (0.0 after normalization of 128)
// padding.  The resize uses bilinear interpolation.
//
// Normalization: (pixel - mean) / std   (SCRFD: mean=127.5, std=128.0)
// Channel order: BGR→RGB (swap channels 0↔2)
// ---------------------------------------------------------------------------

__global__ void letterboxResizeNormKernel(
    const uint8_t* __restrict__ src,
    int                         srcW,
    int                         srcH,
    float* __restrict__         dst,
    int                         dstSize,
    float                       scale,
    float                       padX,
    float                       padY,
    float                       mean,
    float                       invStd,
    float                       padValue)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstSize || y >= dstSize) return;

    const int planeStride = dstSize * dstSize;
    const int pixelIdx = y * dstSize + x;

    // Map output pixel to input pixel in the letterbox region
    const float srcX = (static_cast<float>(x) - padX) / scale;
    const float srcY = (static_cast<float>(y) - padY) / scale;

    // Check if this pixel falls within the resized image (not padding)
    const float newW = static_cast<float>(srcW) * scale;
    const float newH = static_cast<float>(srcH) * scale;

    if (static_cast<float>(x) < padX || static_cast<float>(x) >= padX + newW ||
        static_cast<float>(y) < padY || static_cast<float>(y) >= padY + newH) {
        // Padding pixel — normalized grey value
        dst[0 * planeStride + pixelIdx] = padValue;  // R
        dst[1 * planeStride + pixelIdx] = padValue;  // G
        dst[2 * planeStride + pixelIdx] = padValue;  // B
        return;
    }

    // Bilinear interpolation from source
    const int x0 = max(static_cast<int>(floorf(srcX)), 0);
    const int y0 = max(static_cast<int>(floorf(srcY)), 0);
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);

    const float wx = srcX - floorf(srcX);
    const float wy = srcY - floorf(srcY);

    // BGR source layout: row-major, 3 bytes per pixel
    // Output is RGB (NCHW): channel 0=R, 1=G, 2=B
    // Source BGR: channel 0=B, 1=G, 2=R → swap 0↔2 for RGB
    const int rgbMap[3] = {2, 1, 0};  // dst[c] reads src[rgbMap[c]]

    for (int c = 0; c < 3; ++c) {
        const int sc = rgbMap[c];
        const float tl = static_cast<float>(src[y0 * srcW * 3 + x0 * 3 + sc]);
        const float tr = static_cast<float>(src[y0 * srcW * 3 + x1 * 3 + sc]);
        const float bl = static_cast<float>(src[y1 * srcW * 3 + x0 * 3 + sc]);
        const float br = static_cast<float>(src[y1 * srcW * 3 + x1 * 3 + sc]);

        const float interpolated =
            tl * (1.0f - wx) * (1.0f - wy) +
            tr * wx * (1.0f - wy) +
            bl * (1.0f - wx) * wy +
            br * wx * wy;

        dst[c * planeStride + pixelIdx] = (interpolated - mean) * invStd;
    }
}


// ---------------------------------------------------------------------------
// alignCropNormKernel
//
// Input:  BGR24 device image (srcW × srcH)
// Output: NCHW float32 [1, 3, dstSize, dstSize] — RGB, normalized
//
// Applies an inverse affine warp (2x3 matrix) to sample from the source,
// converting BGR→RGB and normalizing in a single pass.
//
// The affine matrix is the INVERSE transform: for each output pixel (x,y),
// compute the source coordinate and bilinear-sample.
//
// AuraFace / ArcFace normalization: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
// (algebraically identical to the (pixel - 127.5)/127.5 form used by glintr100)
// ---------------------------------------------------------------------------

__global__ void alignCropNormKernel(
    const uint8_t* __restrict__ src,
    int                         srcW,
    int                         srcH,
    float* __restrict__         dst,
    int                         dstSize,
    const float* __restrict__   invAffine,  // [2][3] row-major
    float                       normScale,  // 1.0 / (255.0 * std)
    float                       normBias)   // -mean / std
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstSize || y >= dstSize) return;

    const int planeStride = dstSize * dstSize;
    const int pixelIdx = y * dstSize + x;

    // Inverse affine: map output (x,y) → source coordinates
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    const float srcX = invAffine[0] * fx + invAffine[1] * fy + invAffine[2];
    const float srcY = invAffine[3] * fx + invAffine[4] * fy + invAffine[5];

    // Border handling: fill with zero (black) if out of bounds
    if (srcX < 0.0f || srcX >= static_cast<float>(srcW - 1) ||
        srcY < 0.0f || srcY >= static_cast<float>(srcH - 1)) {
        dst[0 * planeStride + pixelIdx] = normBias;  // R = (0 - mean) / std
        dst[1 * planeStride + pixelIdx] = normBias;  // G
        dst[2 * planeStride + pixelIdx] = normBias;  // B
        return;
    }

    // Bilinear interpolation
    const int x0 = static_cast<int>(floorf(srcX));
    const int y0 = static_cast<int>(floorf(srcY));
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);

    const float wx = srcX - static_cast<float>(x0);
    const float wy = srcY - static_cast<float>(y0);

    // BGR→RGB: swap channels 0↔2
    const int rgbMap[3] = {2, 1, 0};

    for (int c = 0; c < 3; ++c) {
        const int sc = rgbMap[c];
        const float tl = static_cast<float>(src[y0 * srcW * 3 + x0 * 3 + sc]);
        const float tr = static_cast<float>(src[y0 * srcW * 3 + x1 * 3 + sc]);
        const float bl = static_cast<float>(src[y1 * srcW * 3 + x0 * 3 + sc]);
        const float br = static_cast<float>(src[y1 * srcW * 3 + x1 * 3 + sc]);

        const float interpolated =
            tl * (1.0f - wx) * (1.0f - wy) +
            tr * wx * (1.0f - wy) +
            bl * (1.0f - wx) * wy +
            br * wx * wy;

        dst[c * planeStride + pixelIdx] = interpolated * normScale + normBias;
    }
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// Public API — host-side launcher functions
// ---------------------------------------------------------------------------

void oe::cv::launchLetterboxResizeNorm(
    const uint8_t* d_bgrSrc,
    int            srcW,
    int            srcH,
    float*         d_nchwDst,
    int            dstSize,
    float          scale,
    float          padX,
    float          padY,
    float          mean,
    float          std,
    cudaStream_t   stream)
{
    const float invStd = 1.0f / std;
    const float padValue = (kGreyPadPixel - mean) * invStd;  // normalized grey

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(
        (dstSize + block.x - 1) / block.x,
        (dstSize + block.y - 1) / block.y);

    letterboxResizeNormKernel<<<grid, block, 0, stream>>>(
        d_bgrSrc, srcW, srcH, d_nchwDst, dstSize,
        scale, padX, padY, mean, invStd, padValue);
    OE_CUDA_CHECK(cudaGetLastError());
}


void oe::cv::launchAlignCropNorm(
    const uint8_t* d_bgrSrc,
    int            srcW,
    int            srcH,
    float*         d_nchwDst,
    int            dstSize,
    const float*   d_invAffine,
    float          mean,
    float          std,
    cudaStream_t   stream)
{
    // AuraFace: (pixel/255 - mean) / std = pixel * (1/(255*std)) + (-mean/std)
    const float normScale = 1.0f / (kPixelRangeMax * std);
    const float normBias  = -mean / std;

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(
        (dstSize + block.x - 1) / block.x,
        (dstSize + block.y - 1) / block.y);

    alignCropNormKernel<<<grid, block, 0, stream>>>(
        d_bgrSrc, srcW, srcH, d_nchwDst, dstSize,
        d_invAffine, normScale, normBias);
    OE_CUDA_CHECK(cudaGetLastError());
}
