// ---------------------------------------------------------------------------
// OmniEdge_AI — BasicVSR++ GPU Preprocessing Kernels
//
// Fused CUDA kernels for BasicVSR++ input preparation:
//   resizeNormFP32Kernel — bilinear resize + BGR->RGB + normalize [0,1]
//   chwToHwcBgrKernel    — float32 NCHW RGB -> uint8 HWC BGR (for nvJPEG)
// ---------------------------------------------------------------------------

#include "cv/basicvsrpp_preprocess.hpp"

#include "gpu/oe_cuda_check.hpp"

#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Kernel: resizeNormFP32Kernel
//
// One thread per output pixel. Fuses bilinear resize + BGR->RGB + normalize.
// Writes to NCHW float32 at the frame's offset in the 5D tensor.
// ---------------------------------------------------------------------------
__global__ static void resizeNormFP32Kernel(
    const uint8_t* __restrict__ src,
    int                          srcW,
    int                          srcH,
    float* __restrict__          dst,
    int                          dstW,
    int                          dstH,
    int                          frameOffset)    // offset in floats to this frame's start
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) { return; }

    // Bilinear interpolation source coordinates
    const float sourceX = (static_cast<float>(x) + 0.5f) *
                           (static_cast<float>(srcW) / static_cast<float>(dstW)) - 0.5f;
    const float sourceY = (static_cast<float>(y) + 0.5f) *
                           (static_cast<float>(srcH) / static_cast<float>(dstH)) - 0.5f;

    const int floorX = max(static_cast<int>(floorf(sourceX)), 0);
    const int floorY = max(static_cast<int>(floorf(sourceY)), 0);
    const int ceilX  = min(floorX + 1, srcW - 1);
    const int ceilY  = min(floorY + 1, srcH - 1);

    const float weightX = sourceX - floorf(sourceX);
    const float weightY = sourceY - floorf(sourceY);

    // Sample four BGR24 neighbors
    const uint8_t* topLeft     = src + (floorY * srcW + floorX) * 3;
    const uint8_t* topRight    = src + (floorY * srcW + ceilX)  * 3;
    const uint8_t* bottomLeft  = src + (ceilY  * srcW + floorX) * 3;
    const uint8_t* bottomRight = src + (ceilY  * srcW + ceilX)  * 3;

    const int planeStride = dstH * dstW;
    float* frameDst = dst + frameOffset;

    // Channel loop: src BGR (0=B,1=G,2=R) -> dst RGB planes (0=R,1=G,2=B)
    const int bgrToRgb[3] = {2, 1, 0};
    for (int c = 0; c < 3; ++c) {
        const int sc = bgrToRgb[c];
        const float interpolated =
            (topLeft[sc]     * (1.0f - weightX) + topRight[sc]    * weightX) * (1.0f - weightY) +
            (bottomLeft[sc]  * (1.0f - weightX) + bottomRight[sc] * weightX) * weightY;
        frameDst[c * planeStride + y * dstW + x] = interpolated / 255.0f;
    }
}


// ---------------------------------------------------------------------------
// Kernel: chwToHwcBgrKernel
//
// Converts float32 NCHW RGB [0,1] to uint8 HWC BGR [0,255] for nvJPEG.
// ---------------------------------------------------------------------------
__global__ static void chwToHwcBgrKernel(
    const float* __restrict__ nchwSrc,
    uint8_t*     __restrict__ bgrDst,
    int                       width,
    int                       height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const int planeStride = height * width;
    const int pixelIdx    = y * width + x;

    // Read RGB planes (NCHW), write BGR interleaved (HWC)
    const float r = nchwSrc[0 * planeStride + pixelIdx];
    const float g = nchwSrc[1 * planeStride + pixelIdx];
    const float b = nchwSrc[2 * planeStride + pixelIdx];

    uint8_t* out = bgrDst + pixelIdx * 3;
    out[0] = static_cast<uint8_t>(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    out[1] = static_cast<uint8_t>(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    out[2] = static_cast<uint8_t>(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
}


// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

namespace oe::cv {

void launchResizeNormFP32(
    const uint8_t* d_bgrSrc, int srcW, int srcH,
    float*         d_nchwDst, int dstW, int dstH,
    int            frameIndex, int /*totalFrames*/,
    cudaStream_t   stream)
{
    // Frame offset into the [1, N, 3, H, W] tensor
    const int frameOffset = frameIndex * 3 * dstH * dstW;

    const dim3 block(16, 16);
    const dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y);

    resizeNormFP32Kernel<<<grid, block, 0, stream>>>(
        d_bgrSrc, srcW, srcH,
        d_nchwDst, dstW, dstH,
        frameOffset);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchChwToHwcBgr(
    const float* d_nchwSrc,
    uint8_t*     d_bgrDst,
    int          width, int height,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    chwToHwcBgrKernel<<<grid, block, 0, stream>>>(
        d_nchwSrc, d_bgrDst, width, height);
    OE_CUDA_CHECK(cudaGetLastError());
}

} // namespace oe::cv
