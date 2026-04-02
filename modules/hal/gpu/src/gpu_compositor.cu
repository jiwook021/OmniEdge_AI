// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Compositor CUDA Kernels
//
// Implements:
//   upscaleMask()  — bilinear upscale float mask (srcW×srcH) → (dstW×dstH)
//   composite()    — per-pixel blend using mask threshold
// ---------------------------------------------------------------------------

#include "gpu/gpu_compositor.hpp"

#include <stdexcept>
#include <format>

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "gpu/oe_cuda_check.hpp"


// ---------------------------------------------------------------------------
// Kernel: bilinearUpscaleMaskKernel
// ---------------------------------------------------------------------------

__global__ static void bilinearUpscaleMaskKernel(
    const float* __restrict__ src,
    int                        srcW,
    int                        srcH,
    float* __restrict__        dst,
    int                        dstW,
    int                        dstH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) { return; }

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

    const float interpolated =
        src[floorY * srcW + floorX] * (1.0f - weightX) * (1.0f - weightY) +
        src[floorY * srcW + ceilX]  * weightX           * (1.0f - weightY) +
        src[ceilY  * srcW + floorX] * (1.0f - weightX) * weightY           +
        src[ceilY  * srcW + ceilX]  * weightX           * weightY;

    dst[y * dstW + x] = interpolated;
}

// ---------------------------------------------------------------------------
// Kernel: compositeMaskKernel
//
// output[y][x] = mask[y][x] > 0.5 ? original[y][x] : blurred[y][x]
// All images are BGR24 (CV_8UC3, row-major, no padding between rows).
// ---------------------------------------------------------------------------

__global__ static void compositeMaskKernel(
    const uchar3* __restrict__ original,
    const uchar3* __restrict__ blurred,
    const float*  __restrict__ mask,
    uchar3*       __restrict__ output,
    int                        width,
    int                        height,
    int                        origStep3,    // original row step in uchar3 units
    int                        blurStep3,
    int                        maskStepF,    // mask row step in float units
    int                        outStep3)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const float m = mask[y * maskStepF + x];
    output[y * outStep3 + x] =
        (m > 0.5f) ? original[y * origStep3 + x]
                   : blurred [y * blurStep3  + x];
}

// ---------------------------------------------------------------------------
// upscaleMask()
// ---------------------------------------------------------------------------

void upscaleMask(const float* d_src,
                 int           srcW, int srcH,
                 float*        d_dst,
                 int           dstW, int dstH,
                 cudaStream_t  stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y);

    bilinearUpscaleMaskKernel<<<grid, block, 0, stream>>>(
        d_src, srcW, srcH,
        d_dst, dstW, dstH);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// composite()
// ---------------------------------------------------------------------------

void composite(const cv::cuda::GpuMat& original,
               const cv::cuda::GpuMat& blurred,
               const cv::cuda::GpuMat& mask,
               cv::cuda::GpuMat&       output,
               cudaStream_t            stream)
{
    const int W = original.cols;
    const int H = original.rows;

    // Allocate output if needed
    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    // cv::cuda::GpuMat step is in bytes; divide by element size for typed ptr math
    compositeMaskKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uchar3*>(original.data),
        reinterpret_cast<const uchar3*>(blurred.data),
        reinterpret_cast<const float*>(mask.data),
        reinterpret_cast<uchar3*>(output.data),
        W, H,
        static_cast<int>(original.step / sizeof(uchar3)),
        static_cast<int>(blurred.step  / sizeof(uchar3)),
        static_cast<int>(mask.step     / sizeof(float)),
        static_cast<int>(output.step   / sizeof(uchar3)));
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: ispBcsKernel — brightness / contrast / saturation in one pass
// ---------------------------------------------------------------------------

__global__ static void ispBcsKernel(
    uchar3* __restrict__ img,
    int                   width,
    int                   height,
    int                   step3,    // row step in uchar3 units
    float                 brightness,
    float                 contrast,
    float                 saturation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    uchar3 px = img[y * step3 + x];
    float b = px.x, g = px.y, r = px.z;

    // Contrast + Brightness: out = pixel * contrast + brightness
    b = b * contrast + brightness;
    g = g * contrast + brightness;
    r = r * contrast + brightness;

    // Saturation: interpolate toward luma
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    b = luma + saturation * (b - luma);
    g = luma + saturation * (g - luma);
    r = luma + saturation * (r - luma);

    px.x = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 255.0f));
    px.y = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 255.0f));
    px.z = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 255.0f));
    img[y * step3 + x] = px;
}

// ---------------------------------------------------------------------------
// applyIspBcs()
// ---------------------------------------------------------------------------

void applyIspBcs(cv::cuda::GpuMat& frame,
                 float brightness, float contrast, float saturation,
                 cudaStream_t stream)
{
    const int W = frame.cols;
    const int H = frame.rows;

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    ispBcsKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uchar3*>(frame.data),
        W, H,
        static_cast<int>(frame.step / sizeof(uchar3)),
        brightness, contrast, saturation);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: ispSharpenKernel — 3×3 unsharp mask
// ---------------------------------------------------------------------------

__global__ static void ispSharpenKernel(
    const uchar3* __restrict__ src,
    uchar3*       __restrict__ dst,
    int                        width,
    int                        height,
    int                        srcStep3,
    int                        dstStep3,
    float                      amount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    const uchar3 center = src[y * srcStep3 + x];

    // Border pixels: copy unchanged
    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        dst[y * dstStep3 + x] = center;
        return;
    }

    // 3×3 box average (simple unsharp mask base)
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const uchar3 p = src[(y + dy) * srcStep3 + (x + dx)];
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
        }
    }
    const float inv9 = 1.0f / 9.0f;

    // Unsharp mask: result = center + amount * (center − avg)
    const float b = center.x + amount * (center.x - sum.x * inv9);
    const float g = center.y + amount * (center.y - sum.y * inv9);
    const float r = center.z + amount * (center.z - sum.z * inv9);

    uchar3 out;
    out.x = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 255.0f));
    out.y = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 255.0f));
    out.z = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 255.0f));
    dst[y * dstStep3 + x] = out;
}

// ---------------------------------------------------------------------------
// applyIspSharpen()
// ---------------------------------------------------------------------------

void applyIspSharpen(const cv::cuda::GpuMat& src,
                     cv::cuda::GpuMat&       dst,
                     float                   amount,
                     cudaStream_t            stream)
{
    const int W = src.cols;
    const int H = src.rows;

    if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) {
        dst.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    ispSharpenKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uchar3*>(src.data),
        reinterpret_cast<uchar3*>(dst.data),
        W, H,
        static_cast<int>(src.step / sizeof(uchar3)),
        static_cast<int>(dst.step / sizeof(uchar3)),
        amount);
    OE_CUDA_CHECK(cudaGetLastError());
}

