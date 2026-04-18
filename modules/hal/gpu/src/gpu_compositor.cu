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
    const unsigned char* __restrict__ original,
    const unsigned char* __restrict__ blurred,
    const float*         __restrict__ mask,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              blurStepBytes,
    int                              maskStepF,    // mask row step in float units
    int                              outStepBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const float m = mask[y * maskStepF + x];
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : *reinterpret_cast<const uchar3*>(blurred  + y * blurStepBytes + x * 3);
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
        original.data,
        blurred.data,
        reinterpret_cast<const float*>(mask.data),
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(blurred.step),
        static_cast<int>(mask.step     / sizeof(float)),
        static_cast<int>(output.step));
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: compositeSolidBgKernel
//
// output[y][x] = mask[y][x] > 0.5 ? original[y][x] : (bgB, bgG, bgR)
// ---------------------------------------------------------------------------

__global__ static void compositeSolidBgKernel(
    const unsigned char* __restrict__ original,
    const float*         __restrict__ mask,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              maskStepF,
    int                              outStepBytes,
    uchar3                           bgColor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const float m = mask[y * maskStepF + x];
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : bgColor;
}

// ---------------------------------------------------------------------------
// compositeSolidBg()
// ---------------------------------------------------------------------------

void compositeSolidBg(const cv::cuda::GpuMat& original,
                      const cv::cuda::GpuMat& mask,
                      cv::cuda::GpuMat&       output,
                      uint8_t bgR, uint8_t bgG, uint8_t bgB,
                      cudaStream_t stream)
{
    const int W = original.cols;
    const int H = original.rows;

    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    // BGR order: x=B, y=G, z=R
    const uchar3 bgColor = make_uchar3(bgB, bgG, bgR);

    compositeSolidBgKernel<<<grid, block, 0, stream>>>(
        original.data,
        reinterpret_cast<const float*>(mask.data),
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(mask.step     / sizeof(float)),
        static_cast<int>(output.step),
        bgColor);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: compositeBgImageKernel
//
// output[y][x] = mask[y][x] > 0.5 ? original[y][x] : bgImage[y][x]
// Same as compositeMaskKernel but uses a user-uploaded image as background.
// ---------------------------------------------------------------------------

__global__ static void compositeBgImageKernel(
    const unsigned char* __restrict__ original,
    const float*         __restrict__ mask,
    const unsigned char* __restrict__ bgImage,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              maskStepF,
    int                              bgStepBytes,
    int                              outStepBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const float m = mask[y * maskStepF + x];
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : *reinterpret_cast<const uchar3*>(bgImage  + y * bgStepBytes   + x * 3);
}

// ---------------------------------------------------------------------------
// compositeBgImage()
// ---------------------------------------------------------------------------

void compositeBgImage(const cv::cuda::GpuMat& original,
                      const cv::cuda::GpuMat& mask,
                      const cv::cuda::GpuMat& bgImage,
                      cv::cuda::GpuMat&       output,
                      cudaStream_t            stream)
{
    const int W = original.cols;
    const int H = original.rows;

    // Validate inputs: all matrices must match in size; types must be correct.
    if (bgImage.size() != original.size() || bgImage.type() != CV_8UC3) {
        return;  // caller must resize bgImage to match original
    }
    if (mask.size() != original.size() || mask.type() != CV_32FC1) {
        return;
    }

    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    compositeBgImageKernel<<<grid, block, 0, stream>>>(
        original.data,
        reinterpret_cast<const float*>(mask.data),
        bgImage.data,
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(mask.step     / sizeof(float)),
        static_cast<int>(bgImage.step),
        static_cast<int>(output.step));
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: ispBcsKernel — brightness / contrast / saturation in one pass
// ---------------------------------------------------------------------------

__global__ static void ispBcsKernel(
    unsigned char* __restrict__ img,
    int                   width,
    int                   height,
    int                   stepBytes,  // row stride in bytes
    float                 brightness,
    float                 contrast,
    float                 saturation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    uchar3 px = *reinterpret_cast<uchar3*>(img + y * stepBytes + x * 3);
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
    *reinterpret_cast<uchar3*>(img + y * stepBytes + x * 3) = px;
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
        frame.data,
        W, H,
        static_cast<int>(frame.step),
        brightness, contrast, saturation);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: ispSharpenKernel — 3×3 unsharp mask
// ---------------------------------------------------------------------------

__global__ static void ispSharpenKernel(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int                              width,
    int                              height,
    int                              srcStepBytes,
    int                              dstStepBytes,
    float                            amount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    const uchar3 center = *reinterpret_cast<const uchar3*>(src + y * srcStepBytes + x * 3);

    // Border pixels: copy unchanged
    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        *reinterpret_cast<uchar3*>(dst + y * dstStepBytes + x * 3) = center;
        return;
    }

    // 3x3 box average (simple unsharp mask base)
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const uchar3 p = *reinterpret_cast<const uchar3*>(
                src + (y + dy) * srcStepBytes + (x + dx) * 3);
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
        }
    }
    const float inv9 = 1.0f / 9.0f;

    // Unsharp mask: result = center + amount * (center - avg)
    const float b = center.x + amount * (center.x - sum.x * inv9);
    const float g = center.y + amount * (center.y - sum.y * inv9);
    const float r = center.z + amount * (center.z - sum.z * inv9);

    uchar3 out;
    out.x = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 255.0f));
    out.y = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 255.0f));
    out.z = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 255.0f));
    *reinterpret_cast<uchar3*>(dst + y * dstStepBytes + x * 3) = out;
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
        src.data,
        dst.data,
        W, H,
        static_cast<int>(src.step),
        static_cast<int>(dst.step),
        amount);
    OE_CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// Separable Gaussian Blur (BGR24, two-pass)
//
// Horizontal pass: src -> scratch
// Vertical pass  : scratch -> dst
// Kernel coefficients are precomputed on host, baked into the kernel launch
// via __constant__ memory (max radius 15 → 31 taps).
// ---------------------------------------------------------------------------

static constexpr int kMaxGaussianRadius = 15;

__constant__ float c_gaussianWeights[2 * kMaxGaussianRadius + 1];

__global__ static void gaussianBlurHorizontalKernel(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int                              width,
    int                              height,
    int                              srcStepBytes,
    int                              dstStepBytes,
    int                              radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll 1
    for (int k = -radius; k <= radius; ++k) {
        const int xi = min(max(x + k, 0), width - 1);
        const uchar3 p = *reinterpret_cast<const uchar3*>(
            src + y * srcStepBytes + xi * 3);
        const float w = c_gaussianWeights[k + radius];
        acc.x += w * p.x;
        acc.y += w * p.y;
        acc.z += w * p.z;
    }

    uchar3 out;
    out.x = static_cast<unsigned char>(fminf(fmaxf(acc.x, 0.0f), 255.0f));
    out.y = static_cast<unsigned char>(fminf(fmaxf(acc.y, 0.0f), 255.0f));
    out.z = static_cast<unsigned char>(fminf(fmaxf(acc.z, 0.0f), 255.0f));
    *reinterpret_cast<uchar3*>(dst + y * dstStepBytes + x * 3) = out;
}

__global__ static void gaussianBlurVerticalKernel(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int                              width,
    int                              height,
    int                              srcStepBytes,
    int                              dstStepBytes,
    int                              radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll 1
    for (int k = -radius; k <= radius; ++k) {
        const int yi = min(max(y + k, 0), height - 1);
        const uchar3 p = *reinterpret_cast<const uchar3*>(
            src + yi * srcStepBytes + x * 3);
        const float w = c_gaussianWeights[k + radius];
        acc.x += w * p.x;
        acc.y += w * p.y;
        acc.z += w * p.z;
    }

    uchar3 out;
    out.x = static_cast<unsigned char>(fminf(fmaxf(acc.x, 0.0f), 255.0f));
    out.y = static_cast<unsigned char>(fminf(fmaxf(acc.y, 0.0f), 255.0f));
    out.z = static_cast<unsigned char>(fminf(fmaxf(acc.z, 0.0f), 255.0f));
    *reinterpret_cast<uchar3*>(dst + y * dstStepBytes + x * 3) = out;
}

void launchGaussianBlur(const cv::cuda::GpuMat& src,
                        cv::cuda::GpuMat&       scratch,
                        cv::cuda::GpuMat&       dst,
                        int                     kernelRadius,
                        float                   sigma,
                        cudaStream_t            stream)
{
    if (src.type() != CV_8UC3) {
        throw std::invalid_argument("launchGaussianBlur: src must be CV_8UC3");
    }
    if (sigma <= 0.0f) {
        throw std::invalid_argument("launchGaussianBlur: sigma must be > 0");
    }

    const int W = src.cols;
    const int H = src.rows;
    const int radius = std::min(std::max(kernelRadius, 1), kMaxGaussianRadius);

    if (scratch.empty() || scratch.size() != src.size() || scratch.type() != CV_8UC3) {
        scratch.create(H, W, CV_8UC3);
    }
    if (dst.empty() || dst.size() != src.size() || dst.type() != CV_8UC3) {
        dst.create(H, W, CV_8UC3);
    }

    float h_weights[2 * kMaxGaussianRadius + 1];
    const float twoSigmaSq = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const float v = std::exp(-static_cast<float>(k * k) / twoSigmaSq);
        h_weights[k + radius] = v;
        sum += v;
    }
    for (int i = 0; i < 2 * radius + 1; ++i) {
        h_weights[i] /= sum;
    }
    OE_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        c_gaussianWeights, h_weights,
        sizeof(float) * (2 * radius + 1),
        0, cudaMemcpyHostToDevice, stream));

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    gaussianBlurHorizontalKernel<<<grid, block, 0, stream>>>(
        src.data, scratch.data, W, H,
        static_cast<int>(src.step), static_cast<int>(scratch.step),
        radius);
    OE_CUDA_CHECK(cudaGetLastError());

    gaussianBlurVerticalKernel<<<grid, block, 0, stream>>>(
        scratch.data, dst.data, W, H,
        static_cast<int>(scratch.step), static_cast<int>(dst.step),
        radius);
    OE_CUDA_CHECK(cudaGetLastError());
}


// ===========================================================================
// Fused upscale + composite kernels
//
// Each output pixel computes the bilinear-interpolated mask value on-the-fly
// from the low-resolution mask, then immediately uses it for compositing.
// This eliminates the intermediate full-resolution float mask buffer
// (~8 MB at 1080p) and the associated global memory round-trip.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper device function: bilinear mask sample at output pixel (x, y)
// ---------------------------------------------------------------------------
__device__ static __forceinline__ float sampleMaskBilinear(
    const float* __restrict__ maskSrc,
    int                       maskW,
    int                       maskH,
    int                       dstW,
    int                       dstH,
    int                       x,
    int                       y)
{
    const float sourceX = (static_cast<float>(x) + 0.5f) *
                           (static_cast<float>(maskW) / static_cast<float>(dstW)) - 0.5f;
    const float sourceY = (static_cast<float>(y) + 0.5f) *
                           (static_cast<float>(maskH) / static_cast<float>(dstH)) - 0.5f;

    const int floorX = max(static_cast<int>(floorf(sourceX)), 0);
    const int floorY = max(static_cast<int>(floorf(sourceY)), 0);
    const int ceilX  = min(floorX + 1, maskW - 1);
    const int ceilY  = min(floorY + 1, maskH - 1);

    const float weightX = sourceX - floorf(sourceX);
    const float weightY = sourceY - floorf(sourceY);

    return maskSrc[floorY * maskW + floorX] * (1.0f - weightX) * (1.0f - weightY) +
           maskSrc[floorY * maskW + ceilX]  * weightX           * (1.0f - weightY) +
           maskSrc[ceilY  * maskW + floorX] * (1.0f - weightX) * weightY           +
           maskSrc[ceilY  * maskW + ceilX]  * weightX           * weightY;
}

// ---------------------------------------------------------------------------
// Kernel: upscaleAndCompositeKernel (blur background variant)
// ---------------------------------------------------------------------------
__global__ static void upscaleAndCompositeKernel(
    const unsigned char* __restrict__ original,
    const unsigned char* __restrict__ blurred,
    const float*         __restrict__ maskSrc,
    int                              maskW,
    int                              maskH,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              blurStepBytes,
    int                              outStepBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    const float m = sampleMaskBilinear(maskSrc, maskW, maskH, width, height, x, y);
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : *reinterpret_cast<const uchar3*>(blurred  + y * blurStepBytes + x * 3);
}

// ---------------------------------------------------------------------------
// Kernel: upscaleAndCompositeSolidBgKernel
// ---------------------------------------------------------------------------
__global__ static void upscaleAndCompositeSolidBgKernel(
    const unsigned char* __restrict__ original,
    const float*         __restrict__ maskSrc,
    int                              maskW,
    int                              maskH,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              outStepBytes,
    uchar3                           bgColor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    const float m = sampleMaskBilinear(maskSrc, maskW, maskH, width, height, x, y);
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : bgColor;
}

// ---------------------------------------------------------------------------
// Kernel: upscaleAndCompositeBgImageKernel
// ---------------------------------------------------------------------------
__global__ static void upscaleAndCompositeBgImageKernel(
    const unsigned char* __restrict__ original,
    const float*         __restrict__ maskSrc,
    int                              maskW,
    int                              maskH,
    const unsigned char* __restrict__ bgImage,
    unsigned char*       __restrict__ output,
    int                              width,
    int                              height,
    int                              origStepBytes,
    int                              bgStepBytes,
    int                              outStepBytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) { return; }

    const float m = sampleMaskBilinear(maskSrc, maskW, maskH, width, height, x, y);
    *reinterpret_cast<uchar3*>(output + y * outStepBytes + x * 3) =
        (m > 0.5f) ? *reinterpret_cast<const uchar3*>(original + y * origStepBytes + x * 3)
                   : *reinterpret_cast<const uchar3*>(bgImage  + y * bgStepBytes   + x * 3);
}

// ---------------------------------------------------------------------------
// upscaleAndComposite()
// ---------------------------------------------------------------------------

void upscaleAndComposite(const cv::cuda::GpuMat& original,
                         const cv::cuda::GpuMat& blurred,
                         const float*            d_maskSrc,
                         int                     maskW,
                         int                     maskH,
                         cv::cuda::GpuMat&       output,
                         cudaStream_t            stream)
{
    const int W = original.cols;
    const int H = original.rows;

    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    upscaleAndCompositeKernel<<<grid, block, 0, stream>>>(
        original.data,
        blurred.data,
        d_maskSrc, maskW, maskH,
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(blurred.step),
        static_cast<int>(output.step));
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// upscaleAndCompositeSolidBg()
// ---------------------------------------------------------------------------

void upscaleAndCompositeSolidBg(const cv::cuda::GpuMat& original,
                                const float*            d_maskSrc,
                                int                     maskW,
                                int                     maskH,
                                cv::cuda::GpuMat&       output,
                                uint8_t bgR, uint8_t bgG, uint8_t bgB,
                                cudaStream_t            stream)
{
    const int W = original.cols;
    const int H = original.rows;

    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    const uchar3 bgColor = make_uchar3(bgB, bgG, bgR);

    upscaleAndCompositeSolidBgKernel<<<grid, block, 0, stream>>>(
        original.data,
        d_maskSrc, maskW, maskH,
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(output.step),
        bgColor);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// upscaleAndCompositeBgImage()
// ---------------------------------------------------------------------------

void upscaleAndCompositeBgImage(const cv::cuda::GpuMat& original,
                                const float*            d_maskSrc,
                                int                     maskW,
                                int                     maskH,
                                const cv::cuda::GpuMat& bgImage,
                                cv::cuda::GpuMat&       output,
                                cudaStream_t            stream)
{
    const int W = original.cols;
    const int H = original.rows;

    if (bgImage.size() != original.size() || bgImage.type() != CV_8UC3) {
        return;
    }

    if (output.empty() || output.size() != original.size() ||
        output.type() != original.type()) {
        output.create(H, W, CV_8UC3);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y);

    upscaleAndCompositeBgImageKernel<<<grid, block, 0, stream>>>(
        original.data,
        d_maskSrc, maskW, maskH,
        bgImage.data,
        output.data,
        W, H,
        static_cast<int>(original.step),
        static_cast<int>(bgImage.step),
        static_cast<int>(output.step));
    OE_CUDA_CHECK(cudaGetLastError());
}

