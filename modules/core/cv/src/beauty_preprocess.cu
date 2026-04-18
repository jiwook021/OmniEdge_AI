// ---------------------------------------------------------------------------
// OmniEdge_AI — Beauty Pipeline CUDA Kernels
//
// All kernels use raw device pointers (no cv::cuda::GpuMat).
// BGR24 layout: [B0 G0 R0 B1 G1 R1 ...], row-major, no padding.
// ---------------------------------------------------------------------------

#include "cv/beauty_preprocess.hpp"

#include "gpu/oe_cuda_check.hpp"

#include <cuda_runtime.h>


// ===================================================================
// Kernel: resizeNormFaceMeshKernel
//
// Fused bilinear resize + BGR→RGB + normalize [0,1] to FP32 NCHW.
// One thread per output pixel.
// ===================================================================
__global__ static void resizeNormFaceMeshKernel(
    const uint8_t* __restrict__ src,
    int                          srcW,
    int                          srcH,
    float* __restrict__          dst,
    int                          dstW,
    int                          dstH)
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

    const uint8_t* topLeft     = src + (floorY * srcW + floorX) * 3;
    const uint8_t* topRight    = src + (floorY * srcW + ceilX)  * 3;
    const uint8_t* bottomLeft  = src + (ceilY  * srcW + floorX) * 3;
    const uint8_t* bottomRight = src + (ceilY  * srcW + ceilX)  * 3;

    const int planeStride = dstH * dstW;

    // BGR input -> RGB output planes (channel 0=R, 1=G, 2=B)
    const int bgrToRgb[3] = {2, 1, 0};
    for (int c = 0; c < 3; ++c) {
        const int sc = bgrToRgb[c];
        const float interpolated =
            (topLeft[sc]     * (1.0f - weightX) + topRight[sc]    * weightX) * (1.0f - weightY) +
            (bottomLeft[sc]  * (1.0f - weightX) + bottomRight[sc] * weightX) * weightY;
        dst[c * planeStride + y * dstW + x] = interpolated / 255.0f;
    }
}


// ===================================================================
// Kernel: skinMaskKernel
//
// BGR→YCrCb threshold within a bounding box. Outside the box, mask=0.
// ===================================================================
__global__ static void skinMaskKernel(
    const uint8_t* __restrict__ bgr,
    float* __restrict__         mask,
    int                          w,
    int                          h,
    int                          bboxX,
    int                          bboxY,
    int                          bboxW,
    int                          bboxH,
    float                        crMin,
    float                        crMax,
    float                        cbMin,
    float                        cbMax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }

    // Outside bounding box: mask = 0
    if (x < bboxX || x >= bboxX + bboxW || y < bboxY || y >= bboxY + bboxH) {
        mask[y * w + x] = 0.0f;
        return;
    }

    const int idx = (y * w + x) * 3;
    const float B = static_cast<float>(bgr[idx]);
    const float G = static_cast<float>(bgr[idx + 1]);
    const float R = static_cast<float>(bgr[idx + 2]);

    // RGB → YCrCb (ITU-R BT.601)
    const float Y  =  0.299f * R + 0.587f * G + 0.114f * B;
    const float Cr = (R - Y) * 0.713f + 128.0f;
    const float Cb = (B - Y) * 0.564f + 128.0f;

    const bool isSkin = (Cr >= crMin && Cr <= crMax &&
                         Cb >= cbMin && Cb <= cbMax &&
                         Y  > 40.0f);   // reject very dark pixels

    mask[y * w + x] = isSkin ? 1.0f : 0.0f;
}


// ===================================================================
// Kernel: bilateralFilterMaskedKernel
//
// Edge-preserving bilateral filter applied only where mask > 0.5.
// Processes only the bounding box region for performance.
// ===================================================================
__global__ static void bilateralFilterMaskedKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const float*   __restrict__ mask,
    int                          w,
    int                          h,
    int                          bboxX,
    int                          bboxY,
    int                          bboxW,
    int                          bboxH,
    float                        spatialCoeff,   // -0.5 / (sigmaSpatial^2)
    float                        colorCoeff,     // -0.5 / (sigmaColor^2)
    int                          radius,
    float                        strength)       // 0-1 blend factor
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x + bboxX;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y + bboxY;

    if (gx >= bboxX + bboxW || gy >= bboxY + bboxH) { return; }
    if (gx >= w || gy >= h) { return; }

    const int pixIdx = gy * w + gx;
    const int srcIdx = pixIdx * 3;

    // Copy unmasked pixels unchanged
    if (mask[pixIdx] <= 0.5f) {
        output[srcIdx]     = input[srcIdx];
        output[srcIdx + 1] = input[srcIdx + 1];
        output[srcIdx + 2] = input[srcIdx + 2];
        return;
    }

    const float cB = static_cast<float>(input[srcIdx]);
    const float cG = static_cast<float>(input[srcIdx + 1]);
    const float cR = static_cast<float>(input[srcIdx + 2]);

    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
    float sumWeight = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = gy + dy;
        if (ny < 0 || ny >= h) continue;

        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = gx + dx;
            if (nx < 0 || nx >= w) continue;

            const int nIdx = (ny * w + nx) * 3;
            const float nB = static_cast<float>(input[nIdx]);
            const float nG = static_cast<float>(input[nIdx + 1]);
            const float nR = static_cast<float>(input[nIdx + 2]);

            // Spatial weight: exp(-dist^2 / (2*sigmaSpatial^2))
            const float spatialDist = static_cast<float>(dx * dx + dy * dy);
            const float spatialWeight = expf(spatialDist * spatialCoeff);

            // Color weight: exp(-colorDist^2 / (2*sigmaColor^2))
            const float dB = nB - cB;
            const float dG = nG - cG;
            const float dR = nR - cR;
            const float colorDist = dB * dB + dG * dG + dR * dR;
            const float colorWeight = expf(colorDist * colorCoeff);

            const float weight = spatialWeight * colorWeight;
            sumB += nB * weight;
            sumG += nG * weight;
            sumR += nR * weight;
            sumWeight += weight;
        }
    }

    // Normalize and blend with original based on strength
    const float invW = (sumWeight > 1e-6f) ? (1.0f / sumWeight) : 0.0f;
    const float filtB = sumB * invW;
    const float filtG = sumG * invW;
    const float filtR = sumR * invW;

    // Blend: output = lerp(original, filtered, strength)
    const float outB = cB + strength * (filtB - cB);
    const float outG = cG + strength * (filtG - cG);
    const float outR = cR + strength * (filtR - cR);

    output[srcIdx]     = static_cast<uint8_t>(fminf(fmaxf(outB, 0.0f), 255.0f));
    output[srcIdx + 1] = static_cast<uint8_t>(fminf(fmaxf(outG, 0.0f), 255.0f));
    output[srcIdx + 2] = static_cast<uint8_t>(fminf(fmaxf(outR, 0.0f), 255.0f));
}


// ===================================================================
// Kernel: regionBrightnessKernel
//
// Additive brightness within a mask region. In-place on d_bgr.
// ===================================================================
__global__ static void regionBrightnessKernel(
    uint8_t*       __restrict__ bgr,
    const float*   __restrict__ regionMask,
    int                          w,
    int                          h,
    float                        amount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }
    if (regionMask[y * w + x] <= 0.5f) { return; }

    const int idx = (y * w + x) * 3;
    for (int c = 0; c < 3; ++c) {
        const float val = static_cast<float>(bgr[idx + c]) + amount;
        bgr[idx + c] = static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
    }
}


// ===================================================================
// Kernel: beautySharpenKernel — 3x3 unsharp mask (raw pointers)
// ===================================================================
__global__ static void beautySharpenKernel(
    const uint8_t* __restrict__ src,
    uint8_t*       __restrict__ dst,
    int                          w,
    int                          h,
    float                        amount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }

    const int idx = (y * w + x) * 3;

    // Border pixels: copy unchanged
    if (x == 0 || x >= w - 1 || y == 0 || y >= h - 1) {
        dst[idx]     = src[idx];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 2];
        return;
    }

    // 3x3 box average per channel
    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int nIdx = ((y + dy) * w + (x + dx)) * 3;
            sumB += src[nIdx];
            sumG += src[nIdx + 1];
            sumR += src[nIdx + 2];
        }
    }
    const float inv9 = 1.0f / 9.0f;

    const float cB = src[idx];
    const float cG = src[idx + 1];
    const float cR = src[idx + 2];

    // Unsharp mask: result = center + amount * (center - avg)
    const float outB = cB + amount * (cB - sumB * inv9);
    const float outG = cG + amount * (cG - sumG * inv9);
    const float outR = cR + amount * (cR - sumR * inv9);

    dst[idx]     = static_cast<uint8_t>(fminf(fmaxf(outB, 0.0f), 255.0f));
    dst[idx + 1] = static_cast<uint8_t>(fminf(fmaxf(outG, 0.0f), 255.0f));
    dst[idx + 2] = static_cast<uint8_t>(fminf(fmaxf(outR, 0.0f), 255.0f));
}


// ===================================================================
// Kernel: bcsKernel — brightness/contrast/saturation (in-place)
// ===================================================================
__global__ static void bcsKernel(
    uint8_t* __restrict__ bgr,
    int                    w,
    int                    h,
    float                  brightness,
    float                  contrast,
    float                  saturation)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }

    const int idx = (y * w + x) * 3;
    float b = static_cast<float>(bgr[idx]);
    float g = static_cast<float>(bgr[idx + 1]);
    float r = static_cast<float>(bgr[idx + 2]);

    // Contrast + Brightness
    b = b * contrast + brightness;
    g = g * contrast + brightness;
    r = r * contrast + brightness;

    // Saturation: interpolate toward luma
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    b = luma + saturation * (b - luma);
    g = luma + saturation * (g - luma);
    r = luma + saturation * (r - luma);

    bgr[idx]     = static_cast<uint8_t>(fminf(fmaxf(b, 0.0f), 255.0f));
    bgr[idx + 1] = static_cast<uint8_t>(fminf(fmaxf(g, 0.0f), 255.0f));
    bgr[idx + 2] = static_cast<uint8_t>(fminf(fmaxf(r, 0.0f), 255.0f));
}


// ===================================================================
// Kernel: warmthKernel — color temperature shift within face mask
// ===================================================================
__global__ static void warmthKernel(
    uint8_t*       __restrict__ bgr,
    const float*   __restrict__ faceMask,
    int                          w,
    int                          h,
    float                        warmthOffset)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }
    if (faceMask[y * w + x] <= 0.5f) { return; }

    const int idx = (y * w + x) * 3;
    // Warm: +R, -B. Cool: -R, +B
    const float b = static_cast<float>(bgr[idx])     - warmthOffset;
    const float r = static_cast<float>(bgr[idx + 2]) + warmthOffset;

    bgr[idx]     = static_cast<uint8_t>(fminf(fmaxf(b, 0.0f), 255.0f));
    bgr[idx + 2] = static_cast<uint8_t>(fminf(fmaxf(r, 0.0f), 255.0f));
}


// ===================================================================
// Kernel: shadowFillKernel — brighten low-luma face pixels
// ===================================================================
__global__ static void shadowFillKernel(
    uint8_t*       __restrict__ bgr,
    const float*   __restrict__ faceMask,
    int                          w,
    int                          h,
    float                        amount)        // 0-1 normalized
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }
    if (faceMask[y * w + x] <= 0.5f) { return; }

    const int idx = (y * w + x) * 3;
    const float b = static_cast<float>(bgr[idx]);
    const float g = static_cast<float>(bgr[idx + 1]);
    const float r = static_cast<float>(bgr[idx + 2]);

    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;

    // Only boost shadows (luma < 80)
    const float shadowThreshold = 80.0f;
    if (luma >= shadowThreshold) { return; }

    // Boost proportional to how dark the pixel is
    const float boostFactor = amount * (1.0f - luma / shadowThreshold) * 40.0f;

    bgr[idx]     = static_cast<uint8_t>(fminf(b + boostFactor, 255.0f));
    bgr[idx + 1] = static_cast<uint8_t>(fminf(g + boostFactor, 255.0f));
    bgr[idx + 2] = static_cast<uint8_t>(fminf(r + boostFactor, 255.0f));
}


// ===================================================================
// Kernel: highlightBoostKernel — subtle specular boost on bright face pixels
// ===================================================================
__global__ static void highlightBoostKernel(
    uint8_t*       __restrict__ bgr,
    const float*   __restrict__ faceMask,
    int                          w,
    int                          h,
    float                        amount)        // 0-1 normalized
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }
    if (faceMask[y * w + x] <= 0.5f) { return; }

    const int idx = (y * w + x) * 3;
    const float b = static_cast<float>(bgr[idx]);
    const float g = static_cast<float>(bgr[idx + 1]);
    const float r = static_cast<float>(bgr[idx + 2]);

    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;

    // Only boost highlights (luma > 180)
    const float highlightThreshold = 180.0f;
    if (luma <= highlightThreshold) { return; }

    const float boostFactor = amount * ((luma - highlightThreshold) / (255.0f - highlightThreshold)) * 25.0f;

    bgr[idx]     = static_cast<uint8_t>(fminf(b + boostFactor, 255.0f));
    bgr[idx + 1] = static_cast<uint8_t>(fminf(g + boostFactor, 255.0f));
    bgr[idx + 2] = static_cast<uint8_t>(fminf(r + boostFactor, 255.0f));
}


// ===================================================================
// Kernel: buildUnderEyeMaskKernel
//
// Fills an elliptical region mask from landmark points.
// The ellipse is fit to the bounding box of the landmarks with dilation.
// ===================================================================
__global__ static void buildUnderEyeMaskKernel(
    float*  __restrict__ mask,
    int                   w,
    int                   h,
    float                 cx,       // ellipse center x
    float                 cy,       // ellipse center y
    float                 rx,       // ellipse radius x
    float                 ry,       // ellipse radius y
    float                 dilate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) { return; }

    const float dx = (static_cast<float>(x) - cx) / (rx + dilate);
    const float dy = (static_cast<float>(y) - cy) / (ry + dilate);

    if (dx * dx + dy * dy <= 1.0f) {
        mask[y * w + x] = 1.0f;
    }
}


// ===================================================================
// Kernel: tpsWarpKernel — Thin-Plate Spline face warp
//
// For each output pixel within the face bounding box, compute the
// TPS-warped source coordinate and sample with bilinear interpolation.
//
// TPS mapping: f(x,y) = a0 + ax*x + ay*y + sum_i(w_i * U(r_i))
// where U(r) = r^2 * log(r), r_i = ||(x,y) - p_i||
//
// Weights layout: [w_0, w_1, ..., w_{N-1}, a_0, a_x, a_y]
// ===================================================================
__global__ static void tpsWarpKernel(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const float*   __restrict__ srcX,
    const float*   __restrict__ srcY,
    const float*   __restrict__ weightsX,
    const float*   __restrict__ weightsY,
    int                          numPoints,
    int                          w,
    int                          h,
    int                          bboxX,
    int                          bboxY,
    int                          bboxW,
    int                          bboxH)
{
    const int lx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = blockIdx.y * blockDim.y + threadIdx.y;

    if (lx >= bboxW || ly >= bboxH) { return; }

    const int gx = lx + bboxX;
    const int gy = ly + bboxY;
    if (gx >= w || gy >= h) { return; }

    const float px = static_cast<float>(gx);
    const float py = static_cast<float>(gy);

    // Affine part: a_0 + a_x * x + a_y * y
    float mapX = weightsX[numPoints] + weightsX[numPoints + 1] * px + weightsX[numPoints + 2] * py;
    float mapY = weightsY[numPoints] + weightsY[numPoints + 1] * px + weightsY[numPoints + 2] * py;

    // TPS basis: sum_i(w_i * U(||(px,py) - p_i||))
    for (int i = 0; i < numPoints; ++i) {
        const float dx = px - srcX[i];
        const float dy = py - srcY[i];
        const float r2 = dx * dx + dy * dy;
        if (r2 > 1e-6f) {
            const float r = sqrtf(r2);
            const float u = r2 * logf(r);  // U(r) = r^2 * log(r)
            mapX += weightsX[i] * u;
            mapY += weightsY[i] * u;
        }
    }

    // Clamp mapped coordinates to image bounds
    mapX = fmaxf(0.0f, fminf(mapX, static_cast<float>(w - 1)));
    mapY = fmaxf(0.0f, fminf(mapY, static_cast<float>(h - 1)));

    // Bilinear interpolation
    const int floorX = static_cast<int>(floorf(mapX));
    const int floorY = static_cast<int>(floorf(mapY));
    const int ceilX  = min(floorX + 1, w - 1);
    const int ceilY  = min(floorY + 1, h - 1);

    const float wx = mapX - floorf(mapX);
    const float wy = mapY - floorf(mapY);

    const uint8_t* tl = input + (floorY * w + floorX) * 3;
    const uint8_t* tr = input + (floorY * w + ceilX)  * 3;
    const uint8_t* bl = input + (ceilY  * w + floorX) * 3;
    const uint8_t* br = input + (ceilY  * w + ceilX)  * 3;

    const int outIdx = (gy * w + gx) * 3;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float val =
            tl[c] * (1.0f - wx) * (1.0f - wy) +
            tr[c] * wx          * (1.0f - wy) +
            bl[c] * (1.0f - wx) * wy          +
            br[c] * wx          * wy;
        output[outIdx + c] = static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
    }
}


// ===================================================================
// Host-side launch wrappers
// ===================================================================

namespace oe::beauty {

void launchResizeNormForFaceMesh(
    const uint8_t* d_bgrSrc, int srcW, int srcH,
    float*         d_rgbDst, int dstW, int dstH,
    cudaStream_t   stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y);

    resizeNormFaceMeshKernel<<<grid, block, 0, stream>>>(
        d_bgrSrc, srcW, srcH,
        d_rgbDst, dstW, dstH);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchComputeSkinMask(
    const uint8_t* d_bgr,
    float*         d_mask,
    int            w, int h,
    int            bboxX, int bboxY, int bboxW, int bboxH,
    float          crMin, float crMax,
    float          cbMin, float cbMax,
    cudaStream_t   stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    skinMaskKernel<<<grid, block, 0, stream>>>(
        d_bgr, d_mask, w, h,
        bboxX, bboxY, bboxW, bboxH,
        crMin, crMax, cbMin, cbMax);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchBilateralFilterMasked(
    const uint8_t* d_input,
    uint8_t*       d_output,
    const float*   d_mask,
    int            w, int h,
    int            bboxX, int bboxY, int bboxW, int bboxH,
    float          sigmaSpatial,
    float          sigmaColor,
    int            radius,
    float          strength,
    cudaStream_t   stream)
{
    // First: copy entire frame to output (unmasked pixels)
    OE_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input,
        static_cast<std::size_t>(w) * h * 3, cudaMemcpyDeviceToDevice, stream));

    // Strength slider: 0-100 -> 0.0-1.0
    const float normalizedStrength = fminf(fmaxf(strength / 100.0f, 0.0f), 1.0f);
    if (normalizedStrength < 0.01f) { return; }  // no smoothing

    const float spatialCoeff = -0.5f / (sigmaSpatial * sigmaSpatial);
    const float colorCoeff   = -0.5f / (sigmaColor * sigmaColor);

    // Launch grid covers only the bounding box
    const dim3 block(16, 16);
    const dim3 grid(
        (bboxW + block.x - 1) / block.x,
        (bboxH + block.y - 1) / block.y);

    bilateralFilterMaskedKernel<<<grid, block, 0, stream>>>(
        d_input, d_output, d_mask, w, h,
        bboxX, bboxY, bboxW, bboxH,
        spatialCoeff, colorCoeff, radius,
        normalizedStrength);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchRegionBrightnessAdjust(
    uint8_t*       d_bgr,
    const float*   d_regionMask,
    int            w, int h,
    float          amount,
    cudaStream_t   stream)
{
    if (fabsf(amount) < 0.5f) { return; }  // no-op

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    regionBrightnessKernel<<<grid, block, 0, stream>>>(
        d_bgr, d_regionMask, w, h, amount);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchBeautySharpen(
    const uint8_t* d_src,
    uint8_t*       d_dst,
    int            w, int h,
    float          amount,
    cudaStream_t   stream)
{
    if (amount < 0.01f) {
        // No sharpen: copy src to dst
        OE_CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src,
            static_cast<std::size_t>(w) * h * 3, cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    beautySharpenKernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, w, h, amount);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchBcsAdjust(
    uint8_t*     d_bgr,
    int          w, int h,
    float        brightness,
    float        contrast,
    float        saturation,
    cudaStream_t stream)
{
    // Skip if all params are at identity
    if (fabsf(brightness) < 0.5f && fabsf(contrast - 1.0f) < 0.01f &&
        fabsf(saturation - 1.0f) < 0.01f) {
        return;
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    bcsKernel<<<grid, block, 0, stream>>>(
        d_bgr, w, h, brightness, contrast, saturation);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchWarmthAdjust(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          warmth,
    cudaStream_t   stream)
{
    if (fabsf(warmth) < 0.5f) { return; }

    // Scale warmth (-100..+100) to pixel offset
    const float warmthOffset = warmth * 0.3f;  // max ±30 pixel offset

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    warmthKernel<<<grid, block, 0, stream>>>(
        d_bgr, d_faceMask, w, h, warmthOffset);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchShadowFill(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          amount,
    cudaStream_t   stream)
{
    if (amount < 0.5f) { return; }

    const float normalized = amount / 100.0f;

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    shadowFillKernel<<<grid, block, 0, stream>>>(
        d_bgr, d_faceMask, w, h, normalized);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchHighlightBoost(
    uint8_t*       d_bgr,
    const float*   d_faceMask,
    int            w, int h,
    float          amount,
    cudaStream_t   stream)
{
    if (amount < 0.5f) { return; }

    const float normalized = amount / 100.0f;

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    highlightBoostKernel<<<grid, block, 0, stream>>>(
        d_bgr, d_faceMask, w, h, normalized);
    OE_CUDA_CHECK(cudaGetLastError());
}

void launchBuildUnderEyeMask(
    float*         d_regionMask,
    int            w, int h,
    const float*   landmarkX,
    const float*   landmarkY,
    int            numPoints,
    float          dilate,
    cudaStream_t   stream)
{
    // Compute ellipse bounding box on host from landmark points
    float minX = 1e6f, maxX = -1e6f;
    float minY = 1e6f, maxY = -1e6f;
    for (int i = 0; i < numPoints; ++i) {
        minX = fminf(minX, landmarkX[i]);
        maxX = fmaxf(maxX, landmarkX[i]);
        minY = fminf(minY, landmarkY[i]);
        maxY = fmaxf(maxY, landmarkY[i]);
    }

    const float cx = (minX + maxX) * 0.5f;
    const float cy = (minY + maxY) * 0.5f;
    const float rx = (maxX - minX) * 0.5f + 1.0f;  // +1 to avoid zero radius
    const float ry = (maxY - minY) * 0.5f + 1.0f;

    const dim3 block(16, 16);
    const dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    buildUnderEyeMaskKernel<<<grid, block, 0, stream>>>(
        d_regionMask, w, h, cx, cy, rx, ry, dilate);
    OE_CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t   stream)
{
    // First: copy entire frame (background pixels stay unchanged)
    OE_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input,
        static_cast<std::size_t>(w) * h * 3, cudaMemcpyDeviceToDevice, stream));

    if (numPoints <= 0 || bboxW <= 0 || bboxH <= 0) return;

    // Launch warp kernel only within the bounding box
    const dim3 block(16, 16);
    const dim3 grid(
        (bboxW + block.x - 1) / block.x,
        (bboxH + block.y - 1) / block.y);

    tpsWarpKernel<<<grid, block, 0, stream>>>(
        d_input, d_output,
        d_srcX, d_srcY, d_weightsX, d_weightsY,
        numPoints, w, h,
        bboxX, bboxY, bboxW, bboxH);
    OE_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Kernel: compositeSolidBgKernel
//
// output = (mask > 0.5) ? foreground : solid bg colour
// ---------------------------------------------------------------------------

__global__ static void compositeSolidBgKernel(
    const uint8_t* __restrict__ fg,
    uint8_t*       __restrict__ out,
    const float*   __restrict__ mask,
    int w, int h,
    uint8_t bgB, uint8_t bgG, uint8_t bgR)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx  = y * w + x;
    const int idx3 = idx * 3;
    const float m  = mask[idx];

    if (m > 0.5f) {
        out[idx3]     = fg[idx3];
        out[idx3 + 1] = fg[idx3 + 1];
        out[idx3 + 2] = fg[idx3 + 2];
    } else {
        out[idx3]     = bgB;
        out[idx3 + 1] = bgG;
        out[idx3 + 2] = bgR;
    }
}

void launchCompositeSolidBg(
    const uint8_t* d_foreground,
    uint8_t*       d_output,
    const float*   d_mask,
    int            w, int h,
    uint8_t        bgB, uint8_t bgG, uint8_t bgR,
    cudaStream_t   stream)
{
    const dim3 block(16, 16);
    const dim3 grid((w + 15) / 16, (h + 15) / 16);

    compositeSolidBgKernel<<<grid, block, 0, stream>>>(
        d_foreground, d_output, d_mask,
        w, h, bgB, bgG, bgR);
    OE_CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// Kernel: compositeBgImageKernel
//
// output = (mask > 0.5) ? foreground : bgImage
// ---------------------------------------------------------------------------

__global__ static void compositeBgImageKernel(
    const uint8_t* __restrict__ fg,
    uint8_t*       __restrict__ out,
    const float*   __restrict__ mask,
    const uint8_t* __restrict__ bg,
    int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx  = y * w + x;
    const int idx3 = idx * 3;
    const float m  = mask[idx];

    if (m > 0.5f) {
        out[idx3]     = fg[idx3];
        out[idx3 + 1] = fg[idx3 + 1];
        out[idx3 + 2] = fg[idx3 + 2];
    } else {
        out[idx3]     = bg[idx3];
        out[idx3 + 1] = bg[idx3 + 1];
        out[idx3 + 2] = bg[idx3 + 2];
    }
}

void launchCompositeBgImage(
    const uint8_t* d_foreground,
    uint8_t*       d_output,
    const float*   d_mask,
    const uint8_t* d_bgImage,
    int            w, int h,
    cudaStream_t   stream)
{
    const dim3 block(16, 16);
    const dim3 grid((w + 15) / 16, (h + 15) / 16);

    compositeBgImageKernel<<<grid, block, 0, stream>>>(
        d_foreground, d_output, d_mask, d_bgImage,
        w, h);
    OE_CUDA_CHECK(cudaGetLastError());
}

} // namespace oe::beauty
