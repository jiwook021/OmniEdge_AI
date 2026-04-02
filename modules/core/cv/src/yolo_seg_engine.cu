#include "cv/yolo_seg_engine.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gpu/cuda_stream.hpp"
#include "common/oe_logger.hpp"


// ---------------------------------------------------------------------------
// TrtLogger
// ---------------------------------------------------------------------------

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
        logJson("error", "trt_engine_log", {{"msg", msg}});
    } else if (severity == Severity::kWARNING) {
        logJson("warn", "trt_engine_log", {{"msg", msg}});
    }
    // Suppress INFO / VERBOSE in production
}

// ---------------------------------------------------------------------------
// CUDA kernels (device code compiled via nvcc)
// ---------------------------------------------------------------------------

namespace {

// ---------------------------------------------------------------------------
// resizeNormFP16Kernel
//
// Bilinear-resize a BGR24 frame from (srcW×srcH) to (dstW×dstH) and
// produce an NCHW FP16 output normalised to [0, 1].
// BGR channel order → R, G, B plane order (YOLO convention).
// ---------------------------------------------------------------------------

__global__ void resizeNormFP16Kernel(
    const uint8_t* __restrict__ src,
    int                          srcW,
    int                          srcH,
    __half* __restrict__         dst,
    int                          dstW,
    int                          dstH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) { return; }

    // Source coordinates (bilinear interpolation)
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

    // Bilinear sample — BGR24 input (3 bytes/pixel, no padding)
    const uint8_t* topLeftPixel     = src + (floorY * srcW + floorX) * 3;
    const uint8_t* topRightPixel    = src + (floorY * srcW + ceilX)  * 3;
    const uint8_t* bottomLeftPixel  = src + (ceilY  * srcW + floorX) * 3;
    const uint8_t* bottomRightPixel = src + (ceilY  * srcW + ceilX)  * 3;

    const int planeStride = dstH * dstW;

    // Channel loop: src = BGR (0=B, 1=G, 2=R); dst = RGB planes (0=R,1=G,2=B)
    const int bgrToRgbChannelMap[3] = {2, 1, 0};  // read R, G, B from BGR
    for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
        const int sourceChannel = bgrToRgbChannelMap[channelIndex];
        const float interpolatedValue =
            (topLeftPixel[sourceChannel]     * (1.0f - weightX) + topRightPixel[sourceChannel]    * weightX) * (1.0f - weightY) +
            (bottomLeftPixel[sourceChannel]  * (1.0f - weightX) + bottomRightPixel[sourceChannel] * weightX) * weightY;
        dst[channelIndex * planeStride + y * dstW + x] = __float2half(interpolatedValue / 255.0f);
    }
}

// ---------------------------------------------------------------------------
// computeMaskKernel
//
// Compute the sigmoid-activated combined mask from:
//   prototypes : [32, maskH, maskW]   (device, float)
//   coeffs     : [32]                 (device, float)
//   out        : [maskH, maskW]       (device, float) — result
//
// Each output pixel = sigmoid( dot(coeffs, prototypes[:, y, x]) )
// ---------------------------------------------------------------------------

__global__ void computeMaskKernel(
    const float* __restrict__ prototypes,
    const float* __restrict__ coeffs,
    float*       __restrict__ out,
    int                       maskW,
    int                       maskH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= maskW || y >= maskH) { return; }

    const int pixelStride = maskH * maskW;
    float dotProduct = 0.0f;
    for (int protoIndex = 0; protoIndex < kYoloMaskProtos; ++protoIndex) {
        dotProduct += coeffs[protoIndex] * prototypes[protoIndex * pixelStride + y * maskW + x];
    }
    // Sigmoid activation
    out[y * maskW + x] = 1.0f / (1.0f + expf(-dotProduct));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

std::vector<char> loadEngine(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error(
            std::format("[YoloSegEngine] Cannot open engine file: {}", path));
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineBytes(static_cast<std::size_t>(size));
    if (!file.read(engineBytes.data(), size)) {
        throw std::runtime_error(
            std::format("[YoloSegEngine] Failed to read engine: {}", path));
    }
    return engineBytes;
}

// Intersection-over-Union for Non-Maximum Suppression
float computeIntersectionOverUnion(
    float boxALeft, float boxATop, float boxARight, float boxABottom,
    float boxBLeft, float boxBTop, float boxBRight, float boxBBottom) noexcept
{
    const float overlapLeft   = std::max(boxALeft,  boxBLeft);
    const float overlapTop    = std::max(boxATop,   boxBTop);
    const float overlapRight  = std::min(boxARight, boxBRight);
    const float overlapBottom = std::min(boxABottom, boxBBottom);
    if (overlapRight <= overlapLeft || overlapBottom <= overlapTop) { return 0.0f; }
    const float intersectionArea = (overlapRight - overlapLeft) * (overlapBottom - overlapTop);
    const float boxAArea = (boxARight - boxALeft) * (boxABottom - boxATop);
    const float boxBArea = (boxBRight - boxBLeft) * (boxBBottom - boxBTop);
    return intersectionArea / (boxAArea + boxBArea - intersectionArea + 1e-6f);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// YoloSegEngine lifecycle
// ---------------------------------------------------------------------------

YoloSegEngine::~YoloSegEngine()
{
    destroy();
}

void YoloSegEngine::initialize(const std::string& enginePath)
{
    // Load serialized engine bytes
    const auto engineData = loadEngine(enginePath);

    // Create TRT runtime and deserialize
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        throw std::runtime_error("[YoloSegEngine] createInferRuntime failed");
    }

    engine_ = runtime_->deserializeCudaEngine(
        engineData.data(), engineData.size());
    if (!engine_) {
        throw std::runtime_error(
            std::format("[YoloSegEngine] deserializeCudaEngine failed: {}", enginePath));
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("[YoloSegEngine] createExecutionContext failed");
    }

    allocateBuffers();

    logJson("info", "yolo_seg_engine_ready", {{"engine", enginePath}});
}

void YoloSegEngine::allocateBuffers()
{
    // Input: [1, 3, 416, 416] FP16
    const std::size_t inputBytes =
        static_cast<std::size_t>(1 * 3 * kYoloInputH * kYoloInputW) * sizeof(__half);
    OE_CUDA_CHECK(cudaMalloc(&d_input_, inputBytes));

    // Output0: [1, 116, 8400] FP32
    const std::size_t out0Bytes =
        static_cast<std::size_t>(1 * kYoloDetectionDim * kYoloNumDetections) * sizeof(float);
    OE_CUDA_CHECK(cudaMalloc(&d_output0_, out0Bytes));

    // Output1: [1, 32, 160, 160] FP32
    const std::size_t out1Bytes =
        static_cast<std::size_t>(1 * kYoloMaskProtos * kYoloMaskH * kYoloMaskW) * sizeof(float);
    OE_CUDA_CHECK(cudaMalloc(&d_output1_, out1Bytes));

    // Mask result: [160×160] float
    const std::size_t maskBytes =
        static_cast<std::size_t>(kYoloMaskH * kYoloMaskW) * sizeof(float);
    OE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segmentationMask_), maskBytes));

    // Set tensor addresses for TRT 10 API
    if (!context_->setTensorAddress("images", d_input_)) {
        throw std::runtime_error("[YoloSegEngine] setTensorAddress('images') failed");
    }
    if (!context_->setTensorAddress("output0", d_output0_)) {
        throw std::runtime_error("[YoloSegEngine] setTensorAddress('output0') failed");
    }
    if (!context_->setTensorAddress("output1", d_output1_)) {
        throw std::runtime_error("[YoloSegEngine] setTensorAddress('output1') failed");
    }
}

void YoloSegEngine::destroy() noexcept
{
    if (d_input_)      { cudaFree(d_input_);      d_input_     = nullptr; }
    if (d_output0_)    { cudaFree(d_output0_);    d_output0_   = nullptr; }
    if (d_output1_)    { cudaFree(d_output1_);    d_output1_   = nullptr; }
    if (d_segmentationMask_)    { cudaFree(d_segmentationMask_);    d_segmentationMask_   = nullptr; }
    if (d_resizeBuffer_) { cudaFree(d_resizeBuffer_); d_resizeBuffer_= nullptr; }
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

// ---------------------------------------------------------------------------
// infer() — full pipeline: BGR1080 → SegResult
// ---------------------------------------------------------------------------

SegResult YoloSegEngine::infer(const uint8_t* d_bgr1080, cudaStream_t stream)
{
    constexpr uint32_t W = 1920, H = 1080;

    // 1. Preprocess: resize BGR24 1920×1080 → FP16 NCHW 416×416 (normalized)
    {
        const dim3 block(16, 16);
        const dim3 grid(
            (kYoloInputW + block.x - 1) / block.x,
            (kYoloInputH + block.y - 1) / block.y);

        resizeNormFP16Kernel<<<grid, block, 0, stream>>>(
            d_bgr1080,
            static_cast<int>(W), static_cast<int>(H),
            static_cast<__half*>(d_input_),
            kYoloInputW, kYoloInputH);
        OE_CUDA_CHECK(cudaGetLastError());
    }

    // 2. TRT inference (TRT 10 API)
    if (!context_->enqueueV3(stream)) {
        logJson("warn", "yolo_infer_enqueue_failed", {});
        return SegResult{};
    }

    // 3. Post-process on CPU: copy detection output and find best person
    const std::size_t out0Bytes =
        static_cast<std::size_t>(kYoloDetectionDim * kYoloNumDetections) * sizeof(float);
    std::vector<float> h_output0(kYoloDetectionDim * kYoloNumDetections);
    OE_CUDA_CHECK(cudaMemcpyAsync(
        h_output0.data(), d_output0_, out0Bytes,
        cudaMemcpyDeviceToHost, stream));
    OE_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Detection layout: [116, 8400] stored as [dim, det]
    // For detection i:  cx=out[0*8400+i], cy=out[1*8400+i], w=out[2*8400+i], h=out[3*8400+i]
    //                   scores: out[4..83]*8400+i   (class 0 = person)
    //                   mask_coeffs: out[84..115]*8400+i

    int    bestIdx   = -1;
    float  bestScore = kYoloConfThreshold;

    struct Detection {
        float cx, cy, w, h;
        float score;
        float coeffs[kYoloMaskProtos];
    };
    std::vector<Detection> candidates;
    candidates.reserve(64);

    for (int i = 0; i < kYoloNumDetections; ++i) {
        const float personScore = h_output0[4 * kYoloNumDetections + i];  // class 0
        if (personScore < kYoloConfThreshold) { continue; }

        Detection d;
        d.cx    = h_output0[0 * kYoloNumDetections + i];
        d.cy    = h_output0[1 * kYoloNumDetections + i];
        d.w     = h_output0[2 * kYoloNumDetections + i];
        d.h     = h_output0[3 * kYoloNumDetections + i];
        d.score = personScore;
        for (int k = 0; k < kYoloMaskProtos; ++k) {
            d.coeffs[k] = h_output0[(84 + k) * kYoloNumDetections + i];
        }
        candidates.push_back(d);
    }

    if (candidates.empty()) {
        return SegResult{};
    }

    // Sort descending by score
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score; });

    // Greedy NMS — keep best non-overlapping person detections
    std::vector<int> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) { continue; }
        kept.push_back(static_cast<int>(i));
        const auto& a = candidates[i];
        const float ax1 = a.cx - a.w * 0.5f;
        const float ay1 = a.cy - a.h * 0.5f;
        const float ax2 = a.cx + a.w * 0.5f;
        const float ay2 = a.cy + a.h * 0.5f;
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) { continue; }
            const auto& b = candidates[j];
            if (computeIntersectionOverUnion(ax1, ay1, ax2, ay2,
                    b.cx - b.w * 0.5f, b.cy - b.h * 0.5f,
                    b.cx + b.w * 0.5f, b.cy + b.h * 0.5f) > kYoloNmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    // Use the top-scored kept detection for mask generation
    bestIdx = kept[0];
    bestScore = candidates[bestIdx].score;

    // 4. Upload mask coefficients for the best detection to GPU
    float  h_coeffs[kYoloMaskProtos];
    for (int k = 0; k < kYoloMaskProtos; ++k) {
        h_coeffs[k] = candidates[bestIdx].coeffs[k];
    }
    float* d_coeffs = nullptr;
    OE_CUDA_CHECK(cudaMalloc(&d_coeffs, kYoloMaskProtos * sizeof(float)));
    OE_CUDA_CHECK(cudaMemcpyAsync(
        d_coeffs, h_coeffs, kYoloMaskProtos * sizeof(float),
        cudaMemcpyHostToDevice, stream));

    // 5. Compute the mask: dot(prototypes, coeffs) + sigmoid → d_segmentationMask_
    {
        const dim3 block(16, 16);
        const dim3 grid(
            (kYoloMaskW + block.x - 1) / block.x,
            (kYoloMaskH + block.y - 1) / block.y);

        computeMaskKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(d_output1_),
            d_coeffs,
            d_segmentationMask_,
            kYoloMaskW,
            kYoloMaskH);
        OE_CUDA_CHECK(cudaGetLastError());
    }

    OE_CUDA_CHECK(cudaFreeAsync(d_coeffs, stream));

    return SegResult{
        .hasPerson  = true,
        .confidence = bestScore,
        .d_segmentationMask  = d_segmentationMask_,
    };
}

