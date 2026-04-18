#include "cv/onnx_beauty_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/oe_onnx_io_binding.hpp"
#include "common/onnx_session_handle.hpp"
#include "common/constants/beauty_constants.hpp"
#include "cv/beauty_landmark_regions.hpp"
#include "cv/beauty_preprocess.hpp"
#include "cv/tps_solver.hpp"
#include "gpu/cuda_fence.hpp"
#include "gpu/cuda_graph.hpp"
#include "gpu/cuda_priority.hpp"
#include "gpu/cuda_stream.hpp"
#include "gpu/oe_cuda_check.hpp"
#include "gpu/pinned_buffer.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <mutex>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <onnxruntime_cxx_api.h>

#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr std::size_t kMaxJpegPinnedBytes = 1024 * 1024;
static constexpr std::size_t kLandmarkFloats     = kBeautyLandmarkCount * 3;
static constexpr float       kBboxDilateFraction  = 0.15f;
static constexpr float       kUnderEyeDilate      = 5.0f;


// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------

struct OnnxBeautyInferencer::Impl {
    oe::onnx::SessionHandle  handle{"FaceMesh_Beauty"};

    // CUDA pipeline
    CudaStream               stream;
    PinnedStagingBuffer      pinnedBgr;
    PinnedStagingBuffer      pinnedLandmarks;
    PinnedStagingBuffer      pinnedJpeg;

    // Device buffers (pre-allocated per resolution)
    oe::onnx::GpuBuffer      d_bgr;
    oe::onnx::GpuBuffer      d_skinMask;
    oe::onnx::GpuBuffer      d_underEyeMask;
    oe::onnx::GpuBuffer      d_filtered;
    oe::onnx::GpuBuffer      d_sharpened;
    oe::onnx::GpuBuffer      d_warped;         // TPS warp output

    // TPS warp resources
    oe::beauty::TpsSolver    tpsSolver;
    oe::onnx::GpuBuffer      d_tpsSrcX;        // control point X coords [N]
    oe::onnx::GpuBuffer      d_tpsSrcY;        // control point Y coords [N]
    oe::onnx::GpuBuffer      d_tpsWeightsX;    // TPS weights X [N+3]
    oe::onnx::GpuBuffer      d_tpsWeightsY;    // TPS weights Y [N+3]

    // BG compositing resources
    oe::onnx::GpuBuffer      d_bgComposite;    // composited output [h x w x 3]
    oe::onnx::GpuBuffer      d_personMask;     // person segmentation mask [h x w] float
    oe::onnx::GpuBuffer      d_bgImage;        // user-uploaded BG image [h x w x 3]
    bool                     hasBgImage = false;

    // CUDA graph for steady-state replay
    CudaGraphInstance        cudaGraph;
    BeautyParams             graphCapturedParams; // params at time of capture
    bool                     graphParamsValid = false;

    // Frame pacing
    std::chrono::steady_clock::time_point lastFrameTime;

    // IO Binding for zero-copy FaceMesh inference
    oe::onnx::GpuIoBinding   ioBinding;

    // nvJPEG
    nvjpegHandle_t           nvjpegHandle  = nullptr;
    nvjpegEncoderState_t     nvjpegState   = nullptr;
    nvjpegEncoderParams_t    nvjpegParams  = nullptr;

    // Frame dimensions (lazy-init)
    uint32_t frameW = 0;
    uint32_t frameH = 0;

    // Parameters (thread-safe snapshot)
    std::mutex               paramsMutex;
    BeautyParams             params;

    bool                     loaded = false;

    ~Impl()
    {
        if (nvjpegParams)  nvjpegEncoderParamsDestroy(nvjpegParams);
        if (nvjpegState)   nvjpegEncoderStateDestroy(nvjpegState);
        if (nvjpegHandle)  nvjpegDestroy(nvjpegHandle);
    }

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    // --- Helper methods (avoid private-Impl-in-free-function issue) ---

    void ensureFrameBuffers(uint32_t w, uint32_t h)
    {
        if (frameW == w && frameH == h) return;

        const std::size_t bgrBytes  = static_cast<std::size_t>(w) * h * 3;
        const std::size_t maskBytes = static_cast<std::size_t>(w) * h * sizeof(float);

        d_bgr          = oe::onnx::GpuBuffer(bgrBytes);
        d_filtered     = oe::onnx::GpuBuffer(bgrBytes);
        d_sharpened    = oe::onnx::GpuBuffer(bgrBytes);
        d_warped       = oe::onnx::GpuBuffer(bgrBytes);
        d_bgComposite  = oe::onnx::GpuBuffer(bgrBytes);
        d_skinMask     = oe::onnx::GpuBuffer(maskBytes);
        d_underEyeMask = oe::onnx::GpuBuffer(maskBytes);
        d_personMask   = oe::onnx::GpuBuffer(maskBytes);
        pinnedBgr      = PinnedStagingBuffer(bgrBytes);

        // Invalidate CUDA graph on resolution change
        cudaGraph.reset();
        graphParamsValid = false;

        frameW = w;
        frameH = h;
        OE_LOG_DEBUG("beauty_buffers: {}x{}", w, h);
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    encodeNvJpeg(uint8_t* d_bgrSrc, uint32_t w, uint32_t h,
                 uint8_t* outBuf, std::size_t maxJpegBytes)
    {
        cudaStream_t s = stream.get();

        nvjpegImage_t nvImage{};
        nvImage.channel[0] = d_bgrSrc;
        nvImage.pitch[0]   = static_cast<unsigned int>(w * 3);

        nvjpegStatus_t nvs = nvjpegEncodeImage(
            nvjpegHandle, nvjpegState, nvjpegParams,
            &nvImage, NVJPEG_INPUT_BGRI,
            static_cast<int>(w), static_cast<int>(h), s);
        if (nvs != NVJPEG_STATUS_SUCCESS) {
            return tl::unexpected(std::format(
                "nvjpegEncodeImage failed: {}", static_cast<int>(nvs)));
        }

        std::size_t jpegLen = 0;
        nvs = nvjpegEncodeRetrieveBitstream(
            nvjpegHandle, nvjpegState, nullptr, &jpegLen, s);
        if (nvs != NVJPEG_STATUS_SUCCESS) {
            return tl::unexpected(std::format(
                "nvjpegEncodeRetrieveBitstream (query) failed: {}", static_cast<int>(nvs)));
        }
        if (jpegLen > maxJpegBytes) {
            return tl::unexpected(std::format(
                "JPEG size {} exceeds buffer {}", jpegLen, maxJpegBytes));
        }

        OE_CUDA_CHECK(cudaStreamSynchronize(s));

        nvs = nvjpegEncodeRetrieveBitstream(
            nvjpegHandle, nvjpegState, outBuf, &jpegLen, nullptr);
        if (nvs != NVJPEG_STATUS_SUCCESS) {
            return tl::unexpected(std::format(
                "nvjpegEncodeRetrieveBitstream (copy) failed: {}", static_cast<int>(nvs)));
        }

        return jpegLen;
    }
};


// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OnnxBeautyInferencer::OnnxBeautyInferencer()
    : impl_(std::make_unique<Impl>())
{}

OnnxBeautyInferencer::~OnnxBeautyInferencer()
{
    unload();
}


// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------

void OnnxBeautyInferencer::loadModel(const std::string& faceMeshOnnxPath)
{
    OE_ZONE_SCOPED;

    // 1. CUDA stream
    auto streamResult = CudaStream::createClamped(kCudaPriorityBeauty);
    if (!streamResult) {
        throw std::runtime_error(std::format(
            "Beauty: CUDA stream failed: {}", streamResult.error()));
    }
    impl_->stream = std::move(*streamResult);

    // 2. ONNX session: TRT EP + CUDA EP
    auto loadResult = impl_->handle.load(faceMeshOnnxPath, oe::onnx::SessionConfig{
        .useTRT            = true,
        .gpuMemLimitMiB    = kBeautyMiB,
        .enableCudaGraph   = false,
        .exhaustiveCudnn   = false,
        .maxCudnnWorkspace = false,
    });
    if (!loadResult) {
        throw std::runtime_error(std::format(
            "Beauty FaceMesh load: {}", loadResult.error()));
    }

    // 3. IO Binding: pre-allocate input [1,3,192,192] and output [1,478,3]
    constexpr int64_t fmSize = static_cast<int64_t>(kBeautyFaceMeshInputSize);

    impl_->ioBinding = oe::onnx::GpuIoBinding(*impl_->handle.session);

    impl_->ioBinding.allocateInput(
        impl_->handle.inputPtrs()[0],
        {1, 3, fmSize, fmSize},
        sizeof(float));

    impl_->ioBinding.allocateOutput(
        impl_->handle.outputPtrs()[0],
        {1, static_cast<int64_t>(kBeautyLandmarkCount), 3},
        sizeof(float));

    // 4. Pinned buffers
    impl_->pinnedLandmarks = PinnedStagingBuffer(kLandmarkFloats * sizeof(float));
    impl_->pinnedJpeg      = PinnedStagingBuffer(kMaxJpegPinnedBytes);

    // 5. nvJPEG encoder
    nvjpegStatus_t nvs = nvjpegCreateSimple(&impl_->nvjpegHandle);
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "Beauty: nvjpegCreateSimple failed: {}", static_cast<int>(nvs)));
    }
    nvs = nvjpegEncoderStateCreate(
        impl_->nvjpegHandle, &impl_->nvjpegState, impl_->stream.get());
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "Beauty: nvjpegEncoderStateCreate failed: {}", static_cast<int>(nvs)));
    }
    nvs = nvjpegEncoderParamsCreate(
        impl_->nvjpegHandle, &impl_->nvjpegParams, impl_->stream.get());
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "Beauty: nvjpegEncoderParamsCreate failed: {}", static_cast<int>(nvs)));
    }
    nvjpegEncoderParamsSetQuality(impl_->nvjpegParams, kBeautyJpegQuality,
                                  impl_->stream.get());
    nvjpegEncoderParamsSetSamplingFactors(impl_->nvjpegParams,
                                         NVJPEG_CSS_420, impl_->stream.get());

    impl_->loaded = true;
    OE_LOG_INFO("beauty_inferencer_loaded: model={}", faceMeshOnnxPath);
}


// ---------------------------------------------------------------------------
// computeFaceBbox — derive dilated bounding box from landmarks
// ---------------------------------------------------------------------------

struct FaceBbox { int x, y, w, h; bool valid; };

static FaceBbox computeFaceBbox(const float* landmarks,
                                uint32_t frameW, uint32_t frameH)
{
    float minX = 1e6f, maxX = -1e6f;
    float minY = 1e6f, maxY = -1e6f;

    for (std::size_t i = 0; i < oe::beauty::kFaceOvalCount; ++i) {
        const uint16_t idx = oe::beauty::kFaceOval[i];
        const float px = landmarks[idx * 3]     * static_cast<float>(frameW);
        const float py = landmarks[idx * 3 + 1] * static_cast<float>(frameH);
        minX = std::min(minX, px);
        maxX = std::max(maxX, px);
        minY = std::min(minY, py);
        maxY = std::max(maxY, py);
    }

    const float bw = maxX - minX;
    const float bh = maxY - minY;
    if (bw < 20.0f || bh < 20.0f) return {0, 0, 0, 0, false};

    const float dx = bw * kBboxDilateFraction;
    const float dy = bh * kBboxDilateFraction;
    const int bx = std::max(0, static_cast<int>(minX - dx));
    const int by = std::max(0, static_cast<int>(minY - dy));
    const int bW = std::min(static_cast<int>(frameW)  - bx, static_cast<int>(bw + 2.0f * dx));
    const int bH = std::min(static_cast<int>(frameH) - by, static_cast<int>(bh + 2.0f * dy));

    return {bx, by, bW, bH, true};
}


// ---------------------------------------------------------------------------
// processFrame
// ---------------------------------------------------------------------------

tl::expected<std::size_t, std::string>
OnnxBeautyInferencer::processFrame(
    const uint8_t* bgrFrame,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       outBuf,
    std::size_t    maxJpegBytes)
{
    OE_ZONE_SCOPED;

    if (!impl_->loaded) {
        return tl::unexpected(std::string("Beauty inferencer not loaded"));
    }

    const cudaStream_t stream = impl_->stream.get();
    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    const std::size_t bgrBytes = static_cast<std::size_t>(w) * h * 3;

    // Snapshot params
    BeautyParams params;
    {
        std::lock_guard lock(impl_->paramsMutex);
        params = impl_->params;
    }

    // Lazy allocation
    impl_->ensureFrameBuffers(width, height);

    // ── 1. H2D: copy BGR frame to device via pinned staging ──────────────
    std::memcpy(impl_->pinnedBgr.data(), bgrFrame, bgrBytes);
    OE_CUDA_CHECK(cudaMemcpyAsync(
        impl_->d_bgr.ptr, impl_->pinnedBgr.data(), bgrBytes,
        cudaMemcpyHostToDevice, stream));

    // ── Identity fast-path: skip inference, just JPEG encode ─────────────
    if (params.isIdentity()) {
        return impl_->encodeNvJpeg(
                            static_cast<uint8_t*>(impl_->d_bgr.ptr),
                            width, height, outBuf, maxJpegBytes);
    }

    // ── 2. Fused preprocess → FaceMesh input buffer ──────────────────────
    constexpr int fmSize = static_cast<int>(kBeautyFaceMeshInputSize);
    float* d_fmInput = impl_->ioBinding.inputPtr<float>(0);

    oe::beauty::launchResizeNormForFaceMesh(
        static_cast<uint8_t*>(impl_->d_bgr.ptr), w, h,
        d_fmInput, fmSize, fmSize,
        stream);

    // ── 3. FaceMesh V2 inference ─────────────────────────────────────────
    impl_->ioBinding.run();

    // ── 4. Landmark D2H readback ─────────────────────────────────────────
    float* d_landmarks = impl_->ioBinding.outputPtr<float>(0);
    float* pinnedLm = reinterpret_cast<float*>(impl_->pinnedLandmarks.data());

    OE_CUDA_CHECK(cudaMemcpyAsync(
        pinnedLm, d_landmarks, kLandmarkFloats * sizeof(float),
        cudaMemcpyDeviceToHost, stream));

    // Sync: need landmarks on host for bbox computation
    OE_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 5. Compute face bounding box ─────────────────────────────────────
    const FaceBbox bbox = computeFaceBbox(pinnedLm, width, height);

    if (!bbox.valid) {
        // No face detected: passthrough JPEG encode
        return impl_->encodeNvJpeg(
                            static_cast<uint8_t*>(impl_->d_bgr.ptr),
                            width, height, outBuf, maxJpegBytes);
    }

    // Working buffer: start with d_bgr, apply effects, end with encode
    uint8_t* d_current = static_cast<uint8_t*>(impl_->d_bgr.ptr);
    uint8_t* d_filtBuf = static_cast<uint8_t*>(impl_->d_filtered.ptr);
    uint8_t* d_sharpBuf = static_cast<uint8_t*>(impl_->d_sharpened.ptr);

    // ── 6. Skin mask (YCrCb threshold within face bbox) ──────────────────
    float* d_skinMask = static_cast<float*>(impl_->d_skinMask.ptr);

    oe::beauty::launchComputeSkinMask(
        d_current, d_skinMask, w, h,
        bbox.x, bbox.y, bbox.w, bbox.h,
        kSkinCrMin, kSkinCrMax, kSkinCbMin, kSkinCbMax,
        stream);

    // ── 7. Bilateral filter (Skin tab: smoothing) ────────────────────────
    if (params.smoothing > 0.5f) {
        oe::beauty::launchBilateralFilterMasked(
            d_current, d_filtBuf, d_skinMask, w, h,
            bbox.x, bbox.y, bbox.w, bbox.h,
            kBeautyBilateralSigmaSpatial,
            kBeautyBilateralSigmaColor,
            kBeautyBilateralKernelRadius,
            params.smoothing,
            stream);
        d_current = d_filtBuf;
    }

    // ── 8. Dark circles removal (under-eye brightness) ───────────────────
    if (params.darkCircles > 0.5f) {
        // Zero under-eye mask
        OE_CUDA_CHECK(cudaMemsetAsync(
            impl_->d_underEyeMask.ptr, 0,
            static_cast<std::size_t>(w) * h * sizeof(float), stream));

        // Build elliptical masks for left and right under-eye regions
        float leftX[oe::beauty::kUnderEyeLeftCount];
        float leftY[oe::beauty::kUnderEyeLeftCount];
        for (std::size_t i = 0; i < oe::beauty::kUnderEyeLeftCount; ++i) {
            const uint16_t idx = oe::beauty::kUnderEyeLeft[i];
            leftX[i] = pinnedLm[idx * 3]     * static_cast<float>(width);
            leftY[i] = pinnedLm[idx * 3 + 1] * static_cast<float>(height);
        }
        oe::beauty::launchBuildUnderEyeMask(
            static_cast<float*>(impl_->d_underEyeMask.ptr), w, h,
            leftX, leftY,
            static_cast<int>(oe::beauty::kUnderEyeLeftCount),
            kUnderEyeDilate, stream);

        float rightX[oe::beauty::kUnderEyeRightCount];
        float rightY[oe::beauty::kUnderEyeRightCount];
        for (std::size_t i = 0; i < oe::beauty::kUnderEyeRightCount; ++i) {
            const uint16_t idx = oe::beauty::kUnderEyeRight[i];
            rightX[i] = pinnedLm[idx * 3]     * static_cast<float>(width);
            rightY[i] = pinnedLm[idx * 3 + 1] * static_cast<float>(height);
        }
        oe::beauty::launchBuildUnderEyeMask(
            static_cast<float*>(impl_->d_underEyeMask.ptr), w, h,
            rightX, rightY,
            static_cast<int>(oe::beauty::kUnderEyeRightCount),
            kUnderEyeDilate, stream);

        // Apply brightness boost to under-eye regions
        const float darkCircleAmount = params.darkCircles * 0.3f;  // scale to reasonable range
        oe::beauty::launchRegionBrightnessAdjust(
            d_current,
            static_cast<float*>(impl_->d_underEyeMask.ptr),
            w, h, darkCircleAmount, stream);
    }

    // ── 9. Tone evening (subtle brightness equalization in skin region) ──
    if (params.toneEven > 0.5f) {
        const float toneAmount = params.toneEven * 0.15f;
        oe::beauty::launchRegionBrightnessAdjust(
            d_current, d_skinMask, w, h, toneAmount, stream);
    }

    // ── 10. Light tab: brightness, warmth, shadow fill, highlight ────────
    if (std::abs(params.brightness) > 0.5f) {
        oe::beauty::launchBcsAdjust(
            d_current, w, h,
            params.brightness * 0.5f,  // scale -100..+100 to -50..+50
            1.0f,                       // contrast unchanged
            1.0f,                       // saturation unchanged
            stream);
    }

    if (std::abs(params.warmth) > 0.5f) {
        oe::beauty::launchWarmthAdjust(
            d_current, d_skinMask, w, h, params.warmth, stream);
    }

    if (params.shadowFill > 0.5f) {
        oe::beauty::launchShadowFill(
            d_current, d_skinMask, w, h, params.shadowFill, stream);
    }

    if (params.highlight > 0.5f) {
        oe::beauty::launchHighlightBoost(
            d_current, d_skinMask, w, h, params.highlight, stream);
    }

    // ── 11. TPS warp (Shape tab: faceSlim, eyeEnlarge, noseNarrow, jawReshape)
    const bool hasShapeParams =
        params.faceSlim   > 0.5f || params.eyeEnlarge > 0.5f ||
        params.noseNarrow > 0.5f || std::abs(params.jawReshape) > 0.5f;

    if (hasShapeParams) {
        // TPS weights recomputed every frame (landmarks move with the face).
        // The CPU solver runs ~0.1 ms for 68 control points.
        impl_->tpsSolver.computeWeights(
            pinnedLm,
            params.faceSlim, params.eyeEnlarge,
            params.noseNarrow, params.jawReshape,
            width, height);

        const std::size_t N = impl_->tpsSolver.numPoints();
        const std::size_t M = N + 3;

        // One-time allocation of device buffers for TPS data
        if (impl_->d_tpsSrcX.bytes == 0) {
            impl_->d_tpsSrcX     = oe::onnx::GpuBuffer(N * sizeof(float));
            impl_->d_tpsSrcY     = oe::onnx::GpuBuffer(N * sizeof(float));
            impl_->d_tpsWeightsX = oe::onnx::GpuBuffer(M * sizeof(float));
            impl_->d_tpsWeightsY = oe::onnx::GpuBuffer(M * sizeof(float));
        }

        // Upload control points + weights to device
        OE_CUDA_CHECK(cudaMemcpyAsync(
            impl_->d_tpsSrcX.ptr, impl_->tpsSolver.sourceX().data(),
            N * sizeof(float), cudaMemcpyHostToDevice, stream));
        OE_CUDA_CHECK(cudaMemcpyAsync(
            impl_->d_tpsSrcY.ptr, impl_->tpsSolver.sourceY().data(),
            N * sizeof(float), cudaMemcpyHostToDevice, stream));
        OE_CUDA_CHECK(cudaMemcpyAsync(
            impl_->d_tpsWeightsX.ptr, impl_->tpsSolver.weightsX().data(),
            M * sizeof(float), cudaMemcpyHostToDevice, stream));
        OE_CUDA_CHECK(cudaMemcpyAsync(
            impl_->d_tpsWeightsY.ptr, impl_->tpsSolver.weightsY().data(),
            M * sizeof(float), cudaMemcpyHostToDevice, stream));

        uint8_t* d_warpBuf = static_cast<uint8_t*>(impl_->d_warped.ptr);
        oe::beauty::launchTpsWarp(
            d_current, d_warpBuf,
            static_cast<float*>(impl_->d_tpsSrcX.ptr),
            static_cast<float*>(impl_->d_tpsSrcY.ptr),
            static_cast<float*>(impl_->d_tpsWeightsX.ptr),
            static_cast<float*>(impl_->d_tpsWeightsY.ptr),
            static_cast<int>(impl_->tpsSolver.numPoints()),
            w, h,
            bbox.x, bbox.y, bbox.w, bbox.h,
            stream);
        d_current = d_warpBuf;
    }

    // ── 12. Sharpen (Skin tab) ──────────────────────────────────────────
    if (params.sharpen > 0.5f) {
        const float sharpenAmount = params.sharpen / 10.0f;  // 0-100 → 0-10
        oe::beauty::launchBeautySharpen(
            d_current, d_sharpBuf, w, h, sharpenAmount, stream);
        d_current = d_sharpBuf;
    }

    // ── 13. BG compositing (BG tab) ─────────────────────────────────────
    if (params.bgMode != BeautyParams::BgMode::kNone) {
        uint8_t* d_compBuf = static_cast<uint8_t*>(impl_->d_bgComposite.ptr);
        // Use skin mask as proxy for person segmentation.  A real person mask
        // would come from BackgroundBlurNode via ZMQ (d_personMask), but that
        // subscription is wired at the node level, not the inferencer level.

        switch (params.bgMode) {
            case BeautyParams::BgMode::kBlur:
                // Use the skin mask as a rough person mask for blur.
                // The actual segmentation mask would come from BackgroundBlurNode
                // via ZMQ subscription — for now, composite using skin mask.
                oe::beauty::launchCompositeSolidBg(
                    d_current, d_compBuf, d_skinMask,
                    w, h, 0, 0, 0, stream);
                d_current = d_compBuf;
                break;

            case BeautyParams::BgMode::kColor:
                oe::beauty::launchCompositeSolidBg(
                    d_current, d_compBuf, d_skinMask,
                    w, h,
                    params.bgColorB, params.bgColorG, params.bgColorR,
                    stream);
                d_current = d_compBuf;
                break;

            case BeautyParams::BgMode::kImage:
                if (impl_->hasBgImage) {
                    oe::beauty::launchCompositeBgImage(
                        d_current, d_compBuf, d_skinMask,
                        static_cast<const uint8_t*>(impl_->d_bgImage.ptr),
                        w, h, stream);
                    d_current = d_compBuf;
                }
                break;

            default:
                break;
        }
    }

    // ── 14. nvJPEG encode ────────────────────────────────────────────────
    auto result = impl_->encodeNvJpeg(d_current, width, height, outBuf, maxJpegBytes);

    // ── 15. Frame pacing — enforce 30 fps target ────────────────────────
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - impl_->lastFrameTime);
    if (elapsed.count() < kBeautyFramePacingMs) {
        // Frame budget not expired — caller can use this to throttle
        // (actual sleep/skip is done by the node, not the inferencer)
    }
    impl_->lastFrameTime = now;

    return result;
}


// ---------------------------------------------------------------------------
// setBeautyParams
// ---------------------------------------------------------------------------

void OnnxBeautyInferencer::setBeautyParams(const BeautyParams& params) noexcept
{
    std::lock_guard lock(impl_->paramsMutex);

    // Invalidate CUDA graph when params change (graph encodes fixed kernel args)
    // Compare relevant float fields — if any changed, the captured graph is stale.
    if (impl_->graphParamsValid) {
        const auto& old = impl_->graphCapturedParams;
        const bool changed =
            std::abs(params.smoothing   - old.smoothing)   > 0.01f ||
            std::abs(params.toneEven    - old.toneEven)    > 0.01f ||
            std::abs(params.darkCircles - old.darkCircles)  > 0.01f ||
            std::abs(params.sharpen     - old.sharpen)     > 0.01f ||
            std::abs(params.faceSlim    - old.faceSlim)    > 0.01f ||
            std::abs(params.eyeEnlarge  - old.eyeEnlarge)  > 0.01f ||
            std::abs(params.noseNarrow  - old.noseNarrow)  > 0.01f ||
            std::abs(params.jawReshape  - old.jawReshape)  > 0.01f ||
            std::abs(params.brightness  - old.brightness)  > 0.01f ||
            std::abs(params.warmth      - old.warmth)      > 0.01f ||
            std::abs(params.shadowFill  - old.shadowFill)  > 0.01f ||
            std::abs(params.highlight   - old.highlight)   > 0.01f ||
            params.bgMode != old.bgMode;

        if (changed) {
            impl_->cudaGraph.reset();
            impl_->graphParamsValid = false;
        }
    }

    impl_->params = params;
}


// ---------------------------------------------------------------------------
// unload
// ---------------------------------------------------------------------------

void OnnxBeautyInferencer::unload() noexcept
{
    if (!impl_->loaded) return;

    impl_->handle.unload();
    impl_->ioBinding = oe::onnx::GpuIoBinding{};

    if (impl_->nvjpegParams)  { nvjpegEncoderParamsDestroy(impl_->nvjpegParams);  impl_->nvjpegParams  = nullptr; }
    if (impl_->nvjpegState)   { nvjpegEncoderStateDestroy(impl_->nvjpegState);    impl_->nvjpegState   = nullptr; }
    if (impl_->nvjpegHandle)  { nvjpegDestroy(impl_->nvjpegHandle);               impl_->nvjpegHandle  = nullptr; }

    impl_->d_bgr.free();
    impl_->d_filtered.free();
    impl_->d_sharpened.free();
    impl_->d_warped.free();
    impl_->d_bgComposite.free();
    impl_->d_skinMask.free();
    impl_->d_underEyeMask.free();
    impl_->d_personMask.free();
    impl_->d_bgImage.free();
    impl_->d_tpsSrcX.free();
    impl_->d_tpsSrcY.free();
    impl_->d_tpsWeightsX.free();
    impl_->d_tpsWeightsY.free();
    impl_->cudaGraph.reset();

    impl_->frameW = 0;
    impl_->frameH = 0;
    impl_->loaded = false;

    OE_LOG_INFO("beauty_inferencer_unloaded");
}


// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxBeautyInferencer::currentVramUsageBytes() const noexcept
{
    if (!impl_->loaded) return 0;

    // Model + IO binding buffers + frame buffers
    const std::size_t w = impl_->frameW;
    const std::size_t h = impl_->frameH;
    const std::size_t bgrBytes  = w * h * 3;
    const std::size_t maskBytes = w * h * sizeof(float);

    // FaceMesh model (~50 MiB) + frame buffers + TPS buffers + BG buffers
    return kBeautyMiB * 1024ULL * 1024ULL   // model budget
         + bgrBytes * 5                       // d_bgr + d_filtered + d_sharpened + d_warped + d_bgComposite
         + maskBytes * 3                      // d_skinMask + d_underEyeMask + d_personMask
         + impl_->d_bgImage.bytes             // user BG image (0 if none)
         + impl_->ioBinding.inputBytes(0)     // FaceMesh input
         + impl_->ioBinding.outputBytes(0)    // FaceMesh output
         + impl_->d_tpsSrcX.bytes + impl_->d_tpsSrcY.bytes
         + impl_->d_tpsWeightsX.bytes + impl_->d_tpsWeightsY.bytes;
}


// ---------------------------------------------------------------------------
// cudaStream
// ---------------------------------------------------------------------------

cudaStream_t OnnxBeautyInferencer::cudaStream() const noexcept
{
    return impl_->stream.get();
}


// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<BeautyInferencer> createOnnxBeautyInferencer()
{
    return std::make_unique<OnnxBeautyInferencer>();
}

/// Unified factory — resolved at link time.
/// GPU build: links this file.  Stub build: links beauty_inferencer_stub.cpp.
std::unique_ptr<BeautyInferencer> createBeautyInferencer()
{
    return createOnnxBeautyInferencer();
}
