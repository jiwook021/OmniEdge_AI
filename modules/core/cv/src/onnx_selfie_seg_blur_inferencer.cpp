#include "cv/onnx_selfie_seg_blur_inferencer.hpp"

#include "common/hf_model_fetcher.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/oe_onnx_io_binding.hpp"
#include "common/oe_tracy.hpp"
#include "common/onnx_session_handle.hpp"
#include "cv/basicvsrpp_preprocess.hpp"
#include "gpu/cuda_priority.hpp"
#include "gpu/cuda_stream.hpp"
#include "gpu/gpu_compositor.hpp"
#include "gpu/oe_cuda_check.hpp"
#include "gpu/pinned_buffer.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>


// ---------------------------------------------------------------------------
// Constants — MediaPipe Selfie Segmentation (binary variant)
//
// Input  : "pixel_values" [1, 3, 256, 256] fp32, RGB [0,1] (rescale 1/255).
// Output : "alphas"       [1, 1, 256, 256] fp32, foreground probability.
// ---------------------------------------------------------------------------
namespace {

constexpr int   kSelfieSegInputSize   = 256;
constexpr int   kSelfieSegMaskSize    = 256;
constexpr int   kSelfieSegInputFloats = 3 * kSelfieSegInputSize * kSelfieSegInputSize;
constexpr int   kSelfieSegMaskFloats  = kSelfieSegMaskSize * kSelfieSegMaskSize;

constexpr const char* kSelfieSegHfRepo = "onnx-community/mediapipe_selfie_segmentation";
constexpr const char* kSelfieSegHfFile = "onnx/model.onnx";

constexpr const char* kSelfieSegInputName  = "pixel_values";
constexpr const char* kSelfieSegOutputName = "alphas";

constexpr std::size_t kMaxJpegPinnedBytes = 2 * 1024 * 1024;

constexpr int   kDefaultBlurKernelRadius = 15;
constexpr float kDefaultBlurSigma        = 8.0f;

constexpr int kParamLogFramesMax = 3;  ///< first-N-frames channel-order verification

} // anonymous namespace


// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct OnnxSelfieSegBlurInferencer::Impl {
    oe::onnx::SessionHandle handle{"MediaPipeSelfieSeg"};

    // CUDA pipeline
    CudaStream               stream;
    PinnedStagingBuffer      pinnedBgr;
    PinnedStagingBuffer      pinnedJpeg;

    // IO Binding for zero-copy ONNX inference.
    oe::onnx::GpuIoBinding   ioBinding;

    // Full-resolution BGR24 work buffers owned as GpuMats so the compositor
    // can consume them directly (upscaleAndComposite / applyIspBcs /
    // applyIspSharpen all take cv::cuda::GpuMat). Allocated in
    // ensureFrameBuffers() and re-allocated on resolution change.
    cv::cuda::GpuMat         original;       ///< H2D BGR24 source
    cv::cuda::GpuMat         blurScratch;    ///< separable blur horizontal pass
    cv::cuda::GpuMat         blurred;        ///< separable blur vertical result
    cv::cuda::GpuMat         composited;     ///< mask-composite output
    cv::cuda::GpuMat         postIsp;        ///< after BCS+sharpen (final)

    // CUDA graph for steady-state replay of composite → ISP BCS → sharpen.
    CompositorGraph          cgraph;
    IspParams                graphCapturedIsp;
    int                      graphCapturedKernelRadius = 0;
    float                    graphCapturedSigma        = 0.0f;
    bool                     graphParamsValid          = false;

    // nvJPEG state
    nvjpegHandle_t           nvjpegHandle = nullptr;
    nvjpegEncoderState_t     nvjpegState  = nullptr;
    nvjpegEncoderParams_t    nvjpegParams = nullptr;

    // Frame dimensions (lazy-init)
    uint32_t frameW = 0;
    uint32_t frameH = 0;

    // Runtime-tunable parameters (thread-safe snapshot)
    std::mutex               paramsMutex;
    IspParams                params;
    int                      kernelRadius = kDefaultBlurKernelRadius;
    float                    sigma        = kDefaultBlurSigma;

    // First-N-frames diagnostics (mask_mean log for channel-order check).
    int                      maskMeanFramesLogged = 0;

    bool                     loaded = false;

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    ~Impl()
    {
        if (nvjpegParams) nvjpegEncoderParamsDestroy(nvjpegParams);
        if (nvjpegState)  nvjpegEncoderStateDestroy(nvjpegState);
        if (nvjpegHandle) nvjpegDestroy(nvjpegHandle);
    }

    void ensureFrameBuffers(uint32_t w, uint32_t h)
    {
        if (frameW == w && frameH == h) return;

        const int iw = static_cast<int>(w);
        const int ih = static_cast<int>(h);
        const std::size_t bgrBytes = static_cast<std::size_t>(w) * h * 3;

        original    .create(ih, iw, CV_8UC3);
        blurScratch .create(ih, iw, CV_8UC3);
        blurred     .create(ih, iw, CV_8UC3);
        composited  .create(ih, iw, CV_8UC3);
        postIsp     .create(ih, iw, CV_8UC3);

        pinnedBgr = PinnedStagingBuffer(bgrBytes);

        cgraph.invalidate();
        graphParamsValid = false;

        frameW = w;
        frameH = h;
        OE_LOG_DEBUG("selfie_seg_buffers: {}x{}", w, h);
    }

    /// Compare captured graph state against current params. If anything the
    /// graph baked in has changed, invalidate so the next frame re-captures.
    void syncGraphCaptureState(const IspParams& currentIsp,
                               int              currentKernelRadius,
                               float            currentSigma)
    {
        if (!graphParamsValid) return;

        const auto& old = graphCapturedIsp;
        const bool ispChanged =
            std::abs(currentIsp.brightness - old.brightness) > 1e-3f ||
            std::abs(currentIsp.contrast   - old.contrast)   > 1e-3f ||
            std::abs(currentIsp.saturation - old.saturation) > 1e-3f ||
            std::abs(currentIsp.sharpness  - old.sharpness)  > 1e-3f;

        const bool blurChanged =
            currentKernelRadius != graphCapturedKernelRadius ||
            std::abs(currentSigma - graphCapturedSigma) > 1e-3f;

        if (ispChanged || blurChanged) {
            cgraph.invalidate();
            graphParamsValid = false;
        }
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    encodeNvJpeg(cv::cuda::GpuMat& src, uint32_t w, uint32_t h,
                 uint8_t* outBuf, std::size_t maxJpegBytes)
    {
        cudaStream_t s = stream.get();

        nvjpegImage_t nvImage{};
        nvImage.channel[0] = src.ptr<uint8_t>();
        nvImage.pitch[0]   = static_cast<unsigned int>(src.step);

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

    /// Optionally emit the first N frame mask means so we can verify
    /// channel order (BGR vs RGB) at runtime. Sync on stream first so the
    /// mask is readable from host.
    void logMaskMeanIfNeeded(float* d_mask, uint32_t maskW, uint32_t maskH)
    {
        if (maskMeanFramesLogged >= kParamLogFramesMax) return;

        const std::size_t n = static_cast<std::size_t>(maskW) * maskH;
        std::vector<float> hostMask(n);
        const cudaError_t err = cudaMemcpyAsync(
            hostMask.data(), d_mask, n * sizeof(float),
            cudaMemcpyDeviceToHost, stream.get());
        if (err != cudaSuccess) return;
        if (cudaStreamSynchronize(stream.get()) != cudaSuccess) return;

        double sum = 0.0;
        for (float v : hostMask) sum += static_cast<double>(v);
        const double mean = sum / static_cast<double>(n);
        OE_LOG_INFO("selfie_seg_mask_mean frame={} mean={:.4f} (expect ~0.15–0.55 on person-present frames)",
                    maskMeanFramesLogged, mean);
        ++maskMeanFramesLogged;
    }

    /// Execute the core GPU pipeline (H2D → preprocess → inference → blur →
    /// fused composite → ISP). Leaves the final 1080p frame in `postIsp`.
    [[nodiscard]] tl::expected<void, std::string>
    runPipeline(const uint8_t* bgrFrame, uint32_t width, uint32_t height)
    {
        cudaStream_t s = stream.get();
        const std::size_t bgrBytes = static_cast<std::size_t>(width) * height * 3;

        // Snapshot runtime params.
        IspParams ispSnap;
        int       kernelRadiusSnap;
        float     sigmaSnap;
        {
            std::lock_guard lock(paramsMutex);
            ispSnap          = params;
            kernelRadiusSnap = kernelRadius;
            sigmaSnap        = sigma;
        }

        ensureFrameBuffers(width, height);
        syncGraphCaptureState(ispSnap, kernelRadiusSnap, sigmaSnap);

        // 1. H2D: pinned staging → `original` GpuMat (pitch-aware).
        std::memcpy(pinnedBgr.data(), bgrFrame, bgrBytes);
        OE_CUDA_CHECK(cudaMemcpy2DAsync(
            original.ptr<uint8_t>(), original.step,
            pinnedBgr.data(),         static_cast<std::size_t>(width) * 3,
            static_cast<std::size_t>(width) * 3, height,
            cudaMemcpyHostToDevice, s));

        // 2. Preprocess: fused resize + BGR→RGB + /255 → [1,3,256,256].
        float* d_input = ioBinding.inputPtr<float>(0);
        oe::cv::launchResizeNormFP32(
            original.ptr<uint8_t>(),
            static_cast<int>(width), static_cast<int>(height),
            d_input,
            kSelfieSegInputSize, kSelfieSegInputSize,
            /*frameIndex=*/0, /*totalFrames=*/1,
            s);

        // 3. Segmentation inference.
        try {
            ioBinding.run();
        } catch (const Ort::Exception& ex) {
            return tl::unexpected(std::format("SelfieSeg Run: {}", ex.what()));
        }
        float* d_mask = ioBinding.outputPtr<float>(0);

        // First-N-frames diagnostic — verifies channel order isn't inverted.
        logMaskMeanIfNeeded(d_mask, kSelfieSegMaskSize, kSelfieSegMaskSize);

        // 4. Gaussian blur at full resolution (separable 2-pass).
        launchGaussianBlur(
            original, blurScratch, blurred,
            kernelRadiusSnap, sigmaSnap, s);

        // 5. Fused upscale-mask + composite.
        upscaleAndComposite(
            original, blurred,
            d_mask, kSelfieSegMaskSize, kSelfieSegMaskSize,
            composited, s);

        // 6. ISP: in-place BCS, out-of-place sharpen. If no ISP change,
        //    avoid the extra copy by aliasing `postIsp` to `composited`
        //    (GpuMat assignment is refcount-only).
        if (!ispSnap.isIdentity()) {
            applyIspBcs(composited,
                        ispSnap.brightness, ispSnap.contrast, ispSnap.saturation,
                        s);
            applyIspSharpen(composited, postIsp, ispSnap.sharpness, s);
        } else {
            postIsp = composited;
        }

        if (!graphParamsValid) {
            graphCapturedIsp          = ispSnap;
            graphCapturedKernelRadius = kernelRadiusSnap;
            graphCapturedSigma        = sigmaSnap;
            graphParamsValid          = true;
        }

        return {};
    }
};


// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
OnnxSelfieSegBlurInferencer::OnnxSelfieSegBlurInferencer()
    : impl_(std::make_unique<Impl>())
{}

OnnxSelfieSegBlurInferencer::~OnnxSelfieSegBlurInferencer()
{
    unload();
}


// ---------------------------------------------------------------------------
// loadEngine
// ---------------------------------------------------------------------------
void OnnxSelfieSegBlurInferencer::loadEngine(const std::string& modelPath,
                                             uint32_t           inputWidth,
                                             uint32_t           inputHeight)
{
    OE_ZONE_SCOPED;

    // The ONNX model is fixed 256×256; inputWidth/inputHeight describe the
    // full-resolution source frame and are informational here. Actual frame
    // buffers size lazily on the first processFrame() call.
    OE_LOG_INFO("selfie_seg_load: source_hint={}x{}, model_path={}",
                inputWidth, inputHeight, modelPath);

    // 1. CUDA stream — CV tier priority.
    auto streamResult = CudaStream::createClamped(kCudaPriorityCv);
    if (!streamResult) {
        throw std::runtime_error(std::format(
            "SelfieSeg: CUDA stream failed: {}", streamResult.error()));
    }
    impl_->stream = std::move(*streamResult);

    // 2. Auto-fetch from HF on first run.
    std::filesystem::path resolved = modelPath;
    if (!std::filesystem::exists(resolved)) {
        const auto cacheDir = resolved.has_parent_path()
            ? resolved.parent_path()
            : std::filesystem::path("model.cache/bg_blur");
        OE_LOG_INFO("selfie_seg_fetch_start: repo={}, file={}, cache={}",
                    kSelfieSegHfRepo, kSelfieSegHfFile, cacheDir.string());
        auto fetched = fetchHfModel(kSelfieSegHfRepo, kSelfieSegHfFile, cacheDir);
        if (!fetched) {
            throw std::runtime_error(std::format(
                "Selfie Seg fetch failed (model missing and HF download unavailable): {}",
                fetched.error()));
        }
        resolved = *fetched;
        OE_LOG_INFO("selfie_seg_fetch_complete: path={}", resolved.string());
    }

    // 3. ONNX session — CUDA EP, CUDA-graph enabled (fixed-shape model).
    auto loadResult = impl_->handle.load(resolved.string(), oe::onnx::SessionConfig{
        .useTRT            = false,
        .gpuMemLimitMiB    = kBgBlurMiB,
        .enableCudaGraph   = true,
        .exhaustiveCudnn   = true,
        .maxCudnnWorkspace = true,
    });
    if (!loadResult) {
        throw std::runtime_error(std::format(
            "SelfieSeg session load: {}", loadResult.error()));
    }

    // 4. IO binding: [1,3,256,256] fp32 input, [1,1,256,256] fp32 output.
    impl_->ioBinding = oe::onnx::GpuIoBinding(*impl_->handle.session);

    impl_->ioBinding.allocateInput(
        kSelfieSegInputName,
        {1, 3, kSelfieSegInputSize, kSelfieSegInputSize},
        sizeof(float));

    impl_->ioBinding.allocateOutput(
        kSelfieSegOutputName,
        {1, 1, kSelfieSegMaskSize, kSelfieSegMaskSize},
        sizeof(float));

    // 5. Pinned JPEG staging (D2H retrieve buffer already owned by nvJPEG).
    impl_->pinnedJpeg = PinnedStagingBuffer(kMaxJpegPinnedBytes);

    // 6. nvJPEG encoder setup.
    nvjpegStatus_t nvs = nvjpegCreateSimple(&impl_->nvjpegHandle);
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "SelfieSeg: nvjpegCreateSimple failed: {}", static_cast<int>(nvs)));
    }
    nvs = nvjpegEncoderStateCreate(
        impl_->nvjpegHandle, &impl_->nvjpegState, impl_->stream.get());
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "SelfieSeg: nvjpegEncoderStateCreate failed: {}", static_cast<int>(nvs)));
    }
    nvs = nvjpegEncoderParamsCreate(
        impl_->nvjpegHandle, &impl_->nvjpegParams, impl_->stream.get());
    if (nvs != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::format(
            "SelfieSeg: nvjpegEncoderParamsCreate failed: {}", static_cast<int>(nvs)));
    }
    nvjpegEncoderParamsSetQuality(impl_->nvjpegParams, 85, impl_->stream.get());
    nvjpegEncoderParamsSetSamplingFactors(
        impl_->nvjpegParams, NVJPEG_CSS_420, impl_->stream.get());

    impl_->loaded = true;
    OE_LOG_INFO("selfie_seg_session_ready: inputs={}, outputs={}",
                impl_->handle.inputCount(), impl_->handle.outputCount());
}


// ---------------------------------------------------------------------------
// processFrame — returns JPEG bytes
// ---------------------------------------------------------------------------
tl::expected<std::size_t, std::string>
OnnxSelfieSegBlurInferencer::processFrame(
    const uint8_t* bgrFrame,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       outBuf,
    std::size_t    maxJpegBytes)
{
    OE_ZONE_SCOPED;

    if (!impl_->loaded) {
        return tl::unexpected(std::string("SelfieSeg inferencer not loaded"));
    }
    if (bgrFrame == nullptr || outBuf == nullptr) {
        return tl::unexpected(std::string("SelfieSeg: null buffer"));
    }

    if (auto r = impl_->runPipeline(bgrFrame, width, height); !r) {
        return tl::unexpected(std::move(r.error()));
    }

    return impl_->encodeNvJpeg(impl_->postIsp, width, height, outBuf, maxJpegBytes);
}


// ---------------------------------------------------------------------------
// processFrameGetBgr — returns packed BGR24 bytes (for pipeline chaining)
// ---------------------------------------------------------------------------
tl::expected<std::size_t, std::string>
OnnxSelfieSegBlurInferencer::processFrameGetBgr(
    const uint8_t* bgrFrame,
    uint32_t       width,
    uint32_t       height,
    uint8_t*       outBgrBuf,
    std::size_t    maxBgrBytes)
{
    OE_ZONE_SCOPED;

    if (!impl_->loaded) {
        return tl::unexpected(std::string("SelfieSeg inferencer not loaded"));
    }
    if (bgrFrame == nullptr || outBgrBuf == nullptr) {
        return tl::unexpected(std::string("SelfieSeg: null buffer"));
    }
    const std::size_t bgrBytes = static_cast<std::size_t>(width) * height * 3;
    if (bgrBytes > maxBgrBytes) {
        return tl::unexpected(std::format(
            "BGR output buffer too small: needed={} capacity={}", bgrBytes, maxBgrBytes));
    }

    if (auto r = impl_->runPipeline(bgrFrame, width, height); !r) {
        return tl::unexpected(std::move(r.error()));
    }

    // D2H copy of the final GpuMat to the caller's packed BGR24 buffer.
    cudaStream_t stream = impl_->stream.get();
    OE_CUDA_CHECK(cudaMemcpy2DAsync(
        outBgrBuf,                      static_cast<std::size_t>(width) * 3,
        impl_->postIsp.ptr<uint8_t>(),   impl_->postIsp.step,
        static_cast<std::size_t>(width) * 3, height,
        cudaMemcpyDeviceToHost, stream));
    OE_CUDA_CHECK(cudaStreamSynchronize(stream));

    return bgrBytes;
}


// ---------------------------------------------------------------------------
// setIspParams — thread-safe, invalidates CUDA graph if captured state changes
// ---------------------------------------------------------------------------
void OnnxSelfieSegBlurInferencer::setIspParams(const IspParams& params) noexcept
{
    std::lock_guard lock(impl_->paramsMutex);

    // Graph invalidation check happens in syncGraphCaptureState() at the
    // next frame; here we just snapshot the new params.
    impl_->params = params;
}


// ---------------------------------------------------------------------------
// unload
// ---------------------------------------------------------------------------
void OnnxSelfieSegBlurInferencer::unload() noexcept
{
    if (!impl_->loaded) return;

    impl_->handle.unload();
    impl_->ioBinding = oe::onnx::GpuIoBinding{};

    if (impl_->nvjpegParams)  { nvjpegEncoderParamsDestroy(impl_->nvjpegParams);  impl_->nvjpegParams  = nullptr; }
    if (impl_->nvjpegState)   { nvjpegEncoderStateDestroy(impl_->nvjpegState);    impl_->nvjpegState   = nullptr; }
    if (impl_->nvjpegHandle)  { nvjpegDestroy(impl_->nvjpegHandle);               impl_->nvjpegHandle  = nullptr; }

    impl_->original    .release();
    impl_->blurScratch .release();
    impl_->blurred     .release();
    impl_->composited  .release();
    impl_->postIsp     .release();
    impl_->cgraph.invalidate();

    impl_->frameW = 0;
    impl_->frameH = 0;
    impl_->graphParamsValid = false;
    impl_->maskMeanFramesLogged = 0;
    impl_->loaded = false;

    OE_LOG_INFO("selfie_seg_inferencer_unloaded");
}


// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------
std::size_t OnnxSelfieSegBlurInferencer::currentVramUsageBytes() const noexcept
{
    if (!impl_->loaded) return 0;

    const std::size_t w = impl_->frameW;
    const std::size_t h = impl_->frameH;
    const std::size_t bgrBytesPerMat = w * h * 3;

    // Model budget + 5 full-res BGR GpuMats + IO binding (input [1,3,256,256]
    // + output [1,1,256,256]).
    return kBgBlurMiB * 1024ULL * 1024ULL
         + bgrBytesPerMat * 5
         + impl_->ioBinding.inputBytes(0)
         + impl_->ioBinding.outputBytes(0);
}


// ---------------------------------------------------------------------------
// cudaStream
// ---------------------------------------------------------------------------
cudaStream_t OnnxSelfieSegBlurInferencer::cudaStream() const noexcept
{
    return impl_->stream.get();
}


// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------
std::unique_ptr<BlurInferencer> createOnnxSelfieSegBlurInferencer()
{
    return std::make_unique<OnnxSelfieSegBlurInferencer>();
}

/// Unified factory — linked by main_bg_blur.cpp in GPU builds; the CPU stub
/// build links a different TU that defines the same symbol.
std::unique_ptr<BlurInferencer> createBlurInferencer()
{
    return createOnnxSelfieSegBlurInferencer();
}
