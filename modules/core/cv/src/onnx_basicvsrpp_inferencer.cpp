#include "cv/onnx_basicvsrpp_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/oe_onnx_io_binding.hpp"
#include "common/onnx_session_handle.hpp"
#include "common/constants/memory_constants.hpp"
#include "common/constants/cv_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "cv/basicvsrpp_preprocess.hpp"
#include "gpu/cuda_stream.hpp"
#include "gpu/cuda_priority.hpp"
#include "gpu/oe_cuda_check.hpp"
#include "gpu/pinned_buffer.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <onnxruntime_cxx_api.h>

#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxBasicVsrppInferencer::Impl {
	oe::onnx::SessionHandle handle{"BasicVSR++"};

	// Model's fixed input resolution (queried from ONNX metadata at load time)
	uint32_t modelInputH = 0;
	uint32_t modelInputW = 0;

	// --- GPU pipeline resources ---
	CudaStream                  stream;           // non-default stream for all async work
	PinnedStagingBuffer         pinnedBgr;        // pinned host buffer for DMA H2D
	oe::onnx::GpuBuffer         d_bgr;            // device BGR24 staging (one frame at a time)
	oe::onnx::GpuIoBinding      ioBinding;        // GPU-resident IO binding for zero-copy inference

	// --- nvJPEG encoder resources ---
	nvjpegHandle_t              nvjpegHandle  = nullptr;
	nvjpegEncoderState_t        nvjpegState   = nullptr;
	nvjpegEncoderParams_t       nvjpegParams  = nullptr;
	oe::onnx::GpuBuffer         d_bgrOut;         // device buffer for NCHW→BGR conversion
	PinnedStagingBuffer         pinnedJpeg;       // pinned output for compressed JPEG D2H

	// CPU warmup buffer (used only during warmup before IO binding is ready)
	std::vector<float>          inputBuffer;

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
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OnnxBasicVsrppInferencer::OnnxBasicVsrppInferencer()
	: impl_(std::make_unique<Impl>())
{
}

OnnxBasicVsrppInferencer::~OnnxBasicVsrppInferencer()
{
	unloadModel();
}

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------

tl::expected<void, std::string> OnnxBasicVsrppInferencer::loadModel(
	const std::string& onnxModelPath)
{
	// enableCudaGraph must be false when useTRT is true: TRT EP partitions
	// some nodes, leaving CUDA EP without the full graph — capture fails.
	if (auto r = impl_->handle.load(onnxModelPath, oe::onnx::SessionConfig{
		.useTRT            = true,
		.gpuMemLimitMiB    = kBasicVsrppMiB,
		.enableCudaGraph   = false,
		.exhaustiveCudnn   = true,
		.maxCudnnWorkspace = true,
	}); !r) {
		return tl::unexpected(std::format("BasicVSR++ load: {}", r.error()));
	}

	// Query model's expected input shape [1, N, 3, H, W]
	{
		auto typeInfo   = impl_->handle.session->GetInputTypeInfo(0);
		auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
		auto shape      = tensorInfo.GetShape();
		if (shape.size() == 5 && shape[3] > 0 && shape[4] > 0) {
			impl_->modelInputH = static_cast<uint32_t>(shape[3]);
			impl_->modelInputW = static_cast<uint32_t>(shape[4]);
		} else {
			return tl::unexpected(std::string(
				"BasicVSR++ model input shape is not [1, N, 3, H, W]"));
		}
	}

	OE_LOG_INFO("basicvsrpp_model_loaded: path={}, input={}x{}",
	            onnxModelPath, impl_->modelInputW, impl_->modelInputH);

	// Warmup: trigger TRT engine compilation during init (not on first
	// real inference) to avoid the 1-3 min blocking that kills the watchdog.
	if (auto warmup = runWarmup(onnxModelPath); !warmup) {
		return tl::unexpected(warmup.error());
	}

	// --- Allocate GPU pipeline resources (post-warmup, session is final) ---

	// 1. CUDA stream at video-denoise priority
	auto streamResult = CudaStream::createClamped(kCudaPriorityVideoDenoise);
	if (!streamResult) {
		return tl::unexpected(std::format("BasicVSR++ stream: {}", streamResult.error()));
	}
	impl_->stream = std::move(*streamResult);

	// 2. Pinned host buffer for one BGR24 source frame (largest expected: 1920x1080x3)
	impl_->pinnedBgr = PinnedStagingBuffer(kMaxBgr24FrameBytes);

	// 3. Device buffer for one BGR24 frame (reused per-frame in the loop)
	impl_->d_bgr = oe::onnx::GpuBuffer(kMaxBgr24FrameBytes);

	// 4. GPU IO Binding — pre-allocate input [1,5,3,H,W] and output [1,5,3,outH,outW]
	{
		const uint32_t inH = impl_->modelInputH;
		const uint32_t inW = impl_->modelInputW;
		constexpr int64_t N = 5;  // temporal window

		const std::vector<int64_t> inputShape = {
			1, N, 3, static_cast<int64_t>(inH), static_cast<int64_t>(inW)};

		// Query output shape from model metadata
		auto outTypeInfo   = impl_->handle.session->GetOutputTypeInfo(0);
		auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
		auto outShape      = outTensorInfo.GetShape();

		impl_->ioBinding = oe::onnx::GpuIoBinding(*impl_->handle.session);
		impl_->ioBinding.allocateInput(
			impl_->handle.inputNamePtrs[0], inputShape, sizeof(float));
		impl_->ioBinding.allocateOutput(
			impl_->handle.outputNamePtrs[0], outShape, sizeof(float));

		OE_LOG_INFO("basicvsrpp_gpu_pipeline_ready: input=[1,{},3,{},{}] output=[{},{},{},{},{}]",
		            N, inH, inW,
		            outShape[0], outShape[1], outShape[2], outShape[3], outShape[4]);

		// 5. nvJPEG encoder — GPU-side JPEG encoding of the center frame
		const uint32_t outH = static_cast<uint32_t>(outShape[3]);
		const uint32_t outW = static_cast<uint32_t>(outShape[4]);
		const std::size_t outBgrBytes =
			static_cast<std::size_t>(outW) * outH * 3;

		// Device buffer for CHW→BGR conversion result
		impl_->d_bgrOut = oe::onnx::GpuBuffer(outBgrBytes);

		// Pinned buffer for compressed JPEG D2H (max ~500 KB for 1080p JPEG)
		constexpr std::size_t kMaxJpegPinnedBytes = 1024 * 1024;  // 1 MiB
		impl_->pinnedJpeg = PinnedStagingBuffer(kMaxJpegPinnedBytes);

		// Initialize nvJPEG
		nvjpegStatus_t nvStatus = nvjpegCreateSimple(&impl_->nvjpegHandle);
		if (nvStatus != NVJPEG_STATUS_SUCCESS) {
			return tl::unexpected(std::format(
				"nvjpegCreateSimple failed: {}", static_cast<int>(nvStatus)));
		}
		nvStatus = nvjpegEncoderStateCreate(
			impl_->nvjpegHandle, &impl_->nvjpegState, impl_->stream.get());
		if (nvStatus != NVJPEG_STATUS_SUCCESS) {
			return tl::unexpected(std::format(
				"nvjpegEncoderStateCreate failed: {}", static_cast<int>(nvStatus)));
		}
		nvStatus = nvjpegEncoderParamsCreate(
			impl_->nvjpegHandle, &impl_->nvjpegParams, impl_->stream.get());
		if (nvStatus != NVJPEG_STATUS_SUCCESS) {
			return tl::unexpected(std::format(
				"nvjpegEncoderParamsCreate failed: {}", static_cast<int>(nvStatus)));
		}
		nvjpegEncoderParamsSetQuality(impl_->nvjpegParams, kDefaultJpegQuality,
		                              impl_->stream.get());
		nvjpegEncoderParamsSetSamplingFactors(impl_->nvjpegParams,
		                                     NVJPEG_CSS_420, impl_->stream.get());
	}

	return {};
}

// ---------------------------------------------------------------------------
// runWarmup — trigger TRT compilation; fall back to CUDA-only on failure
// ---------------------------------------------------------------------------

tl::expected<void, std::string>
OnnxBasicVsrppInferencer::runWarmup(const std::string& onnxModelPath)
{
	OE_LOG_INFO("basicvsrpp_warmup_start: triggering TRT engine compilation "
	            "(may take 1-3 min on first run, cached afterwards)");

	const uint32_t wH = impl_->modelInputH;
	const uint32_t wW = impl_->modelInputW;
	const int64_t  wN = 5;
	const std::size_t wTotal =
		static_cast<std::size_t>(wN) * 3 * wW * wH;

	impl_->inputBuffer.assign(wTotal, 0.0f);

	const std::array<int64_t, 5> shape = {
		1, wN, 3, static_cast<int64_t>(wH), static_cast<int64_t>(wW)};
	const auto& memInfo = oe::onnx::cpuMemoryInfo();
	auto tensor = Ort::Value::CreateTensor<float>(
		memInfo, impl_->inputBuffer.data(), wTotal,
		shape.data(), shape.size());

	try {
		impl_->handle.session->Run(
			Ort::RunOptions{nullptr},
			impl_->handle.inputPtrs(), &tensor, 1,
			impl_->handle.outputPtrs(), impl_->handle.outputCount());
		OE_LOG_INFO("basicvsrpp_warmup_done: ep=TensorRT+CUDA");
		return {};
	} catch (const Ort::Exception& warmupEx) {
		OE_LOG_WARN("basicvsrpp_trt_warmup_failed: {}", warmupEx.what());
	}

	// TRT failed — recreate session with CUDA EP only (still GPU)
	OE_LOG_INFO("basicvsrpp_fallback_cuda: recreating session with CUDA EP only");
	if (auto r = impl_->handle.loadCudaOnly(onnxModelPath, kBasicVsrppMiB); !r) {
		return tl::unexpected(std::format(
			"BasicVSR++ both TRT and CUDA-only sessions failed: {}", r.error()));
	}

	auto tensor2 = Ort::Value::CreateTensor<float>(
		memInfo, impl_->inputBuffer.data(), wTotal,
		shape.data(), shape.size());
	try {
		impl_->handle.session->Run(
			Ort::RunOptions{nullptr},
			impl_->handle.inputPtrs(), &tensor2, 1,
			impl_->handle.outputPtrs(), impl_->handle.outputCount());
		OE_LOG_INFO("basicvsrpp_warmup_done: ep=CUDA (TRT fallback)");
	} catch (const Ort::Exception& cudaWarmupEx) {
		return tl::unexpected(std::format(
			"BasicVSR++ CUDA-only warmup failed: {}", cudaWarmupEx.what()));
	}

	return {};
}

// ---------------------------------------------------------------------------
// processFrames
// ---------------------------------------------------------------------------

tl::expected<std::size_t, std::string> OnnxBasicVsrppInferencer::processFrames(
	const uint8_t* const* bgrFrames,
	uint32_t frameCount,
	uint32_t width,
	uint32_t height,
	uint8_t* outJpegBuf,
	std::size_t maxJpegBytes)
{
	OE_ZONE_SCOPED;

	if (!impl_->handle.loaded) {
		return tl::unexpected(std::string("Model not loaded"));
	}
	if (!impl_->ioBinding.valid()) {
		return tl::unexpected(std::string("GPU IO binding not initialized"));
	}

	const uint32_t inW = impl_->modelInputW;
	const uint32_t inH = impl_->modelInputH;
	const std::size_t srcBgrBytes =
		static_cast<std::size_t>(width) * height * 3;
	cudaStream_t stream = impl_->stream.get();

	// --- GPU preprocessing: for each frame, H2D + fused resize/normalize ---
	float* d_nchwInput = impl_->ioBinding.inputPtr<float>(0);

	for (uint32_t f = 0; f < frameCount; ++f) {
		// Copy source BGR24 to pinned staging buffer (CPU memcpy, DMA-ready)
		std::memcpy(impl_->pinnedBgr.data(), bgrFrames[f], srcBgrBytes);

		// Async H2D: pinned host → device BGR buffer
		OE_CUDA_CHECK(cudaMemcpyAsync(
			impl_->d_bgr.ptr, impl_->pinnedBgr.data(),
			srcBgrBytes, cudaMemcpyHostToDevice, stream));

		// Fused GPU kernel: bilinear resize + BGR→RGB + normalize [0,1] → NCHW
		oe::cv::launchResizeNormFP32(
			static_cast<const uint8_t*>(impl_->d_bgr.ptr),
			static_cast<int>(width), static_cast<int>(height),
			d_nchwInput,
			static_cast<int>(inW), static_cast<int>(inH),
			static_cast<int>(f), static_cast<int>(frameCount),
			stream);
	}

	// --- ONNX inference via IO binding (zero implicit copies) ---
	try {
		impl_->ioBinding.run();
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format("ONNX inference error: {}", ex.what()));
	}

	// --- D2H: copy only the center frame from GPU output ---
	// Output shape is [1, N, 3, outH, outW] — query from pre-allocated binding
	auto outTypeInfo   = impl_->handle.session->GetOutputTypeInfo(0);
	auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
	auto outShape      = outTensorInfo.GetShape();
	const uint32_t outH = static_cast<uint32_t>(outShape[3]);
	const uint32_t outW = static_cast<uint32_t>(outShape[4]);
	const std::size_t outPixels =
		static_cast<std::size_t>(outW) * outH;
	const std::size_t centerFrameFloats = 3 * outPixels;

	const uint32_t centerIdx = frameCount / 2;
	const std::size_t centerOffsetFloats =
		static_cast<std::size_t>(centerIdx) * centerFrameFloats;

	// --- GPU: convert center frame NCHW RGB float → HWC BGR uint8 on device ---
	const float* d_output = impl_->ioBinding.outputPtr<float>(0);
	const float* d_centerNchw = d_output + centerOffsetFloats;
	auto* d_bgrOut = static_cast<uint8_t*>(impl_->d_bgrOut.ptr);

	oe::cv::launchChwToHwcBgr(
		d_centerNchw, d_bgrOut,
		static_cast<int>(outW), static_cast<int>(outH), stream);

	// --- GPU: nvJPEG encode from device BGR buffer ---
	nvjpegImage_t nvImage{};
	nvImage.channel[0] = d_bgrOut;
	nvImage.pitch[0]   = static_cast<unsigned int>(outW * 3);

	nvjpegStatus_t nvStatus = nvjpegEncodeImage(
		impl_->nvjpegHandle,
		impl_->nvjpegState,
		impl_->nvjpegParams,
		&nvImage,
		NVJPEG_INPUT_BGRI,
		static_cast<int>(outW),
		static_cast<int>(outH),
		stream);
	if (nvStatus != NVJPEG_STATUS_SUCCESS) {
		return tl::unexpected(std::format(
			"nvjpegEncodeImage failed: {}", static_cast<int>(nvStatus)));
	}

	// Retrieve compressed JPEG size
	std::size_t jpegLength = 0;
	nvStatus = nvjpegEncodeRetrieveBitstream(
		impl_->nvjpegHandle, impl_->nvjpegState,
		nullptr, &jpegLength, stream);
	if (nvStatus != NVJPEG_STATUS_SUCCESS) {
		return tl::unexpected(std::format(
			"nvjpegEncodeRetrieveBitstream (query) failed: {}", static_cast<int>(nvStatus)));
	}

	if (jpegLength > maxJpegBytes) {
		return tl::unexpected(std::string("JPEG output exceeded buffer"));
	}

	// Sync stream before retrieving bitstream to host
	OE_CUDA_CHECK(cudaStreamSynchronize(stream));

	// Retrieve compressed JPEG to output buffer
	nvStatus = nvjpegEncodeRetrieveBitstream(
		impl_->nvjpegHandle, impl_->nvjpegState,
		outJpegBuf, &jpegLength, nullptr);
	if (nvStatus != NVJPEG_STATUS_SUCCESS) {
		return tl::unexpected(std::format(
			"nvjpegEncodeRetrieveBitstream (copy) failed: {}", static_cast<int>(nvStatus)));
	}

	return jpegLength;
}

// ---------------------------------------------------------------------------
// unloadModel
// ---------------------------------------------------------------------------

void OnnxBasicVsrppInferencer::unloadModel()
{
	// Release IO binding before session (it holds a session pointer)
	impl_->ioBinding = oe::onnx::GpuIoBinding{};
	impl_->handle.unload();

	// nvJPEG cleanup
	if (impl_->nvjpegParams)  { nvjpegEncoderParamsDestroy(impl_->nvjpegParams);  impl_->nvjpegParams  = nullptr; }
	if (impl_->nvjpegState)   { nvjpegEncoderStateDestroy(impl_->nvjpegState);    impl_->nvjpegState   = nullptr; }
	if (impl_->nvjpegHandle)  { nvjpegDestroy(impl_->nvjpegHandle);               impl_->nvjpegHandle  = nullptr; }

	impl_->d_bgrOut.free();
	impl_->d_bgr.free();
	impl_->pinnedJpeg = PinnedStagingBuffer{};
	impl_->pinnedBgr = PinnedStagingBuffer{};
	impl_->stream = CudaStream{};
	impl_->inputBuffer.clear();
	OE_LOG_INFO("basicvsrpp_model_unloaded");
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxBasicVsrppInferencer::currentVramUsageBytes() const noexcept
{
	if (!impl_->handle.loaded) return 0;
	return kBasicVsrppMiB * kBytesPerMebibyte;
}

// ---------------------------------------------------------------------------
// cudaStream
// ---------------------------------------------------------------------------

cudaStream_t OnnxBasicVsrppInferencer::cudaStream() const noexcept
{
	return impl_->stream.get();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<DenoiseInferencer> createOnnxBasicVsrppInferencer()
{
	return std::make_unique<OnnxBasicVsrppInferencer>();
}

