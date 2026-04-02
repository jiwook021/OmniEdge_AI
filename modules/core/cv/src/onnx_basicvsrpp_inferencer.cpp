#include "cv/onnx_basicvsrpp_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/constants/memory_constants.hpp"
#include "common/constants/cv_constants.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxBasicVsrppInferencer::Impl {
	Ort::Env                env{ORT_LOGGING_LEVEL_WARNING, "BasicVSR++"};
	Ort::SessionOptions     sessionOptions;
	std::unique_ptr<Ort::Session> session;

	// Cached input/output names
	std::vector<std::string>       inputNames;
	std::vector<std::string>       outputNames;
	std::vector<const char*>       inputNamePtrs;
	std::vector<const char*>       outputNamePtrs;

	// Model's fixed input resolution (queried from ONNX metadata at load time)
	uint32_t modelInputH = 0;
	uint32_t modelInputW = 0;

	// Pre-allocated buffers
	std::vector<float>             inputBuffer;   // [1, N, 3, H, W] RGB float
	std::vector<uint8_t>           jpegBuf;       // reusable JPEG encode output
	std::vector<int>               jpegParams;    // cv::imencode parameters

	bool modelLoaded = false;

	// -----------------------------------------------------------------
	// packFrameToChw — convert one BGR24 frame to RGB float32 CHW
	//
	// Writes 3 * pixelsPerFrame floats starting at &inputBuffer[destOffset].
	// Resizes the frame to (modelInputW x modelInputH) if dimensions differ.
	// -----------------------------------------------------------------
	void packFrameToChw(const uint8_t* bgrData,
	                    uint32_t srcW, uint32_t srcH,
	                    std::size_t destOffset,
	                    std::size_t pixelsPerFrame)
	{
		// Wrap raw BGR24 data as a cv::Mat (zero-copy)
		cv::Mat bgr(static_cast<int>(srcH), static_cast<int>(srcW), CV_8UC3,
		            const_cast<uint8_t*>(bgrData));

		// Resize to model input resolution if dimensions differ
		if (srcW != modelInputW || srcH != modelInputH) {
			cv::resize(bgr, bgr,
			           cv::Size(static_cast<int>(modelInputW),
			                    static_cast<int>(modelInputH)),
			           0, 0, cv::INTER_LINEAR);
		}

		// BGR -> RGB, then uint8 [0,255] -> float32 [0,1]
		cv::Mat rgb;
		cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

		cv::Mat rgbFloat;
		rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

		// Split into R, G, B channel planes (CHW layout)
		cv::Mat channels[3];
		cv::split(rgbFloat, channels);

		const std::size_t planeBytes = pixelsPerFrame * sizeof(float);
		std::memcpy(&inputBuffer[destOffset + 0 * pixelsPerFrame],
		            channels[0].data, planeBytes);
		std::memcpy(&inputBuffer[destOffset + 1 * pixelsPerFrame],
		            channels[1].data, planeBytes);
		std::memcpy(&inputBuffer[destOffset + 2 * pixelsPerFrame],
		            channels[2].data, planeBytes);
	}

	// -----------------------------------------------------------------
	// encodeCenterFrameToJpeg — extract center frame from CHW float
	// output, convert to BGR uint8, JPEG-encode via OpenCV.
	//
	// Returns bytes written to outJpegBuf, or 0 on failure.
	// -----------------------------------------------------------------
	std::size_t encodeCenterFrameToJpeg(
		const float* outputData,
		uint32_t frameCount,
		uint32_t outW, uint32_t outH,
		uint8_t* outJpegBuf, std::size_t maxJpegBytes)
	{
		const std::size_t outPixels =
			static_cast<std::size_t>(outW) * outH;
		const uint32_t centerIdx = frameCount / 2;
		const std::size_t centerOffset =
			static_cast<std::size_t>(centerIdx) * 3u * outPixels;

		// Wrap R, G, B channel planes as cv::Mat (zero-copy from tensor)
		const cv::Mat rPlane(static_cast<int>(outH), static_cast<int>(outW),
		                     CV_32FC1,
		                     const_cast<float*>(outputData + centerOffset));
		const cv::Mat gPlane(static_cast<int>(outH), static_cast<int>(outW),
		                     CV_32FC1,
		                     const_cast<float*>(outputData + centerOffset + outPixels));
		const cv::Mat bPlane(static_cast<int>(outH), static_cast<int>(outW),
		                     CV_32FC1,
		                     const_cast<float*>(outputData + centerOffset + 2 * outPixels));

		// Merge in BGR order (OpenCV convention), scale [0,1] -> [0,255]
		cv::Mat bgrFloat;
		cv::merge(std::vector<cv::Mat>{bPlane, gPlane, rPlane}, bgrFloat);

		cv::Mat bgrOut;
		bgrFloat.convertTo(bgrOut, CV_8UC3, 255.0);

		// JPEG encode
		jpegBuf.clear();
		if (!cv::imencode(".jpg", bgrOut, jpegBuf, jpegParams)) {
			return 0;
		}

		if (jpegBuf.size() > maxJpegBytes) {
			return 0;
		}

		std::memcpy(outJpegBuf, jpegBuf.data(), jpegBuf.size());
		return jpegBuf.size();
	}
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OnnxBasicVsrppInferencer::OnnxBasicVsrppInferencer()
	: impl_(std::make_unique<Impl>())
{
	impl_->jpegParams = {cv::IMWRITE_JPEG_QUALITY,
	                     kDefaultJpegQuality};
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
	try {
		impl_->sessionOptions.SetIntraOpNumThreads(1);
		impl_->sessionOptions.SetGraphOptimizationLevel(
			GraphOptimizationLevel::ORT_ENABLE_ALL);

		// GPU-only: TensorRT EP first, CUDA EP for ops TRT can't compile.
		OrtTensorRTProviderOptions trtOptions{};
		trtOptions.device_id = 0;
		trtOptions.trt_max_workspace_size = 1ULL << 30;  // 1 GB
		trtOptions.trt_fp16_enable = 1;
		trtOptions.trt_engine_cache_enable = 1;
		trtOptions.trt_engine_cache_path = "/tmp/oe_trt_cache";

		try {
			impl_->sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("basicvsrpp_trt_ep_failed: {}", ex.what());
		}

		// CUDA EP required
		OrtCUDAProviderOptions cudaOptions{};
		cudaOptions.device_id = 0;
		cudaOptions.arena_extend_strategy = 0;
		cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
		cudaOptions.do_copy_in_default_stream = 1;
		try {
			impl_->sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
		} catch (const Ort::Exception& ex) {
			return tl::unexpected(std::format(
				"BasicVSR++ CUDA EP registration failed: {}", ex.what()));
		}

		// Create session
		impl_->session = std::make_unique<Ort::Session>(
			impl_->env, onnxModelPath.c_str(), impl_->sessionOptions);

		// Cache input/output names
		Ort::AllocatorWithDefaultOptions allocator;
		const std::size_t numInputs  = impl_->session->GetInputCount();
		const std::size_t numOutputs = impl_->session->GetOutputCount();

		impl_->inputNames.clear();
		impl_->outputNames.clear();
		for (std::size_t i = 0; i < numInputs; ++i) {
			auto namePtr = impl_->session->GetInputNameAllocated(i, allocator);
			impl_->inputNames.emplace_back(namePtr.get());
		}
		for (std::size_t i = 0; i < numOutputs; ++i) {
			auto namePtr = impl_->session->GetOutputNameAllocated(i, allocator);
			impl_->outputNames.emplace_back(namePtr.get());
		}

		impl_->inputNamePtrs.clear();
		impl_->outputNamePtrs.clear();
		for (const auto& n : impl_->inputNames) {
			impl_->inputNamePtrs.push_back(n.c_str());
		}
		for (const auto& n : impl_->outputNames) {
			impl_->outputNamePtrs.push_back(n.c_str());
		}

		// Query model's expected input shape [1, N, 3, H, W]
		{
			auto typeInfo   = impl_->session->GetInputTypeInfo(0);
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

		impl_->modelLoaded = true;
		OE_LOG_INFO("basicvsrpp_model_loaded: path={}, input={}x{}",
		            onnxModelPath, impl_->modelInputW, impl_->modelInputH);

		// Warmup: trigger TRT engine compilation during init (not on first
		// real inference) to avoid the 1-3 min blocking that kills the watchdog.
		runWarmup(onnxModelPath);

		return {};
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"BasicVSR++ model load failed: {}", ex.what()));
	}
}

// ---------------------------------------------------------------------------
// runWarmup — trigger TRT compilation; fall back to CUDA-only on failure
// ---------------------------------------------------------------------------

void OnnxBasicVsrppInferencer::runWarmup(const std::string& onnxModelPath)
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
	auto memInfo = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault);
	auto tensor = Ort::Value::CreateTensor<float>(
		memInfo, impl_->inputBuffer.data(), wTotal,
		shape.data(), shape.size());

	try {
		impl_->session->Run(
			Ort::RunOptions{nullptr},
			impl_->inputNamePtrs.data(), &tensor, 1,
			impl_->outputNamePtrs.data(), impl_->outputNamePtrs.size());
		OE_LOG_INFO("basicvsrpp_warmup_done: ep=TensorRT+CUDA");
		return;
	} catch (const Ort::Exception& warmupEx) {
		OE_LOG_WARN("basicvsrpp_trt_warmup_failed: {}", warmupEx.what());
	}

	// TRT failed — recreate session with CUDA EP only (still GPU)
	OE_LOG_INFO("basicvsrpp_fallback_cuda: recreating session with CUDA EP only");
	impl_->session.reset();

	Ort::SessionOptions cudaOnlyOpts;
	cudaOnlyOpts.SetIntraOpNumThreads(1);
	cudaOnlyOpts.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL);

	OrtCUDAProviderOptions fallbackCuda{};
	fallbackCuda.device_id = 0;
	fallbackCuda.arena_extend_strategy = 0;
	fallbackCuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
	fallbackCuda.do_copy_in_default_stream = 1;
	cudaOnlyOpts.AppendExecutionProvider_CUDA(fallbackCuda);

	impl_->session = std::make_unique<Ort::Session>(
		impl_->env, onnxModelPath.c_str(), cudaOnlyOpts);

	auto tensor2 = Ort::Value::CreateTensor<float>(
		memInfo, impl_->inputBuffer.data(), wTotal,
		shape.data(), shape.size());
	impl_->session->Run(
		Ort::RunOptions{nullptr},
		impl_->inputNamePtrs.data(), &tensor2, 1,
		impl_->outputNamePtrs.data(), impl_->outputNamePtrs.size());
	OE_LOG_INFO("basicvsrpp_warmup_done: ep=CUDA (TRT fallback)");
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

	if (!impl_->modelLoaded) {
		return tl::unexpected(std::string("Model not loaded"));
	}

	const uint32_t inW = impl_->modelInputW;
	const uint32_t inH = impl_->modelInputH;
	const std::size_t pixelsPerFrame =
		static_cast<std::size_t>(inW) * inH;
	const std::size_t totalElements =
		static_cast<std::size_t>(frameCount) * 3u * pixelsPerFrame;

	impl_->inputBuffer.resize(totalElements);

	// Pack each BGR24 frame into the [1, N, 3, H, W] float input tensor
	for (uint32_t f = 0; f < frameCount; ++f) {
		const std::size_t frameOffset =
			static_cast<std::size_t>(f) * 3u * pixelsPerFrame;
		impl_->packFrameToChw(bgrFrames[f], width, height,
		                      frameOffset, pixelsPerFrame);
	}

	// Run ONNX inference
	try {
		const std::array<int64_t, 5> inputShape = {
			1,
			static_cast<int64_t>(frameCount),
			3,
			static_cast<int64_t>(inH),
			static_cast<int64_t>(inW)
		};

		auto memoryInfo = Ort::MemoryInfo::CreateCpu(
			OrtArenaAllocator, OrtMemTypeDefault);
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memoryInfo,
			impl_->inputBuffer.data(),
			impl_->inputBuffer.size(),
			inputShape.data(),
			inputShape.size());

		auto outputTensors = impl_->session->Run(
			Ort::RunOptions{nullptr},
			impl_->inputNamePtrs.data(),
			&inputTensor,
			1,
			impl_->outputNamePtrs.data(),
			impl_->outputNamePtrs.size());

		// Determine output dimensions from tensor shape
		// BasicVSR++ 4x: input [1,N,3,270,480] -> output [1,N,3,1080,1920]
		auto outInfo  = outputTensors[0].GetTensorTypeAndShapeInfo();
		auto outShape = outInfo.GetShape();
		const uint32_t outH = (outShape.size() == 5)
			? static_cast<uint32_t>(outShape[3]) : inH;
		const uint32_t outW = (outShape.size() == 5)
			? static_cast<uint32_t>(outShape[4]) : inW;

		const float* outputData =
			outputTensors[0].GetTensorData<float>();

		// Extract center frame, convert to BGR, JPEG-encode
		const std::size_t jpegBytes = impl_->encodeCenterFrameToJpeg(
			outputData, frameCount, outW, outH,
			outJpegBuf, maxJpegBytes);

		if (jpegBytes == 0) {
			return tl::unexpected(std::string(
				"JPEG compression failed or output exceeded buffer"));
		}

		return jpegBytes;
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"ONNX inference error: {}", ex.what()));
	}
}

// ---------------------------------------------------------------------------
// unloadModel
// ---------------------------------------------------------------------------

void OnnxBasicVsrppInferencer::unloadModel()
{
	impl_->session.reset();
	impl_->inputBuffer.clear();
	impl_->jpegBuf.clear();
	impl_->modelLoaded = false;
	OE_LOG_INFO("basicvsrpp_model_unloaded");
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxBasicVsrppInferencer::currentVramUsageBytes() const noexcept
{
	if (!impl_->modelLoaded) return 0;
	return kBasicVsrppMiB * kBytesPerMebibyte;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<DenoiseInferencer> createOnnxBasicVsrppInferencer()
{
	return std::make_unique<OnnxBasicVsrppInferencer>();
}

