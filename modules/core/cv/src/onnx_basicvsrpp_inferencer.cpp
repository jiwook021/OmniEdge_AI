#include "cv/onnx_basicvsrpp_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/onnx_session_handle.hpp"
#include "common/constants/memory_constants.hpp"
#include "common/constants/cv_constants.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxBasicVsrppInferencer::Impl {
	oe::onnx::SessionHandle handle{"BasicVSR++"};

	// Model's fixed input resolution (queried from ONNX metadata at load time)
	uint32_t modelInputH = 0;
	uint32_t modelInputW = 0;

	// Pre-allocated buffers
	std::vector<float>             inputBuffer;   // [1, N, 3, H, W] RGB float
	std::vector<uint8_t>           jpegBuf;       // reusable JPEG encode output
	std::vector<int>               jpegParams;    // cv::imencode parameters

	// -----------------------------------------------------------------
	// packFrameToChw — convert one BGR24 frame to RGB float32 CHW
	//
	// Uses cv::dnn::blobFromImage for SIMD-optimized BGR→RGB + [0,1]
	// normalization + NCHW layout in a single call.
	// -----------------------------------------------------------------
	void packFrameToChw(const uint8_t* bgrData,
	                    uint32_t srcW, uint32_t srcH,
	                    std::size_t destOffset,
	                    std::size_t pixelsPerFrame)
	{
		cv::Mat bgr(static_cast<int>(srcH), static_cast<int>(srcW), CV_8UC3,
		            const_cast<uint8_t*>(bgrData));

		// blobFromImage: resize + BGR→RGB + scale [0,255]→[0,1] + NCHW
		cv::Mat blob = cv::dnn::blobFromImage(
			bgr,
			1.0 / 255.0,
			cv::Size(static_cast<int>(modelInputW),
			         static_cast<int>(modelInputH)),
			cv::Scalar(0, 0, 0),
			/*swapRB=*/true,
			/*crop=*/false);

		// blob is [1, 3, H, W] — copy to destination offset in inputBuffer
		const std::size_t totalFloats = 3 * pixelsPerFrame;
		std::memcpy(&inputBuffer[destOffset], blob.ptr<float>(),
		            totalFloats * sizeof(float));
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
	if (auto r = impl_->handle.load(onnxModelPath, /*useTRT=*/true); !r) {
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
	if (auto r = impl_->handle.loadCudaOnly(onnxModelPath); !r) {
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

		const auto& memoryInfo = oe::onnx::cpuMemoryInfo();
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memoryInfo,
			impl_->inputBuffer.data(),
			impl_->inputBuffer.size(),
			inputShape.data(),
			inputShape.size());

		auto outputTensors = impl_->handle.session->Run(
			Ort::RunOptions{nullptr},
			impl_->handle.inputPtrs(),
			&inputTensor,
			1,
			impl_->handle.outputPtrs(),
			impl_->handle.outputCount());

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
	impl_->handle.unload();
	impl_->inputBuffer.clear();
	impl_->jpegBuf.clear();
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
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<DenoiseInferencer> createOnnxBasicVsrppInferencer()
{
	return std::make_unique<OnnxBasicVsrppInferencer>();
}

