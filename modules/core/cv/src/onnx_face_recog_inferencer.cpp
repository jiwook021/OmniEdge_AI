#include "cv/onnx_face_recog_inferencer.hpp"

#include "common/hf_model_fetcher.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/oe_onnx_io_binding.hpp"
#include "common/onnx_session_handle.hpp"
#include "common/oe_tracy.hpp"
#include "vram/vram_thresholds.hpp"
#include "common/constants/video_constants.hpp"

#include "cv/scrfd_postprocess.hpp"
#include "cv/face_align.hpp"
#include "cv/face_recog_preprocess.hpp"
#include "gpu/oe_cuda_check.hpp"
#include "gpu/pinned_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>


// ---------------------------------------------------------------------------
// Variant helpers
// ---------------------------------------------------------------------------

FaceRecogVariant parseFaceRecogVariant(std::string_view /*name*/) noexcept
{
	// Only one variant supported today; any input resolves to AuraFace v1.
	return FaceRecogVariant::kScrfdAuraFaceV1;
}

std::string_view faceRecogVariantName(FaceRecogVariant v) noexcept
{
	switch (v) {
	case FaceRecogVariant::kScrfdAuraFaceV1: return "scrfd_auraface_v1";
	}
	return "unknown";
}

std::size_t faceRecogVariantVramBytes(FaceRecogVariant v) noexcept
{
	switch (v) {
	case FaceRecogVariant::kScrfdAuraFaceV1:
		return kFaceRecogScrfdAuraFaceMiB * 1024ULL * 1024ULL;
	}
	return kFaceRecogMiB * 1024ULL * 1024ULL;
}

// ---------------------------------------------------------------------------
// Detection / recognition constants
// ---------------------------------------------------------------------------

namespace {

/// SCRFD input resolution (square).
constexpr int kDetectInputSize = 640;

/// Recognition model aligned face input resolution (square).
constexpr int kRecogInputSize = 112;

/// Face embedding output dimension.
constexpr int kEmbeddingDim = 512;

/// SCRFD detection confidence threshold.
constexpr float kScoreThreshold = 0.5f;

/// NMS IoU threshold.
constexpr float kNmsIoUThreshold = 0.4f;

/// SCRFD normalization: (pixel - mean) / std
constexpr float kDetectMean = 127.5f;
constexpr float kDetectStd  = 128.0f;

/// AuraFace / ArcFace normalization: (pixel/255 - mean) / std
/// Algebraically equivalent to (pixel - 127.5) / 127.5 used by the
/// glintr100 ArcFace family.
constexpr float kRecogMean = 0.5f;
constexpr float kRecogStd  = 0.5f;

} // anonymous namespace

// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxFaceRecogInferencer::Impl {
	FaceRecogVariant variant;

	oe::onnx::SessionHandle detect{"SCRFD"};    // Detection session
	oe::onnx::SessionHandle recog{"AuraFace"};  // Recognition session (glintr100)

	// Pre-allocated CPU buffers (sized in loadModel, reused in detect)
	std::vector<float>   detectInputBuf;   // [1, 3, 640, 640]
	std::vector<float>   recogInputBuf;    // [1, 3, 112, 112]
	std::vector<uint8_t> letterboxBgr;     // 640*640*3 BGR (CPU fallback)
	std::vector<uint8_t> alignedFaceBgr;   // 112*112*3 BGR

	// GPU preprocessing buffers — fused CUDA kernels replace CPU letterbox + normalize
	bool                  gpuPreprocessReady = false;
	uint8_t*              d_bgrFrame     = nullptr;  // device: full BGR frame
	float*                d_detectInput  = nullptr;  // device: [1, 3, 640, 640] NCHW
	float*                d_recogInput   = nullptr;  // device: [1, 3, 112, 112] NCHW
	float*                d_invAffine    = nullptr;  // device: [2][3] affine matrix
	std::unique_ptr<PinnedStagingBuffer> pinnedFrame;  // pinned: BGR frame H2D staging
	cudaStream_t          cudaStream     = nullptr;

	~Impl() {
		if (d_bgrFrame)     cudaFree(d_bgrFrame);
		if (d_detectInput)  cudaFree(d_detectInput);
		if (d_recogInput)   cudaFree(d_recogInput);
		if (d_invAffine)    cudaFree(d_invAffine);
		if (cudaStream)     cudaStreamDestroy(cudaStream);
	}
};

// Session config and I/O name caching now use shared helpers from
// oe_onnx_helpers.hpp — see oe::onnx::configureSession() and
// oe::onnx::cacheIONames().

// ---------------------------------------------------------------------------
// Helper: letterbox resize BGR image using OpenCV (replaces manual bilinear)
// ---------------------------------------------------------------------------

static void letterboxResize(const uint8_t* bgr, uint32_t srcW, uint32_t srcH,
                            uint8_t* dst, int targetSize,
                            const LetterboxParams& lb)
{
	const cv::Mat src(static_cast<int>(srcH), static_cast<int>(srcW), CV_8UC3,
	                  const_cast<uint8_t*>(bgr));

	const int newW = static_cast<int>(static_cast<float>(srcW) * lb.scale);
	const int newH = static_cast<int>(static_cast<float>(srcH) * lb.scale);
	const int offX = static_cast<int>(lb.padX);
	const int offY = static_cast<int>(lb.padY);

	// Resize to scaled dimensions
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

	// Create grey-filled target and copy resized image into padded region
	cv::Mat target(targetSize, targetSize, CV_8UC3, cv::Scalar(128, 128, 128));
	resized.copyTo(target(cv::Rect(offX, offY, newW, newH)));

	std::memcpy(dst, target.data,
	            static_cast<std::size_t>(targetSize * targetSize * 3));
}

// ---------------------------------------------------------------------------
// Helper: BGR uint8 image → float NCHW tensor with (pixel - mean) / std
//
// Uses cv::dnn::blobFromImage for SIMD-optimized BGR→RGB + normalize + NCHW.
// The scaleFactor is 1.0/std, and mean is subtracted before scaling.
// cv::dnn::blobFromImage computes: (pixel - mean) * scaleFactor
// We need: (pixel - mean) / std  ≡  (pixel - mean) * (1/std)
// ---------------------------------------------------------------------------

static void bgrToNchwFloat(const uint8_t* bgr, int w, int h,
                           float mean, float std,
                           float* out, bool bgrToRgb)
{
	const cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(bgr));

	// blobFromImage: output = (pixel - mean) * scaleFactor
	// We need: (pixel - mean) / std = (pixel - mean) * (1/std)
	cv::Mat blob = cv::dnn::blobFromImage(
		src,
		1.0 / static_cast<double>(std),
		cv::Size(w, h),
		cv::Scalar(mean, mean, mean),
		bgrToRgb,
		/*crop=*/false);

	// blob is [1, 3, H, W] — copy directly to output buffer
	const std::size_t totalFloats = static_cast<std::size_t>(3 * w * h);
	std::memcpy(out, blob.ptr<float>(), totalFloats * sizeof(float));
}

// ---------------------------------------------------------------------------
// Helper: L2-normalize a vector in-place (via OpenCV)
// ---------------------------------------------------------------------------

static void l2Normalize(float* vec, int dim)
{
	cv::Mat vecMat(1, dim, CV_32F, vec);
	cv::normalize(vecMat, vecMat, 1.0, 0.0, cv::NORM_L2);
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OnnxFaceRecogInferencer::OnnxFaceRecogInferencer(FaceRecogVariant variant)
	: impl_(std::make_unique<Impl>())
{
	impl_->variant = variant;
}

OnnxFaceRecogInferencer::~OnnxFaceRecogInferencer()
{
	unload();
}

// ---------------------------------------------------------------------------
// loadModel — load detector.onnx + recognizer.onnx, warmup both sessions
// ---------------------------------------------------------------------------

void OnnxFaceRecogInferencer::loadModel(const std::string& modelPackPath)
{
	OE_ZONE_SCOPED;

	const auto packDir        = std::filesystem::path(modelPackPath);
	const auto detectorPath   = packDir / "detector.onnx";
	auto       recognizerPath = packDir / "recognizer.onnx";

	if (!std::filesystem::exists(detectorPath)) {
		throw std::runtime_error(std::format(
			"SCRFD detector model not found: {}", detectorPath.string()));
	}

	// AuraFace v1 (glintr100.onnx) auto-download from Hugging Face when the
	// canonical recognizer.onnx is absent. Apache 2.0, commercial use OK.
	if (!std::filesystem::exists(recognizerPath)) {
		OE_LOG_INFO("auraface_fetch_start: repo=fal/AuraFace-v1, file=glintr100.onnx, cache={}",
		            packDir.string());
		auto fetched = fetchHfModel("fal/AuraFace-v1", "glintr100.onnx", packDir);
		if (!fetched) {
			throw std::runtime_error(std::format(
				"AuraFace fetch failed (recognizer.onnx missing and HF download "
				"unavailable): {}", fetched.error()));
		}
		recognizerPath = *fetched;
		OE_LOG_INFO("auraface_fetch_complete: path={}", recognizerPath.string());
	}

	OE_LOG_INFO("face_recog_loading: variant={}, detector={}, recognizer={}",
	            faceRecogVariantName(impl_->variant),
	            detectorPath.string(), recognizerPath.string());

	// --- Load detection session (SCRFD) via SessionHandle ---
	// enableCudaGraph must be false when useTRT is true: TRT EP partitions
	// some nodes, leaving CUDA EP without the full graph — capture fails with
	// "not all compute graph nodes have been partitioned to CUDAExecutionProvider".
	if (auto r = impl_->detect.load(detectorPath.string(), oe::onnx::SessionConfig{
		.useTRT            = true,
		.gpuMemLimitMiB    = kFaceRecogMiB,
		.enableCudaGraph   = false,
		.exhaustiveCudnn   = true,
		.maxCudnnWorkspace = true,
	}); !r) {
		throw std::runtime_error(std::format("SCRFD load: {}", r.error()));
	}
	OE_LOG_INFO("scrfd_session_created: inputs={}, outputs={}",
	            impl_->detect.inputCount(), impl_->detect.outputCount());
	for (std::size_t i = 0; i < impl_->detect.outputNames.size(); ++i) {
		OE_LOG_DEBUG("scrfd_output[{}]: {}", i, impl_->detect.outputNames[i]);
	}

	// --- Load recognition session (AuraFace v1 / glintr100) via SessionHandle ---
	// Same TRT + CUDA graph incompatibility as SCRFD above.
	if (auto r = impl_->recog.load(recognizerPath.string(), oe::onnx::SessionConfig{
		.useTRT            = true,
		.gpuMemLimitMiB    = kFaceRecogMiB,
		.enableCudaGraph   = false,
		.exhaustiveCudnn   = true,
		.maxCudnnWorkspace = true,
	}); !r) {
		throw std::runtime_error(std::format("AuraFace load: {}", r.error()));
	}
	OE_LOG_INFO("auraface_session_created: inputs={}, outputs={}",
	            impl_->recog.inputCount(), impl_->recog.outputCount());

	// --- Pre-allocate buffers ---
	constexpr std::size_t detectPixels =
		static_cast<std::size_t>(kDetectInputSize * kDetectInputSize);
	constexpr std::size_t recogPixels =
		static_cast<std::size_t>(kRecogInputSize * kRecogInputSize);

	impl_->detectInputBuf.resize(3 * detectPixels);
	impl_->recogInputBuf.resize(3 * recogPixels);
	impl_->letterboxBgr.resize(detectPixels * 3);
	impl_->alignedFaceBgr.resize(recogPixels * 3);

	// --- Allocate GPU preprocessing buffers ---
	// OE_CUDA_CHECK throws std::runtime_error on failure, caught below.
	try {
		OE_CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->cudaStream, cudaStreamNonBlocking));
		OE_CUDA_CHECK(cudaMalloc(&impl_->d_bgrFrame, kMaxBgr24FrameBytes));
		OE_CUDA_CHECK(cudaMalloc(&impl_->d_detectInput, 3 * detectPixels * sizeof(float)));
		OE_CUDA_CHECK(cudaMalloc(&impl_->d_recogInput, 3 * recogPixels * sizeof(float)));
		OE_CUDA_CHECK(cudaMalloc(&impl_->d_invAffine, 6 * sizeof(float)));
		impl_->pinnedFrame = std::make_unique<PinnedStagingBuffer>(kMaxBgr24FrameBytes);
		impl_->gpuPreprocessReady = true;
		OE_LOG_INFO("face_recog_gpu_preprocess_ready: allocated GPU buffers");
	} catch (...) {
		OE_LOG_WARN("face_recog_gpu_preprocess_unavailable: falling back to CPU");
		impl_->gpuPreprocessReady = false;
	}

	// --- Warmup: trigger TRT engine compilation during init ---
	OE_LOG_INFO("face_recog_warmup_start: triggering TRT engine compilation "
	            "(may take 1-3 minutes on first run, cached afterwards)");

	{
		// Detection warmup
		std::vector<float> warmupDetect(3 * detectPixels, 0.0f);
		const int64_t detectShape[] = {1, 3, kDetectInputSize, kDetectInputSize};
		const auto& memInfo = oe::onnx::cpuMemoryInfo();
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, warmupDetect.data(), warmupDetect.size(),
			detectShape, 4);

		try {
			impl_->detect.session->Run(
				Ort::RunOptions{nullptr},
				impl_->detect.inputPtrs(), &inputTensor, 1,
				impl_->detect.outputPtrs(),
				impl_->detect.outputCount());
			OE_LOG_INFO("scrfd_warmup_complete");
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("scrfd_warmup_failed (will retry on first real frame): {}",
			            ex.what());
		}
	}

	{
		// Recognition warmup
		std::vector<float> warmupRecog(3 * recogPixels, 0.0f);
		const int64_t recogShape[] = {1, 3, kRecogInputSize, kRecogInputSize};
		const auto& memInfo = oe::onnx::cpuMemoryInfo();
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, warmupRecog.data(), warmupRecog.size(),
			recogShape, 4);

		try {
			impl_->recog.session->Run(
				Ort::RunOptions{nullptr},
				impl_->recog.inputPtrs(), &inputTensor, 1,
				impl_->recog.outputPtrs(),
				impl_->recog.outputCount());
			OE_LOG_INFO("auraface_warmup_complete");
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("auraface_warmup_failed (will retry on first real frame): {}",
			            ex.what());
		}
	}

	OE_LOG_INFO("face_recog_loaded: variant={}, vram_budget={}MiB",
	            faceRecogVariantName(impl_->variant),
	            faceRecogVariantVramBytes(impl_->variant) / (1024 * 1024));
}

// ---------------------------------------------------------------------------
// detect — full pipeline: letterbox → SCRFD → align → AuraFace → L2-normalize
// ---------------------------------------------------------------------------

tl::expected<std::vector<FaceDetection>, std::string>
OnnxFaceRecogInferencer::detect(const uint8_t* bgrFrame,
                                 uint32_t       width,
                                 uint32_t       height)
{
	OE_ZONE_SCOPED;

	if (!impl_->detect.loaded) {
		return tl::unexpected(std::string("ONNX face recognition model not loaded"));
	}

	// ------------------------------------------------------------------
	// Stage 1: Letterbox resize → NCHW float tensor → SCRFD detection
	// ------------------------------------------------------------------

	const auto lb = computeLetterbox(width, height, kDetectInputSize);

	if (impl_->gpuPreprocessReady) {
		// GPU path: fused letterbox + BGR→RGB + normalize in one CUDA kernel.
		// Eliminates ~3-5ms of CPU cv::resize + cv::dnn::blobFromImage.
		const std::size_t frameBytes = static_cast<std::size_t>(width) * height * 3;
		std::memcpy(impl_->pinnedFrame->data(), bgrFrame, frameBytes);
		OE_CUDA_CHECK(cudaMemcpyAsync(impl_->d_bgrFrame, impl_->pinnedFrame->data(),
		                              frameBytes, cudaMemcpyHostToDevice, impl_->cudaStream));

		oe::cv::launchLetterboxResizeNorm(
			impl_->d_bgrFrame,
			static_cast<int>(width), static_cast<int>(height),
			impl_->d_detectInput, kDetectInputSize,
			lb.scale, lb.padX, lb.padY,
			kDetectMean, kDetectStd,
			impl_->cudaStream);

		// Copy preprocessed NCHW result back to CPU for session->Run()
		OE_CUDA_CHECK(cudaMemcpyAsync(impl_->detectInputBuf.data(), impl_->d_detectInput,
		                              impl_->detectInputBuf.size() * sizeof(float),
		                              cudaMemcpyDeviceToHost, impl_->cudaStream));
		OE_CUDA_CHECK(cudaStreamSynchronize(impl_->cudaStream));
	} else {
		// CPU fallback path
		letterboxResize(bgrFrame, width, height,
		                impl_->letterboxBgr.data(), kDetectInputSize, lb);
		bgrToNchwFloat(impl_->letterboxBgr.data(),
		               kDetectInputSize, kDetectInputSize,
		               kDetectMean, kDetectStd,
		               impl_->detectInputBuf.data(),
		               /*bgrToRgb=*/true);
	}

	// Run detection
	const int64_t detectShape[] = {1, 3, kDetectInputSize, kDetectInputSize};
	const auto& memInfo = oe::onnx::cpuMemoryInfo();
	auto detectInput = Ort::Value::CreateTensor<float>(
		memInfo, impl_->detectInputBuf.data(), impl_->detectInputBuf.size(),
		detectShape, 4);

	std::vector<Ort::Value> detectOutputs;
	try {
		detectOutputs = impl_->detect.session->Run(
			Ort::RunOptions{nullptr},
			impl_->detect.inputPtrs(), &detectInput, 1,
			impl_->detect.outputPtrs(),
			impl_->detect.outputCount());
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(
			std::format("SCRFD inference failed: {}", ex.what()));
	}

	// ------------------------------------------------------------------
	// Stage 2: Decode SCRFD outputs → raw detections → NMS
	// ------------------------------------------------------------------

	std::vector<RawDetection> rawDets;

	// SCRFD _kps models output 9 tensors: 3x (scores, bboxes, landmarks)
	// or 3 fused tensors. We detect the format by output count.
	const std::size_t numOutputs = detectOutputs.size();

	if (numOutputs == 9) {
		// SCRFD 9-output models come in two formats:
		//   Interleaved: [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, ...]
		//   Grouped:     [score_8, score_16, score_32, bbox_8, ..., kps_8, ...]
		// Detect format by checking the last dimension of output[1]:
		//   - If dim == 1, outputs are grouped (output[1] = score_16)
		//   - If dim == 4, outputs are interleaved (output[1] = bbox_8)
		auto shape1 = detectOutputs[1].GetTensorTypeAndShapeInfo().GetShape();
		const int64_t dim1 = shape1.back();
		const bool isInterleaved = (dim1 == 4);

		for (int s = 0; s < kStrideCount; ++s) {
			const float* scores;
			const float* bboxes;
			const float* landmarks;
			int numAnchors;

			if (isInterleaved) {
				const std::size_t base = static_cast<std::size_t>(s * 3);
				scores    = detectOutputs[base].GetTensorData<float>();
				bboxes    = detectOutputs[base + 1].GetTensorData<float>();
				landmarks = detectOutputs[base + 2].GetTensorData<float>();
				auto shape = detectOutputs[base].GetTensorTypeAndShapeInfo().GetShape();
				// Interleaved: expect [1, num_anchors, 1] or [num_anchors, 1]
				numAnchors = static_cast<int>(
					shape.size() == 3 ? shape[1] : shape[0]);
			} else {
				// Grouped: [scores_8, scores_16, scores_32, bbox_8, ..., kps_8, ...]
				scores    = detectOutputs[static_cast<std::size_t>(s)].GetTensorData<float>();
				bboxes    = detectOutputs[static_cast<std::size_t>(s + 3)].GetTensorData<float>();
				landmarks = detectOutputs[static_cast<std::size_t>(s + 6)].GetTensorData<float>();
				// Grouped: expect [num_anchors, 1] — anchor count is shape[0]
				auto shape = detectOutputs[static_cast<std::size_t>(s)]
				                 .GetTensorTypeAndShapeInfo().GetShape();
				numAnchors = static_cast<int>(shape[0]);
			}

			decodeStride(scores, bboxes, landmarks, numAnchors,
			                    kStrides[s], kDetectInputSize,
			                    kScoreThreshold, rawDets);
		}
	} else if (numOutputs == 6) {
		// Some SCRFD variants: 3x (scores, bboxes) without landmarks
		// Detect interleaved vs grouped format
		auto shape1_6 = detectOutputs[1].GetTensorTypeAndShapeInfo().GetShape();
		const bool isInterleaved6 = (shape1_6.back() == 4);

		for (int s = 0; s < kStrideCount; ++s) {
			const float* scores;
			const float* bboxes;
			int numAnchors;

			if (isInterleaved6) {
				const std::size_t base = static_cast<std::size_t>(s * 2);
				scores = detectOutputs[base].GetTensorData<float>();
				bboxes = detectOutputs[base + 1].GetTensorData<float>();
				auto shape = detectOutputs[base].GetTensorTypeAndShapeInfo().GetShape();
				numAnchors = static_cast<int>(
					shape.size() == 3 ? shape[1] : shape[0]);
			} else {
				scores = detectOutputs[static_cast<std::size_t>(s)].GetTensorData<float>();
				bboxes = detectOutputs[static_cast<std::size_t>(s + 3)].GetTensorData<float>();
				auto shape = detectOutputs[static_cast<std::size_t>(s)]
				                 .GetTensorTypeAndShapeInfo().GetShape();
				numAnchors = static_cast<int>(shape[0]);
			}

			decodeStride(scores, bboxes, nullptr, numAnchors,
			                    kStrides[s], kDetectInputSize,
			                    kScoreThreshold, rawDets);
		}
	} else {
		if (numOutputs < 6) {
			return tl::make_unexpected("Unsupported SCRFD output count: " + std::to_string(numOutputs));
		}
		// Fallback: assume grouped format
		// [scores_8, scores_16, scores_32, bboxes_8, bboxes_16, bboxes_32,
		//  kps_8, kps_16, kps_32]
		const bool hasLandmarks = (numOutputs >= 9);
		for (int s = 0; s < kStrideCount; ++s) {
			const float* scores = detectOutputs[static_cast<std::size_t>(s)]
			                          .GetTensorData<float>();
			const float* bboxes = detectOutputs[static_cast<std::size_t>(s + 3)]
			                          .GetTensorData<float>();
			const float* landmarks = hasLandmarks
				? detectOutputs[static_cast<std::size_t>(s + 6)]
				      .GetTensorData<float>()
				: nullptr;

			auto shape = detectOutputs[static_cast<std::size_t>(s)]
			                 .GetTensorTypeAndShapeInfo().GetShape();
			const int numAnchors = static_cast<int>(shape[0]);

			decodeStride(scores, bboxes, landmarks, numAnchors,
			                    kStrides[s], kDetectInputSize,
			                    kScoreThreshold, rawDets);
		}
	}

	// NMS + rescale to original image coordinates
	nms(rawDets, kNmsIoUThreshold);
	rescaleToOriginal(rawDets, lb);
	clampToImage(rawDets, width, height);

	OE_LOG_DEBUG("scrfd_detected: {} faces (pre-NMS: {})",
	             rawDets.size(), rawDets.size());

	if (rawDets.empty()) {
		return std::vector<FaceDetection>{};
	}

	// ------------------------------------------------------------------
	// Stage 3: For each face → align → embed
	// ------------------------------------------------------------------

	std::vector<FaceDetection> results;
	results.reserve(rawDets.size());

	for (const auto& raw : rawDets) {
		// Align face to 112x112 using 5-point landmarks
		alignFace(bgrFrame, width, height,
		                      raw.landmarks,
		                      impl_->alignedFaceBgr.data());

		// Convert aligned face: BGR→RGB, normalize, NCHW
		bgrToNchwFloat(impl_->alignedFaceBgr.data(),
		               kRecogInputSize, kRecogInputSize,
		               kRecogMean * 255.0f, kRecogStd * 255.0f,
		               impl_->recogInputBuf.data(),
		               /*bgrToRgb=*/true);

		// Run recognition model
		const int64_t recogShape[] = {1, 3, kRecogInputSize, kRecogInputSize};
		auto recogInput = Ort::Value::CreateTensor<float>(
			memInfo, impl_->recogInputBuf.data(), impl_->recogInputBuf.size(),
			recogShape, 4);

		std::vector<Ort::Value> recogOutputs;
		try {
			recogOutputs = impl_->recog.session->Run(
				Ort::RunOptions{nullptr},
				impl_->recog.inputPtrs(), &recogInput, 1,
				impl_->recog.outputPtrs(),
				impl_->recog.outputCount());
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("auraface_inference_failed: {}", ex.what());
			continue;
		}

		// Extract embedding
		const float* embData = recogOutputs[0].GetTensorData<float>();
		auto embShape = recogOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
		const int embDim = (embShape.size() >= 2)
			? static_cast<int>(embShape[1])
			: static_cast<int>(embShape[0]);

		if (embDim != kEmbeddingDim) {
			OE_LOG_WARN("auraface_embedding_dim_mismatch: expected={}, got={}",
			            kEmbeddingDim, embDim);
			continue;
		}

		// Build FaceDetection result
		FaceDetection det{};

		// Bounding box in pixel coordinates
		det.bbox.x = static_cast<int>(raw.x1);
		det.bbox.y = static_cast<int>(raw.y1);
		det.bbox.w = static_cast<int>(raw.x2 - raw.x1);
		det.bbox.h = static_cast<int>(raw.y2 - raw.y1);

		// 5-point landmarks (real values from SCRFD, not zero-init)
		for (int p = 0; p < 5; ++p) {
			det.landmarks.pts[p][0] = raw.landmarks[p][0];
			det.landmarks.pts[p][1] = raw.landmarks[p][1];
		}

		// Embedding: copy and L2-normalize
		det.embedding.resize(kEmbeddingDim);
		std::memcpy(det.embedding.data(), embData,
		            static_cast<std::size_t>(kEmbeddingDim) * sizeof(float));
		l2Normalize(det.embedding.data(), kEmbeddingDim);

		results.push_back(std::move(det));
	}

	OE_LOG_DEBUG("face_recog_result: {} faces with embeddings", results.size());

	return results;
}

// ---------------------------------------------------------------------------
// unload
// ---------------------------------------------------------------------------

void OnnxFaceRecogInferencer::unload() noexcept
{
	impl_->detect.unload();
	impl_->recog.unload();
	impl_->detectInputBuf.clear();
	impl_->recogInputBuf.clear();
	impl_->letterboxBgr.clear();
	impl_->alignedFaceBgr.clear();

	// GPU buffers freed by Impl destructor; mark as unavailable
	impl_->gpuPreprocessReady = false;
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxFaceRecogInferencer::currentVramUsageBytes() const noexcept
{
	return impl_->detect.loaded
		? faceRecogVariantVramBytes(impl_->variant)
		: 0;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<FaceRecogInferencer>
createOnnxFaceRecogInferencer(FaceRecogVariant variant)
{
	return std::make_unique<OnnxFaceRecogInferencer>(variant);
}

