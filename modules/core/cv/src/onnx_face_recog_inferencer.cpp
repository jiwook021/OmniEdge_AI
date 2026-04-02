#include "cv/onnx_face_recog_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "vram/vram_thresholds.hpp"

#include "cv/scrfd_postprocess.hpp"
#include "cv/face_align.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>


// ---------------------------------------------------------------------------
// Variant helpers
// ---------------------------------------------------------------------------

FaceRecogVariant parseFaceRecogVariant(std::string_view name) noexcept
{
	if (name == "scrfd_adaface_50")    return FaceRecogVariant::kScrfdAdaFace50;
	if (name == "scrfd_mobilefacenet") return FaceRecogVariant::kScrfdMobileFaceNet;
	return FaceRecogVariant::kScrfdAdaFace101;  // default
}

std::string_view faceRecogVariantName(FaceRecogVariant v) noexcept
{
	switch (v) {
	case FaceRecogVariant::kScrfdAdaFace101:    return "scrfd_adaface_101";
	case FaceRecogVariant::kScrfdAdaFace50:     return "scrfd_adaface_50";
	case FaceRecogVariant::kScrfdMobileFaceNet: return "scrfd_mobilefacenet";
	}
	return "unknown";
}

std::size_t faceRecogVariantVramBytes(FaceRecogVariant v) noexcept
{
	switch (v) {
	case FaceRecogVariant::kScrfdAdaFace101:
		return kFaceRecogScrfdAdaFace101MiB * 1024ULL * 1024ULL;
	case FaceRecogVariant::kScrfdAdaFace50:
		return kFaceRecogScrfdAdaFace50MiB * 1024ULL * 1024ULL;
	case FaceRecogVariant::kScrfdMobileFaceNet:
		return kFaceRecogScrfdMobileFaceNetMiB * 1024ULL * 1024ULL;
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

/// AdaFace normalization: (pixel/255 - mean) / std
constexpr float kRecogMean = 0.5f;
constexpr float kRecogStd  = 0.5f;

} // anonymous namespace

// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxFaceRecogInferencer::Impl {
	FaceRecogVariant variant;

	// --- Detection session (SCRFD) ---
	Ort::Env            envDetect{ORT_LOGGING_LEVEL_WARNING, "SCRFD"};
	Ort::SessionOptions detectOpts;
	std::unique_ptr<Ort::Session> detectSession;
	std::vector<std::string>      detectInputNames;
	std::vector<std::string>      detectOutputNames;
	std::vector<const char*>      detectInputNamePtrs;
	std::vector<const char*>      detectOutputNamePtrs;

	// --- Recognition session (AdaFace / MobileFaceNet) ---
	Ort::Env            envRecog{ORT_LOGGING_LEVEL_WARNING, "AdaFace"};
	Ort::SessionOptions recogOpts;
	std::unique_ptr<Ort::Session> recogSession;
	std::vector<std::string>      recogInputNames;
	std::vector<std::string>      recogOutputNames;
	std::vector<const char*>      recogInputNamePtrs;
	std::vector<const char*>      recogOutputNamePtrs;

	// Pre-allocated buffers (sized in loadModel, reused in detect)
	std::vector<float>   detectInputBuf;   // [1, 3, 640, 640]
	std::vector<float>   recogInputBuf;    // [1, 3, 112, 112]
	std::vector<uint8_t> letterboxBgr;     // 640*640*3 BGR
	std::vector<uint8_t> alignedFaceBgr;   // 112*112*3 BGR

	bool modelLoaded = false;
};

// ---------------------------------------------------------------------------
// Helper: configure session with TRT EP + CUDA EP (same pattern as BasicVSR++)
// ---------------------------------------------------------------------------

static void configureSession(Ort::SessionOptions& opts, const char* label)
{
	opts.SetIntraOpNumThreads(1);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	// TensorRT EP — primary, optional (may fail for unsupported ops)
	OrtTensorRTProviderOptions trtOpts{};
	trtOpts.device_id = 0;
	trtOpts.trt_max_workspace_size = 1ULL << 30;  // 1 GB workspace
	trtOpts.trt_fp16_enable = 1;
	trtOpts.trt_engine_cache_enable = 1;
	trtOpts.trt_engine_cache_path = "/tmp/oe_trt_cache";

	try {
		opts.AppendExecutionProvider_TensorRT(trtOpts);
	} catch (const Ort::Exception& ex) {
		OE_LOG_WARN("{}_trt_ep_failed: {}", label, ex.what());
	}

	// CUDA EP — required fallback for ops TRT can't compile
	OrtCUDAProviderOptions cudaOpts{};
	cudaOpts.device_id = 0;
	cudaOpts.arena_extend_strategy = 0;
	cudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
	cudaOpts.do_copy_in_default_stream = 1;

	try {
		opts.AppendExecutionProvider_CUDA(cudaOpts);
	} catch (const Ort::Exception& ex) {
		throw std::runtime_error(std::format(
			"{} CUDA EP registration failed: {}", label, ex.what()));
	}
}

// ---------------------------------------------------------------------------
// Helper: cache input/output names from a loaded session
// ---------------------------------------------------------------------------

static void cacheIONames(Ort::Session& session,
                         std::vector<std::string>& inNames,
                         std::vector<std::string>& outNames,
                         std::vector<const char*>& inPtrs,
                         std::vector<const char*>& outPtrs)
{
	Ort::AllocatorWithDefaultOptions allocator;

	inNames.clear();
	outNames.clear();

	for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
		auto name = session.GetInputNameAllocated(i, allocator);
		inNames.emplace_back(name.get());
	}
	for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
		auto name = session.GetOutputNameAllocated(i, allocator);
		outNames.emplace_back(name.get());
	}

	inPtrs.clear();
	outPtrs.clear();
	for (const auto& n : inNames)  inPtrs.push_back(n.c_str());
	for (const auto& n : outNames) outPtrs.push_back(n.c_str());
}

// ---------------------------------------------------------------------------
// Helper: letterbox resize BGR image to kDetectInputSize x kDetectInputSize
// ---------------------------------------------------------------------------

static void letterboxResize(const uint8_t* bgr, uint32_t srcW, uint32_t srcH,
                            uint8_t* dst, int targetSize,
                            const LetterboxParams& lb) noexcept
{
	// Fill with grey (128)
	std::memset(dst, 128, static_cast<std::size_t>(targetSize * targetSize * 3));

	const int newW = static_cast<int>(static_cast<float>(srcW) * lb.scale);
	const int newH = static_cast<int>(static_cast<float>(srcH) * lb.scale);
	const int offX = static_cast<int>(lb.padX);
	const int offY = static_cast<int>(lb.padY);

	// Bilinear resize into the padded region
	for (int dy = 0; dy < newH; ++dy) {
		const float sy = static_cast<float>(dy) / lb.scale;
		const int sy0 = static_cast<int>(sy);
		const int sy1 = std::min(sy0 + 1, static_cast<int>(srcH) - 1);
		const float fy = sy - static_cast<float>(sy0);

		for (int dx = 0; dx < newW; ++dx) {
			const float sx = static_cast<float>(dx) / lb.scale;
			const int sx0 = static_cast<int>(sx);
			const int sx1 = std::min(sx0 + 1, static_cast<int>(srcW) - 1);
			const float fx = sx - static_cast<float>(sx0);

			const float w00 = (1.0f - fx) * (1.0f - fy);
			const float w01 = fx * (1.0f - fy);
			const float w10 = (1.0f - fx) * fy;
			const float w11 = fx * fy;

			const uint8_t* p00 = bgr + (static_cast<uint32_t>(sy0) * srcW
			                    + static_cast<uint32_t>(sx0)) * 3;
			const uint8_t* p01 = bgr + (static_cast<uint32_t>(sy0) * srcW
			                    + static_cast<uint32_t>(sx1)) * 3;
			const uint8_t* p10 = bgr + (static_cast<uint32_t>(sy1) * srcW
			                    + static_cast<uint32_t>(sx0)) * 3;
			const uint8_t* p11 = bgr + (static_cast<uint32_t>(sy1) * srcW
			                    + static_cast<uint32_t>(sx1)) * 3;

			uint8_t* out = dst + ((offY + dy) * targetSize + (offX + dx)) * 3;
			for (int ch = 0; ch < 3; ++ch) {
				const float val = w00 * static_cast<float>(p00[ch])
				                + w01 * static_cast<float>(p01[ch])
				                + w10 * static_cast<float>(p10[ch])
				                + w11 * static_cast<float>(p11[ch]);
				out[ch] = static_cast<uint8_t>(
					std::clamp(val + 0.5f, 0.0f, 255.0f));
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Helper: BGR uint8 image → float NCHW tensor with (pixel - mean) / std
// ---------------------------------------------------------------------------

static void bgrToNchwFloat(const uint8_t* bgr, int w, int h,
                           float mean, float std,
                           float* out, bool bgrToRgb) noexcept
{
	const int hw = w * h;
	// Channel order: if bgrToRgb → R=ch0, G=ch1, B=ch2
	// BGR layout in memory: pixel[0]=B, pixel[1]=G, pixel[2]=R
	const int srcChR = bgrToRgb ? 2 : 0;
	const int srcChG = 1;
	const int srcChB = bgrToRgb ? 0 : 2;

	float* ch0 = out;
	float* ch1 = out + hw;
	float* ch2 = out + 2 * hw;

	for (int i = 0; i < hw; ++i) {
		ch0[i] = (static_cast<float>(bgr[i * 3 + srcChR]) - mean) / std;
		ch1[i] = (static_cast<float>(bgr[i * 3 + srcChG]) - mean) / std;
		ch2[i] = (static_cast<float>(bgr[i * 3 + srcChB]) - mean) / std;
	}
}

// ---------------------------------------------------------------------------
// Helper: L2-normalize a vector in-place
// ---------------------------------------------------------------------------

static void l2Normalize(float* vec, int dim) noexcept
{
	float sumSq = 0.0f;
	for (int i = 0; i < dim; ++i) {
		sumSq += vec[i] * vec[i];
	}
	const float norm = std::sqrt(sumSq);
	if (norm > 1e-10f) {
		const float invNorm = 1.0f / norm;
		for (int i = 0; i < dim; ++i) {
			vec[i] *= invNorm;
		}
	}
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

	const auto detectorPath  = std::filesystem::path(modelPackPath) / "detector.onnx";
	const auto recognizerPath = std::filesystem::path(modelPackPath) / "recognizer.onnx";

	if (!std::filesystem::exists(detectorPath)) {
		throw std::runtime_error(std::format(
			"SCRFD detector model not found: {}", detectorPath.string()));
	}
	if (!std::filesystem::exists(recognizerPath)) {
		throw std::runtime_error(std::format(
			"Face recognition model not found: {}", recognizerPath.string()));
	}

	OE_LOG_INFO("face_recog_loading: variant={}, detector={}, recognizer={}",
	            faceRecogVariantName(impl_->variant),
	            detectorPath.string(), recognizerPath.string());

	// --- Configure and create detection session ---
	configureSession(impl_->detectOpts, "scrfd");
	impl_->detectSession = std::make_unique<Ort::Session>(
		impl_->envDetect, detectorPath.c_str(), impl_->detectOpts);
	cacheIONames(*impl_->detectSession,
	             impl_->detectInputNames, impl_->detectOutputNames,
	             impl_->detectInputNamePtrs, impl_->detectOutputNamePtrs);

	OE_LOG_INFO("scrfd_session_created: inputs={}, outputs={}",
	            impl_->detectInputNames.size(), impl_->detectOutputNames.size());
	for (std::size_t i = 0; i < impl_->detectOutputNames.size(); ++i) {
		OE_LOG_DEBUG("scrfd_output[{}]: {}", i, impl_->detectOutputNames[i]);
	}

	// --- Configure and create recognition session ---
	configureSession(impl_->recogOpts, "adaface");
	impl_->recogSession = std::make_unique<Ort::Session>(
		impl_->envRecog, recognizerPath.c_str(), impl_->recogOpts);
	cacheIONames(*impl_->recogSession,
	             impl_->recogInputNames, impl_->recogOutputNames,
	             impl_->recogInputNamePtrs, impl_->recogOutputNamePtrs);

	OE_LOG_INFO("adaface_session_created: inputs={}, outputs={}",
	            impl_->recogInputNames.size(), impl_->recogOutputNames.size());

	// --- Pre-allocate buffers ---
	constexpr std::size_t detectPixels =
		static_cast<std::size_t>(kDetectInputSize * kDetectInputSize);
	constexpr std::size_t recogPixels =
		static_cast<std::size_t>(kRecogInputSize * kRecogInputSize);

	impl_->detectInputBuf.resize(3 * detectPixels);
	impl_->recogInputBuf.resize(3 * recogPixels);
	impl_->letterboxBgr.resize(detectPixels * 3);
	impl_->alignedFaceBgr.resize(recogPixels * 3);

	// --- Warmup: trigger TRT engine compilation during init ---
	OE_LOG_INFO("face_recog_warmup_start: triggering TRT engine compilation "
	            "(may take 1-3 minutes on first run, cached afterwards)");

	{
		// Detection warmup
		std::vector<float> warmupDetect(3 * detectPixels, 0.0f);
		const int64_t detectShape[] = {1, 3, kDetectInputSize, kDetectInputSize};
		auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, warmupDetect.data(), warmupDetect.size(),
			detectShape, 4);

		try {
			impl_->detectSession->Run(
				Ort::RunOptions{nullptr},
				impl_->detectInputNamePtrs.data(), &inputTensor, 1,
				impl_->detectOutputNamePtrs.data(),
				impl_->detectOutputNamePtrs.size());
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
		auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, warmupRecog.data(), warmupRecog.size(),
			recogShape, 4);

		try {
			impl_->recogSession->Run(
				Ort::RunOptions{nullptr},
				impl_->recogInputNamePtrs.data(), &inputTensor, 1,
				impl_->recogOutputNamePtrs.data(),
				impl_->recogOutputNamePtrs.size());
			OE_LOG_INFO("adaface_warmup_complete");
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("adaface_warmup_failed (will retry on first real frame): {}",
			            ex.what());
		}
	}

	impl_->modelLoaded = true;
	OE_LOG_INFO("face_recog_loaded: variant={}, vram_budget={}MiB",
	            faceRecogVariantName(impl_->variant),
	            faceRecogVariantVramBytes(impl_->variant) / (1024 * 1024));
}

// ---------------------------------------------------------------------------
// detect — full pipeline: letterbox → SCRFD → align → AdaFace → L2-normalize
// ---------------------------------------------------------------------------

tl::expected<std::vector<FaceDetection>, std::string>
OnnxFaceRecogInferencer::detect(const uint8_t* bgrFrame,
                                 uint32_t       width,
                                 uint32_t       height)
{
	OE_ZONE_SCOPED;

	if (!impl_->modelLoaded) {
		return tl::unexpected(std::string("ONNX face recognition model not loaded"));
	}

	// ------------------------------------------------------------------
	// Stage 1: Letterbox resize → NCHW float tensor → SCRFD detection
	// ------------------------------------------------------------------

	const auto lb = computeLetterbox(width, height, kDetectInputSize);

	letterboxResize(bgrFrame, width, height,
	                impl_->letterboxBgr.data(), kDetectInputSize, lb);

	// BGR→RGB + normalize to NCHW float
	bgrToNchwFloat(impl_->letterboxBgr.data(),
	               kDetectInputSize, kDetectInputSize,
	               kDetectMean, kDetectStd,
	               impl_->detectInputBuf.data(),
	               /*bgrToRgb=*/true);

	// Run detection
	const int64_t detectShape[] = {1, 3, kDetectInputSize, kDetectInputSize};
	auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto detectInput = Ort::Value::CreateTensor<float>(
		memInfo, impl_->detectInputBuf.data(), impl_->detectInputBuf.size(),
		detectShape, 4);

	std::vector<Ort::Value> detectOutputs;
	try {
		detectOutputs = impl_->detectSession->Run(
			Ort::RunOptions{nullptr},
			impl_->detectInputNamePtrs.data(), &detectInput, 1,
			impl_->detectOutputNamePtrs.data(),
			impl_->detectOutputNamePtrs.size());
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
		// Separate outputs: stride_8_{score,bbox,kps}, stride_16_{...}, stride_32_{...}
		for (int s = 0; s < kStrideCount; ++s) {
			const std::size_t baseIdx = static_cast<std::size_t>(s * 3);
			const float* scores    = detectOutputs[baseIdx].GetTensorData<float>();
			const float* bboxes    = detectOutputs[baseIdx + 1].GetTensorData<float>();
			const float* landmarks = detectOutputs[baseIdx + 2].GetTensorData<float>();

			auto shape = detectOutputs[baseIdx].GetTensorTypeAndShapeInfo().GetShape();
			const int numAnchors = static_cast<int>(shape[1]);

			decodeStride(scores, bboxes, landmarks, numAnchors,
			                    kStrides[s], kDetectInputSize,
			                    kScoreThreshold, rawDets);
		}
	} else if (numOutputs == 6) {
		// Some SCRFD variants: 3x (scores, bboxes) without landmarks
		for (int s = 0; s < kStrideCount; ++s) {
			const std::size_t baseIdx = static_cast<std::size_t>(s * 2);
			const float* scores = detectOutputs[baseIdx].GetTensorData<float>();
			const float* bboxes = detectOutputs[baseIdx + 1].GetTensorData<float>();

			auto shape = detectOutputs[baseIdx].GetTensorTypeAndShapeInfo().GetShape();
			const int numAnchors = static_cast<int>(shape[1]);

			decodeStride(scores, bboxes, nullptr, numAnchors,
			                    kStrides[s], kDetectInputSize,
			                    kScoreThreshold, rawDets);
		}
	} else {
		// Assume standard format: outputs ordered as
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
			const int numAnchors = (shape.size() >= 2)
				? static_cast<int>(shape[1])
				: static_cast<int>(shape[0]);

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
			recogOutputs = impl_->recogSession->Run(
				Ort::RunOptions{nullptr},
				impl_->recogInputNamePtrs.data(), &recogInput, 1,
				impl_->recogOutputNamePtrs.data(),
				impl_->recogOutputNamePtrs.size());
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("adaface_inference_failed: {}", ex.what());
			continue;
		}

		// Extract embedding
		const float* embData = recogOutputs[0].GetTensorData<float>();
		auto embShape = recogOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
		const int embDim = (embShape.size() >= 2)
			? static_cast<int>(embShape[1])
			: static_cast<int>(embShape[0]);

		if (embDim != kEmbeddingDim) {
			OE_LOG_WARN("adaface_embedding_dim_mismatch: expected={}, got={}",
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
	impl_->detectSession.reset();
	impl_->recogSession.reset();
	impl_->detectInputBuf.clear();
	impl_->recogInputBuf.clear();
	impl_->letterboxBgr.clear();
	impl_->alignedFaceBgr.clear();
	impl_->modelLoaded = false;
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxFaceRecogInferencer::currentVramUsageBytes() const noexcept
{
	return impl_->modelLoaded
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

