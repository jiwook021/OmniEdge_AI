#include "cv/onnx_security_inferencer.hpp"

#include "common/hf_model_fetcher.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/onnx_session_handle.hpp"
#include "common/oe_tracy.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// ---------------------------------------------------------------------------
// YOLOX-Nano constants
//
// Input  : [1, 3, 416, 416] float32, NCHW, raw BGR pixel values (no /255, no
//          mean/std). YOLOX handles normalization inside the first conv layer.
// Output : [1, 3549, 85] float32, where 3549 = 52*52 + 26*26 + 13*13 predictions
//          across strides (8, 16, 32), and 85 = cx,cy,w,h,obj,80_class_scores.
// Pad    : Fill value 114 (official YOLOX preprocess); image placed at top-left,
//          right/bottom padded.
// ---------------------------------------------------------------------------
namespace {

constexpr int   kYoloxInputSize   = 416;
constexpr int   kYoloxNumAnchors  = 3549;
constexpr int   kYoloxNumClasses  = 80;
constexpr int   kYoloxChannels    = 85;  ///< 4 bbox + 1 obj + 80 classes
constexpr float kYoloxPadValue    = 114.0f;
constexpr float kNmsIouThreshold  = 0.45f;
constexpr float kScoreThreshold   = 0.1f;  ///< coarse — node applies confidenceThreshold

constexpr std::array<int, 3> kStrides = {8, 16, 32};

constexpr const char* kYoloxHfRepo = "Megvii-BaseDetection/YOLOX";
constexpr const char* kYoloxHfFile = "yolox_nano.onnx";

} // anonymous namespace


// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct OnnxSecurityInferencer::Impl {
    oe::onnx::SessionHandle session{"YoloxNano"};

    /// Flat grid offsets for anchor-free decode: [3549 * 2] = (grid_x, grid_y).
    std::vector<float> gridXY;
    /// Per-anchor stride: [3549].
    std::vector<float> strideOf;

    /// Reusable input buffer [1, 3, 416, 416].
    std::vector<float> inputBuf;
};


// ---------------------------------------------------------------------------
// Helper: precompute YOLOX anchor-free grid (called once in loadEngine)
// ---------------------------------------------------------------------------
static void buildGrids(std::vector<float>& gridXY,
                       std::vector<float>& strideOf)
{
    gridXY.clear();
    strideOf.clear();
    gridXY.reserve(kYoloxNumAnchors * 2);
    strideOf.reserve(kYoloxNumAnchors);

    for (int stride : kStrides) {
        const int gridSize = kYoloxInputSize / stride;
        for (int gy = 0; gy < gridSize; ++gy) {
            for (int gx = 0; gx < gridSize; ++gx) {
                gridXY.push_back(static_cast<float>(gx));
                gridXY.push_back(static_cast<float>(gy));
                strideOf.push_back(static_cast<float>(stride));
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Helper: YOLOX preprocess — BGR uint8 -> [1,3,416,416] float NCHW
//
// Official preprocess (demo/ONNXRuntime/onnx_inference.py):
//   padded_img = ones(416, 416, 3) * 114
//   r = min(416/h, 416/w)
//   resized_img = resize(img, (w*r, h*r))
//   padded_img[:h*r, :w*r] = resized_img       # top-left placement
//   padded_img = transpose(HWC -> CHW)         # no /255, no norm
//   return padded_img, r
//
// Returns the scale factor r so postprocess can map bboxes back to source.
// ---------------------------------------------------------------------------
static float yoloxPreprocess(const uint8_t* bgr, uint32_t srcW, uint32_t srcH,
                             float* outNchw)
{
    const float r = std::min(
        static_cast<float>(kYoloxInputSize) / static_cast<float>(srcH),
        static_cast<float>(kYoloxInputSize) / static_cast<float>(srcW));

    const int newW = static_cast<int>(static_cast<float>(srcW) * r);
    const int newH = static_cast<int>(static_cast<float>(srcH) * r);

    const cv::Mat src(static_cast<int>(srcH), static_cast<int>(srcW), CV_8UC3,
                      const_cast<uint8_t*>(bgr));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(kYoloxInputSize, kYoloxInputSize, CV_8UC3,
                   cv::Scalar(kYoloxPadValue, kYoloxPadValue, kYoloxPadValue));
    resized.copyTo(padded(cv::Rect(0, 0, newW, newH)));

    // Transpose HWC (BGR) -> CHW float32 — channel order matches YOLOX train:
    // YOLOX trains on BGR input, so NO channel swap.
    const int plane = kYoloxInputSize * kYoloxInputSize;
    for (int y = 0; y < kYoloxInputSize; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < kYoloxInputSize; ++x) {
            const int idx = y * kYoloxInputSize + x;
            outNchw[0 * plane + idx] = static_cast<float>(row[x * 3 + 0]);
            outNchw[1 * plane + idx] = static_cast<float>(row[x * 3 + 1]);
            outNchw[2 * plane + idx] = static_cast<float>(row[x * 3 + 2]);
        }
    }

    return r;
}


// ---------------------------------------------------------------------------
// Helper: IoU-based NMS on in-flight detections (COCO-80 class-wise)
// ---------------------------------------------------------------------------
static float iou(const SecurityBbox& a, const SecurityBbox& b) noexcept
{
    const float ax2 = a.x + a.w;
    const float ay2 = a.y + a.h;
    const float bx2 = b.x + b.w;
    const float by2 = b.y + b.h;

    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);

    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;

    const float unionArea = a.w * a.h + b.w * b.h - inter;
    return unionArea > 0.0f ? inter / unionArea : 0.0f;
}

static std::vector<SecurityDetection>
runNms(std::vector<SecurityDetection> dets, float iouThreshold)
{
    std::sort(dets.begin(), dets.end(),
              [](const auto& a, const auto& b) { return a.confidence > b.confidence; });

    std::vector<SecurityDetection> kept;
    std::vector<bool> suppressed(dets.size(), false);
    kept.reserve(dets.size());

    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[j].classId != dets[i].classId) continue;  // class-wise NMS
            if (iou(dets[i].bbox, dets[j].bbox) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}


// ---------------------------------------------------------------------------
// ctor / dtor
// ---------------------------------------------------------------------------
OnnxSecurityInferencer::OnnxSecurityInferencer()
    : impl_(std::make_unique<Impl>())
{}

OnnxSecurityInferencer::~OnnxSecurityInferencer() = default;


// ---------------------------------------------------------------------------
// loadEngine
// ---------------------------------------------------------------------------
tl::expected<void, std::string>
OnnxSecurityInferencer::loadEngine(const std::string& modelPath)
{
    OE_ZONE_SCOPED;

    // Auto-fetch from HF if the model file is absent.
    std::filesystem::path resolved = modelPath;
    if (!std::filesystem::exists(resolved)) {
        const auto cacheDir = resolved.has_parent_path()
            ? resolved.parent_path()
            : std::filesystem::path("model.cache/security");
        OE_LOG_INFO("yolox_fetch_start: repo={}, file={}, cache={}",
                    kYoloxHfRepo, kYoloxHfFile, cacheDir.string());
        auto fetched = fetchHfModel(kYoloxHfRepo, kYoloxHfFile, cacheDir);
        if (!fetched) {
            return tl::unexpected(std::format(
                "YOLOX-Nano fetch failed (model missing and HF download unavailable): {}",
                fetched.error()));
        }
        resolved = *fetched;
        OE_LOG_INFO("yolox_fetch_complete: path={}", resolved.string());
    }

    // TRT EP disabled: YOLOX-Nano is small (~3.9 MB) and CUDA EP is fast
    // enough. Avoids TRT timing-cache churn on first run.
    if (auto r = impl_->session.load(resolved.string(), oe::onnx::SessionConfig{
            .useTRT            = false,
            .gpuMemLimitMiB    = kSecurityCameraMiB,
            .enableCudaGraph   = true,   // fixed [1,3,416,416] input
            .exhaustiveCudnn   = true,
            .maxCudnnWorkspace = true,
        }); !r) {
        return tl::unexpected(std::format("YOLOX session load: {}", r.error()));
    }

    buildGrids(impl_->gridXY, impl_->strideOf);
    impl_->inputBuf.resize(static_cast<std::size_t>(3 * kYoloxInputSize * kYoloxInputSize));

    OE_LOG_INFO("yolox_session_ready: inputs={}, outputs={}, anchors={}",
                impl_->session.inputCount(), impl_->session.outputCount(),
                kYoloxNumAnchors);
    return {};
}


// ---------------------------------------------------------------------------
// infer
// ---------------------------------------------------------------------------
tl::expected<SecurityInferResult, std::string>
OnnxSecurityInferencer::infer(const uint8_t* bgr, uint32_t width, uint32_t height)
{
    OE_ZONE_SCOPED;

    if (!impl_->session.loaded) {
        return tl::unexpected(std::string("YOLOX session not loaded"));
    }
    if (bgr == nullptr || width == 0 || height == 0) {
        return tl::unexpected(std::string("invalid input frame"));
    }

    const float scale = yoloxPreprocess(bgr, width, height, impl_->inputBuf.data());

    // ---- Run ORT inference ----
    const std::array<int64_t, 4> inputShape = {1, 3, kYoloxInputSize, kYoloxInputSize};
    Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, impl_->inputBuf.data(), impl_->inputBuf.size(),
        inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = impl_->session.session->Run(
            Ort::RunOptions{nullptr},
            impl_->session.inputPtrs(), &inputTensor, 1,
            impl_->session.outputPtrs(), impl_->session.outputCount());
    } catch (const Ort::Exception& ex) {
        return tl::unexpected(std::format("YOLOX Run: {}", ex.what()));
    }

    if (outputs.empty()) {
        return tl::unexpected(std::string("YOLOX returned no outputs"));
    }

    // ---- Decode [1, 3549, 85] ----
    const float* out = outputs[0].GetTensorData<float>();

    std::vector<SecurityDetection> dets;
    dets.reserve(64);

    for (int i = 0; i < kYoloxNumAnchors; ++i) {
        const float* row    = out + i * kYoloxChannels;
        const float  objConf = row[4];
        if (objConf < kScoreThreshold) continue;

        // Best class score
        int   bestCls   = 0;
        float bestScore = row[5];
        for (int c = 1; c < kYoloxNumClasses; ++c) {
            if (row[5 + c] > bestScore) {
                bestScore = row[5 + c];
                bestCls   = c;
            }
        }
        const float score = objConf * bestScore;
        if (score < kScoreThreshold) continue;

        // Anchor-free decode:
        //   cx_pixel = (cx_raw + grid_x) * stride
        //   w_pixel  = exp(w_raw) * stride
        const float gridX  = impl_->gridXY[i * 2 + 0];
        const float gridY  = impl_->gridXY[i * 2 + 1];
        const float stride = impl_->strideOf[i];

        const float cx = (row[0] + gridX) * stride;
        const float cy = (row[1] + gridY) * stride;
        const float w  = std::exp(row[2]) * stride;
        const float h  = std::exp(row[3]) * stride;

        // Letterbox-space (cxcywh) -> top-left in original image coords
        const float x1_lb = cx - w * 0.5f;
        const float y1_lb = cy - h * 0.5f;

        SecurityDetection det;
        det.bbox.x      = x1_lb / scale;
        det.bbox.y      = y1_lb / scale;
        det.bbox.w      = w     / scale;
        det.bbox.h      = h     / scale;
        det.confidence  = score;
        det.classId     = bestCls;
        dets.push_back(det);
    }

    SecurityInferResult result;
    result.detections = runNms(std::move(dets), kNmsIouThreshold);
    return result;
}
