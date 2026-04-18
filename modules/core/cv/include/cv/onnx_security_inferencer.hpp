#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxSecurityInferencer: YOLOX-Nano ONNX detection
//
// Runs YOLOX-Nano (Apache 2.0, ~3.9 MB, 416x416) via ONNX Runtime and returns
// bounding-box detections only. The security pipeline consumes bboxes for
// annotation and event logging.
//
// API surface matches the drop-in shape the SecurityCameraNode already calls:
//   loadEngine(modelPath)       -> tl::expected<void, std::string>
//   infer(bgr, w, h)            -> tl::expected<InferResult, std::string>
//
// Output class IDs are COCO-80 (person=0, bicycle=1, ...), compatible with
// the existing SecurityCameraNode::cocoClassName table.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tl/expected.hpp>


struct SecurityBbox {
    float x = 0.0f;   ///< top-left x in pixels (original image coords)
    float y = 0.0f;   ///< top-left y in pixels
    float w = 0.0f;   ///< width in pixels
    float h = 0.0f;   ///< height in pixels
};

struct SecurityDetection {
    SecurityBbox bbox;
    float        confidence = 0.0f;  ///< obj_conf * best_cls_score, [0,1]
    int          classId    = -1;    ///< COCO-80 class index
};

struct SecurityInferResult {
    std::vector<SecurityDetection> detections;
};

class OnnxSecurityInferencer {
public:
    OnnxSecurityInferencer();
    ~OnnxSecurityInferencer();

    OnnxSecurityInferencer(const OnnxSecurityInferencer&)            = delete;
    OnnxSecurityInferencer& operator=(const OnnxSecurityInferencer&) = delete;

    /// Load yolox_nano.onnx; auto-fetches from HF if absent.
    /// @param modelPath Absolute path to yolox_nano.onnx (existence not required —
    ///                  will fetch from Megvii-BaseDetection/YOLOX if missing).
    [[nodiscard]] tl::expected<void, std::string>
    loadEngine(const std::string& modelPath);

    /// Run detection on a BGR24 frame.
    /// @param bgr     Pointer to BGR24 pixel data, size = w*h*3 bytes
    /// @param width   Frame width in pixels
    /// @param height  Frame height in pixels
    [[nodiscard]] tl::expected<SecurityInferResult, std::string>
    infer(const uint8_t* bgr, uint32_t width, uint32_t height);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
