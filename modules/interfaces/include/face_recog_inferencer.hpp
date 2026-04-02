#pragma once

#include <tl/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FaceRecogInferencer — pure virtual interface for face detection + embedding.
//
// Concrete implementations:
//   OnnxFaceRecogInferencer — SCRFD + AdaFace via ONNX Runtime (default)
//   InspireFaceInferencer   — InspireFace SDK + TensorRT (legacy)
//   MockFaceRecogInferencer — canned responses (tests / CPU-only CI)
//
// All implementations produce 512-d L2-normalised embeddings and 5-point
// facial landmarks (left eye, right eye, nose, left mouth, right mouth).
// ---------------------------------------------------------------------------


/// Bounding box in pixel coordinates.
struct FaceBBox {
    int x, y, w, h;
};

/// Five-point facial landmarks (eyes × 2, nose tip, mouth corners).
struct FaceLandmarks {
    float pts[5][2];  // [point_index][x/y]
};

/// Result of a single face detection + embedding pass.
struct FaceDetection {
    FaceBBox      bbox;
    FaceLandmarks landmarks;
    std::vector<float> embedding;   ///< L2-normalised feature vector
};

class FaceRecogInferencer {
public:
    virtual ~FaceRecogInferencer() = default;

    /** Load models from a model pack directory.
     *  For ONNX variants: directory containing detector.onnx + recognizer.onnx.
     *  For InspireFace: path to the TRT model pack directory.
     *  @throws std::runtime_error on model load failure.
     */
    virtual void loadModel(const std::string& modelPackPath) = 0;

    /** Detect faces in a BGR24 frame and return bounding boxes, landmarks,
     *  and 512-d L2-normalised embeddings for each face.
     *  Returns an empty vector if no face is detected (not an error).
     *
     *  @param bgrFrame  Raw BGR24 host bytes (width × height × 3)
     *  @param width     Frame width in pixels
     *  @param height    Frame height in pixels
     *  @return          Detection results, or error string.
     */
    [[nodiscard]] virtual tl::expected<std::vector<FaceDetection>, std::string>
    detect(const uint8_t* bgrFrame,
           uint32_t       width,
           uint32_t       height) = 0;

    /** Unload model and free GPU memory. */
    virtual void unload() noexcept = 0;

    /** VRAM used by this inferencer in bytes. */
    [[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;
};

