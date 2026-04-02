#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// OmniEdge_AI — YOLOv8-seg TensorRT Engine Wrapper
//
// Wraps a serialized YOLOv8n-seg TensorRT FP16 engine.
//
// YOLOv8-seg exported tensors (standard ONNX export):
//   output0 : [1, 116, 3549]  — detections:
//               [0:4]   cx, cy, w, h
//               [4:84]  class scores (80 COCO classes)
//               [84:116] 32 mask coefficients
//   output1 : [1, 32, 104, 104] — prototype masks
//
// Pipeline:
//   1. preprocess()  — BGR 1920×1080 → float16 416×416 (resize + normalize)
//   2. infer()       — run TRT engine
//   3. postprocess() — NMS on class-0 (person), compute binary mask at 104×104
//                      Result stored in d_segmentationMask_ (device memory, [104×104])
//
// All GPU operations run on the caller-supplied CUDA stream.
// Engine GPU buffers are owned by this class.
// Thread safety: none. One instance per process.
// ---------------------------------------------------------------------------


/// Minimum detection confidence to accept a person segmentation.
inline constexpr float kYoloConfThreshold = 0.5f;

/// NMS IoU threshold for duplicate suppression.
inline constexpr float kYoloNmsThreshold  = 0.45f;

/// YOLOv8-seg input dimensions.
/// Reduced from 640×640 to 416×416 to save ~250 MiB VRAM (activation workspace
/// scales quadratically with input resolution).  416×416 retains adequate
/// segmentation quality for webcam distances (1-2 people, upper body).
/// NOTE: The TRT engine must be rebuilt at 416×416 to match these constants.
inline constexpr int kYoloInputW  = 416;
inline constexpr int kYoloInputH  = 416;

/// YOLOv8-seg prototype mask resolution (output1 spatial dims).
/// Mask protos are at 1/4 input resolution: 416/4 = 104.
inline constexpr int kYoloMaskW   = 104;
inline constexpr int kYoloMaskH   = 104;

/// COCO person class index.
inline constexpr int kYoloCocoPerson = 0;

/// Total elements in the [1,116,3549] detection output.
/// At 416×416: P3(52²) + P4(26²) + P5(13²) = 2704 + 676 + 169 = 3549.
inline constexpr int kYoloNumDetections   = 3549;
inline constexpr int kYoloDetectionDim    = 116;

/// Number of mask prototype channels.
inline constexpr int kYoloMaskProtos      = 32;

// ---------------------------------------------------------------------------
// TRT logger — suppresses INFO-level TRT messages in production
// ---------------------------------------------------------------------------

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ---------------------------------------------------------------------------
// SegResult — output of one YoloSegEngine::infer() call
// ---------------------------------------------------------------------------

struct SegResult {
    bool     hasPerson   = false;    ///< true if any person was detected
    float    confidence  = 0.0f;     ///< confidence of the best person detection
    float*   d_segmentationMask   = nullptr;  ///< device ptr to [104×104] float mask (engine-owned)
};

// ---------------------------------------------------------------------------
// YoloSegEngine
// ---------------------------------------------------------------------------

class YoloSegEngine {
public:
    YoloSegEngine() = default;
    ~YoloSegEngine();

    YoloSegEngine(const YoloSegEngine&)            = delete;
    YoloSegEngine& operator=(const YoloSegEngine&) = delete;

    /**
     * @brief Load and prepare a serialized TRT engine.
     *
     * @param enginePath  Path to *.engine file (yolov8n-seg FP16).
     * @throws std::runtime_error on load / binding failure.
     */
    void initialize(const std::string& enginePath);

    /**
     * @brief Run one full inference cycle: preprocess → infer → postprocess.
     *
     * @param d_bgr1080  Device pointer to 1920×1080 BGR24 frame (3-byte/pixel,
     *                   row-major, no padding).
     * @param stream     CUDA stream to enqueue all work onto.
     * @return           SegResult with hasPerson=false when no person found.
     *
     * The returned d_segmentationMask pointer is valid until the next call to infer()
     * or until this object is destroyed — do not free it.
     */
    [[nodiscard]] SegResult infer(const uint8_t* d_bgr1080, cudaStream_t stream);

    /** @brief Release all TRT and CUDA resources. */
    void destroy() noexcept;

private:
    void allocateBuffers();

    // TRT runtime objects
    TrtLogger                                   logger_;
    struct TrtDeleter {
        void operator()(nvinfer1::IRuntime* p) const { if (p) delete p; }
        void operator()(nvinfer1::ICudaEngine* p) const { if (p) delete p; }
        void operator()(nvinfer1::IExecutionContext* p) const { if (p) delete p; }
    };
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>  context_;

    // Device buffers (owned)
    void*   d_input_     = nullptr;  // [1, 3, 416, 416] FP16
    void*   d_output0_   = nullptr;  // [1, 116, 3549]   FP32 (dequantized by TRT)
    void*   d_output1_   = nullptr;  // [1, 32, 104, 104] FP32
    float*  d_segmentationMask_   = nullptr;  // [104×104] — final binary mask (float 0/1)

    // Preprocess intermediate: BGR1080 → resized float NCHW
    void*   d_resizeBuffer_ = nullptr; // [1, 3, 416, 416] FP16 intermediate
};

