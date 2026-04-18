#pragma once

#include "beauty_inferencer.hpp"

#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxBeautyInferencer
//
// Production face beautification inferencer using:
//   - FaceMesh V2 ONNX model (192x192 input, 478 3D landmarks)
//   - ONNX Runtime with TensorRT EP + CUDA EP fallback
//   - Custom CUDA kernels: skin mask, bilateral filter, region brightness,
//     sharpening, BCS, warmth, shadow fill, highlight boost
//   - nvJPEG GPU encoder for output
//
// Pipeline per frame:
//   1. H2D async copy via PinnedStagingBuffer
//   2. Fused GPU preprocess: resize + BGR→RGB + normalize
//   3. FaceMesh V2 inference → 478 landmarks (zero-copy via GpuIoBinding)
//   4. Compute face bounding box + skin mask (YCrCb threshold)
//   5. Bilateral filter within skin mask (smoothing slider)
//   6. Under-eye region brightness (dark circles slider)
//   7. ISP: BCS + warmth + shadow fill + highlight boost (Light tab)
//   8. Unsharp mask sharpen (sharpen slider)
//   9. nvJPEG encode → output buffer
//
// Thread safety: none. One instance per process.
// ---------------------------------------------------------------------------


class OnnxBeautyInferencer final : public BeautyInferencer {
public:
    OnnxBeautyInferencer();
    ~OnnxBeautyInferencer() override;

    OnnxBeautyInferencer(const OnnxBeautyInferencer&)            = delete;
    OnnxBeautyInferencer& operator=(const OnnxBeautyInferencer&) = delete;

    void loadModel(const std::string& faceMeshOnnxPath) override;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame,
                 uint32_t       width,
                 uint32_t       height,
                 uint8_t*       outBuf,
                 std::size_t    maxJpegBytes) override;

    void setBeautyParams(const BeautyParams& params) noexcept override;

    void unload() noexcept override;

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

    [[nodiscard]] cudaStream_t cudaStream() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Factory function for dependency injection in BeautyNode.
[[nodiscard]] std::unique_ptr<BeautyInferencer> createOnnxBeautyInferencer();
