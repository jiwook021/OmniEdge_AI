#pragma once

#include "blur_inferencer.hpp"

#include <memory>


// ---------------------------------------------------------------------------
// OnnxSelfieSegBlurInferencer — MediaPipe Selfie Segmentation (binary) via
// ONNX Runtime CUDA EP.
//
// Model: onnx-community/mediapipe_selfie_segmentation/onnx/model.onnx
//        Apache 2.0, MobileNetV3 backbone, ~462 KB fp32.
//        Input : "pixel_values" [1,3,256,256] fp32, RGB [0,1] (rescale 1/255, no mean/std).
//        Output: "alphas"       [1,1,256,256] fp32, foreground (person) probability.
//
// Pipeline (per frame):
//   H2D BGR → CUDA resize+BGR→RGB+normalize → ORT inference → GaussianBlur
//   full-res → fused upscaleAndComposite (mask 256×256 → frame res) → ISP
//   (BCS + sharpen) → nvJPEG or D2H BGR24 copy.
// ---------------------------------------------------------------------------
class OnnxSelfieSegBlurInferencer final : public BlurInferencer {
public:
    OnnxSelfieSegBlurInferencer();
    ~OnnxSelfieSegBlurInferencer() override;

    OnnxSelfieSegBlurInferencer(const OnnxSelfieSegBlurInferencer&) = delete;
    OnnxSelfieSegBlurInferencer& operator=(const OnnxSelfieSegBlurInferencer&) = delete;

    void loadEngine(const std::string& modelPath,
                    uint32_t           inputWidth,
                    uint32_t           inputHeight) override;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame,
                 uint32_t       width,
                 uint32_t       height,
                 uint8_t*       outBuf,
                 std::size_t    maxJpegBytes) override;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrameGetBgr(const uint8_t* bgrFrame,
                       uint32_t       width,
                       uint32_t       height,
                       uint8_t*       outBgrBuf,
                       std::size_t    maxBgrBytes) override;

    void unload() noexcept override;

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

    void setIspParams(const IspParams& params) noexcept override;

    [[nodiscard]] cudaStream_t cudaStream() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
