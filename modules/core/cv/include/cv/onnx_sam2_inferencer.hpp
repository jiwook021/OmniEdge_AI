#pragma once

#include "sam2_inferencer.hpp"
#include "common/constants/cv_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// OnnxSam2Inferencer — production SAM2 inferencer using ONNX Runtime.
//
// Architecture:
//   SAM2 consists of two ONNX models:
//     1. Image Encoder  — ViT backbone (sam2_hiera_tiny_encoder.onnx)
//        Input:  [1, 3, 1024, 1024] float32 (RGB, normalised)
//        Output: [1, 256, 64, 64] float32 (image embeddings)
//
//     2. Mask Decoder    — lightweight transformer decoder
//        Input:  image embeddings + prompt embeddings (points/boxes/masks)
//        Output: [1, N, 256, 256] float32 (mask logits, N = num masks)
//                [1, N] float32 (IoU predictions per mask)
//
// Execution flow:
//   1. encodeImage() — run the image encoder once per new image
//   2. segmentWithPrompt() — run the mask decoder (fast, ~20ms)
//   3. processFrame() — convenience: encode + decode + overlay + JPEG
//
// Build order (incremental):
//   1. Stub (returns synthetic mask) — for tests
//   2. ONNX encoder + decoder — real inference
//   3. GPU overlay compositing — CUDA alpha blend
//   4. nvJPEG encoding — production output path
// ---------------------------------------------------------------------------


class OnnxSam2Inferencer : public Sam2Inferencer {
public:
    OnnxSam2Inferencer();
    ~OnnxSam2Inferencer() override;

    void loadModel(const std::string& encoderPath,
                   const std::string& decoderPath) override;

    [[nodiscard]] tl::expected<void, std::string>
    encodeImage(const uint8_t* bgrFrame,
                uint32_t       width,
                uint32_t       height) override;

    [[nodiscard]] tl::expected<Sam2Result, std::string>
    segmentWithPrompt(const Sam2Prompt& prompt) override;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t*    bgrFrame,
                 uint32_t          width,
                 uint32_t          height,
                 const Sam2Prompt& prompt,
                 uint8_t*          outBuf,
                 std::size_t       maxJpegBytes) override;

    void unload() noexcept override;

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Factory function — creates the ONNX Runtime SAM2 inferencer.
[[nodiscard]] std::unique_ptr<Sam2Inferencer> createOnnxSam2Inferencer();

