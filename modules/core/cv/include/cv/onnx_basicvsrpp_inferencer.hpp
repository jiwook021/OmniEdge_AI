#pragma once

#include "denoise_inferencer.hpp"

#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxBasicVsrppInferencer
//
// Production video denoising inferencer using BasicVSR++ via ONNX Runtime
// with CUDA Execution Provider.
//
// Model input:  [1, N, 3, H, W] float32 RGB normalised to [0,1]
// Model output: [1, N, 3, H, W] float32 RGB denoised
//
// The inferencer:
//   1. Converts BGR24 frames to RGB float [0,1]  (cv::cvtColor + cv::split)
//   2. Resizes to model input resolution          (cv::resize)
//   3. Stacks N frames into the temporal dimension
//   4. Runs ONNX inference (TensorRT EP, CUDA EP fallback)
//   5. Extracts the center frame from the output
//   6. Encodes the result as JPEG                 (cv::imencode)
// ---------------------------------------------------------------------------


class OnnxBasicVsrppInferencer final : public DenoiseInferencer {
public:
	OnnxBasicVsrppInferencer();
	~OnnxBasicVsrppInferencer() override;

	OnnxBasicVsrppInferencer(const OnnxBasicVsrppInferencer&)            = delete;
	OnnxBasicVsrppInferencer& operator=(const OnnxBasicVsrppInferencer&) = delete;

	[[nodiscard]] tl::expected<void, std::string> loadModel(
		const std::string& onnxModelPath) override;

	[[nodiscard]] tl::expected<std::size_t, std::string> processFrames(
		const uint8_t* const* bgrFrames,
		uint32_t frameCount,
		uint32_t width,
		uint32_t height,
		uint8_t* outJpegBuf,
		std::size_t maxJpegBytes) override;

	void unloadModel() override;

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

	[[nodiscard]] std::string name() const override { return "onnx-basicvsrpp"; }

private:
	/// Trigger TRT engine compilation during init to avoid blocking later.
	void runWarmup(const std::string& onnxModelPath);

	struct Impl;
	std::unique_ptr<Impl> impl_;
};

