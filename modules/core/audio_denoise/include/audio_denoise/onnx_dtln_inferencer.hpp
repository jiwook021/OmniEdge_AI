#pragma once

#include "audio_denoise_inferencer.hpp"

#include <memory>

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxDtlnInferencer
//
// Production audio denoising inferencer using DTLN (Dual-signal Transformation
// LSTM Network) via ONNX Runtime.
//
// Two-stage architecture:
//   Stage 1 (dtln_1.onnx): STFT → magnitude mask → masked STFT
//   Stage 2 (dtln_2.onnx): Time-domain waveform refinement
//
// Both models are stateful (LSTM hidden states carried between frames).
// Each model is ~2.5 MB, total VRAM ~50 MiB with CUDA EP.
//
// Frame processing: 512 samples at 16 kHz = 32 ms per frame.
// ---------------------------------------------------------------------------


class OnnxDtlnInferencer final : public AudioDenoiseInferencer {
public:
	OnnxDtlnInferencer();
	~OnnxDtlnInferencer() override;

	OnnxDtlnInferencer(const OnnxDtlnInferencer&)            = delete;
	OnnxDtlnInferencer& operator=(const OnnxDtlnInferencer&) = delete;

	[[nodiscard]] tl::expected<void, std::string> loadModel(
		const std::string& model1Path,
		const std::string& model2Path) override;

	[[nodiscard]] tl::expected<std::vector<float>, std::string> processFrame(
		std::span<const float> pcmInput) override;

	void resetState() override;
	void unloadModel() override;

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

	[[nodiscard]] std::string name() const override { return "onnx-dtln"; }

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

