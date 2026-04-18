#pragma once

#include "face_recog_inferencer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// OnnxFaceRecogInferencer — SCRFD (detection) + AuraFace v1 (recognition,
// glintr100.onnx) via ONNX Runtime with TensorRT EP + CUDA EP.
//
// Produces 512-d L2-normalised embeddings and 5-point landmarks.
//
// Model directory layout:
//   <modelPackPath>/
//     detector.onnx       — SCRFD-10G
//     recognizer.onnx     — fal/AuraFace-v1 (glintr100.onnx, Apache 2.0)
//
// On first run the recognizer is auto-downloaded from Hugging Face via
// fetchHfModel("fal/AuraFace-v1", "glintr100.onnx", …).
//
// PIMPL hides ONNX Runtime headers from consumers.
// ---------------------------------------------------------------------------


/// Selectable face recognition model variants.
///
/// Kept as an enum (single value today) so the factory / config plumbing
/// remains stable if additional variants are added later.
enum class FaceRecogVariant : uint8_t {
	kScrfdAuraFaceV1,       ///< SCRFD-10G + AuraFace v1 (glintr100.onnx)
};

/// Parse a variant name string from INI config.
/// Returns kScrfdAuraFaceV1 for unrecognised values.
[[nodiscard]] FaceRecogVariant parseFaceRecogVariant(std::string_view name) noexcept;

/// Human-readable variant name for logging.
[[nodiscard]] std::string_view faceRecogVariantName(FaceRecogVariant v) noexcept;

/// Estimated VRAM usage in bytes for a given variant.
[[nodiscard]] std::size_t faceRecogVariantVramBytes(FaceRecogVariant v) noexcept;

class OnnxFaceRecogInferencer final : public FaceRecogInferencer {
public:
	explicit OnnxFaceRecogInferencer(
		FaceRecogVariant variant = FaceRecogVariant::kScrfdAuraFaceV1);
	~OnnxFaceRecogInferencer() override;

	OnnxFaceRecogInferencer(const OnnxFaceRecogInferencer&) = delete;
	OnnxFaceRecogInferencer& operator=(const OnnxFaceRecogInferencer&) = delete;

	/// Load detector.onnx and recognizer.onnx from the model pack directory.
	/// Runs warmup inference to trigger TRT engine compilation during init.
	/// @throws std::runtime_error on model load failure.
	void loadModel(const std::string& modelPackPath) override;

	/// Detect faces, extract 5-point landmarks, and compute 512-d embeddings.
	[[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
	detect(const uint8_t* bgrFrame, uint32_t width, uint32_t height) override;

	void unload() noexcept override;

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

/// Factory function — called by main_face_recog.cpp.
[[nodiscard]] std::unique_ptr<FaceRecogInferencer>
createOnnxFaceRecogInferencer(
	FaceRecogVariant variant = FaceRecogVariant::kScrfdAuraFaceV1);

