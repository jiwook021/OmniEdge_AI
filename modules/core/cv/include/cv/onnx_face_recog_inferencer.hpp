#pragma once

#include "face_recog_inferencer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// OnnxFaceRecogInferencer — SCRFD (detection) + AdaFace/MobileFaceNet (recognition)
// via ONNX Runtime with TensorRT EP + CUDA EP.
//
// Replaces InspireFaceInferencer as the default face recognition backend.
// All variants produce 512-d L2-normalised embeddings and 5-point landmarks.
//
// Model directory layout:
//   <modelPackPath>/
//     detector.onnx       — SCRFD-10G or SCRFD-2.5G
//     recognizer.onnx     — AdaFace IR-101, IR-50, or MobileFaceNet
//
// PIMPL hides ONNX Runtime headers from consumers.
// ---------------------------------------------------------------------------


/// Selectable face recognition model variants.
enum class FaceRecogVariant : uint8_t {
	kScrfdAdaFace101,       ///< SCRFD-10G + AdaFace IR-101 (default, maximum accuracy)
	kScrfdAdaFace50,        ///< SCRFD-10G + AdaFace IR-50  (balanced)
	kScrfdMobileFaceNet,    ///< SCRFD-2.5G + ArcFace MobileFaceNet (minimum VRAM)
};

/// Parse a variant name string from YAML config.
/// Returns kScrfdAdaFace101 for unrecognised values.
[[nodiscard]] FaceRecogVariant parseFaceRecogVariant(std::string_view name) noexcept;

/// Human-readable variant name for logging.
[[nodiscard]] std::string_view faceRecogVariantName(FaceRecogVariant v) noexcept;

/// Estimated VRAM usage in bytes for a given variant.
[[nodiscard]] std::size_t faceRecogVariantVramBytes(FaceRecogVariant v) noexcept;

class OnnxFaceRecogInferencer final : public FaceRecogInferencer {
public:
	explicit OnnxFaceRecogInferencer(
		FaceRecogVariant variant = FaceRecogVariant::kScrfdAdaFace101);
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
	FaceRecogVariant variant = FaceRecogVariant::kScrfdAdaFace101);

