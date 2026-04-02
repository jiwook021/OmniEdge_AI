#include "face_recog_inferencer.hpp"
#include "cv/inspire_face_wrapper.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/constants/cv_constants.hpp"

#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// InspireFaceInferencer — concrete FaceRecogInferencer using InspireFace SDK.
//
// Adapts InspireFaceWrapper (C SDK RAII wrapper) to the FaceRecogInferencer
// interface used by FaceRecognitionNode.
// ---------------------------------------------------------------------------


class InspireFaceInferencer final : public FaceRecogInferencer {
public:
	void loadModel(const std::string& modelPackPath) override
	{
		wrapper_ = std::make_unique<InspireFaceWrapper>(modelPackPath);
		OE_LOG_INFO("inspire_face_loaded: path={}", modelPackPath);
	}

	[[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
	detect(const uint8_t* bgrFrame,
	       uint32_t       width,
	       uint32_t       height) override
	{
		if (!wrapper_) {
			return tl::unexpected(std::string("InspireFace not loaded"));
		}

		auto detected = wrapper_->detect(bgrFrame,
		                                  static_cast<int>(width),
		                                  static_cast<int>(height));

		std::vector<FaceDetection> results;
		results.reserve(detected.size());

		for (auto& face : detected) {
			FaceDetection det;

			// Convert normalised bbox to pixel coordinates
			det.bbox.x = static_cast<int>(face.x1 * static_cast<float>(width));
			det.bbox.y = static_cast<int>(face.y1 * static_cast<float>(height));
			det.bbox.w = static_cast<int>((face.x2 - face.x1) * static_cast<float>(width));
			det.bbox.h = static_cast<int>((face.y2 - face.y1) * static_cast<float>(height));

			// Zero-init landmarks (InspireFaceWrapper does not expose them directly)
			det.landmarks = {};

			// Extract embedding
			det.embedding.resize(kFaceEmbeddingDim);
			const bool ok = wrapper_->extractEmbedding(
				bgrFrame,
				static_cast<int>(width),
				static_cast<int>(height),
				face.token,
				det.embedding.data());

			if (!ok) {
				OE_LOG_WARN("inspire_face_embedding_failed");
				continue;
			}

			results.push_back(std::move(det));
		}

		return results;
	}

	void unload() noexcept override
	{
		wrapper_.reset();
	}

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		// InspireFace SDK does not expose a VRAM query API.
		// Return the budgeted amount from conventions.
		return wrapper_ ? kFaceRecogEstimatedVramBytes : 0;
	}

private:
	std::unique_ptr<InspireFaceWrapper> wrapper_;
};

// Factory function called by main_face_recog.cpp
[[nodiscard]] std::unique_ptr<FaceRecogInferencer> createInspireFaceInferencer()
{
	return std::make_unique<InspireFaceInferencer>();
}

