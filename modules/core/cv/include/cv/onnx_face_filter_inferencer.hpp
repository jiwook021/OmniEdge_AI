#pragma once

#include "face_filter_inferencer.hpp"
#include "common/constants/cv_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// OnnxFaceFilterInferencer — production FaceMesh V2 + texture warp inferencer.
//
// Inference:
//   ONNX Runtime with TensorRT EP (primary) + CUDA EP (fallback).
//   FaceMesh V2: 192x192 BGR input -> 478x3 landmarks + 4x4 transform.
//
// Texture warping:
//   Per-triangle affine warp using the canonical 468-landmark triangulation.
//   Each active triangle's texture region is warped from UV space onto the
//   detected face landmark positions and alpha-composited onto the frame.
//
// Build order (incremental):
//   1. Passthrough (stub) — JPEG encode only
//   2. ONNX inference — draw raw landmarks on frame
//   3. Single hardcoded filter — overlay without UV mapping
//   4. UV-based triangle warping — the real feature
//   5. Filter switching and manifest loading — polish
// ---------------------------------------------------------------------------


/// Per-filter asset data: texture atlas + UV map.
struct FilterAsset {
	std::string id;
	std::string name;
	std::vector<uint8_t> textureAtlas;   ///< RGBA texture pixels
	uint32_t             textureWidth  = 0;
	uint32_t             textureHeight = 0;

	/// UV coordinates per triangle vertex: [triangleIndex][vertexInTriangle][u/v].
	/// Only triangles with non-transparent texture regions are populated.
	struct TriangleUV {
		uint32_t landmarkIndices[3];  ///< Indices into FaceMesh 478 landmarks
		float    uvCoords[3][2];      ///< Texture UV coords for each vertex
	};
	std::vector<TriangleUV> activeTriangles;
};

class OnnxFaceFilterInferencer : public FaceFilterInferencer {
public:
	OnnxFaceFilterInferencer();
	~OnnxFaceFilterInferencer() override;

	void loadModel(const std::string& onnxPath) override;
	void loadFilterAssets(const std::string& manifestPath) override;
	void setActiveFilter(const std::string& filterId) override;

	[[nodiscard]] tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* bgrFrame,
	             uint32_t       width,
	             uint32_t       height,
	             uint8_t*       outBuf,
	             std::size_t    maxJpegBytes) override;

	[[nodiscard]] FaceMeshResult lastLandmarks() const noexcept override;
	[[nodiscard]] std::vector<FilterInfo> availableFilters() const override;
	void unload() noexcept override;
	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

/// Factory function — creates the ONNX Runtime FaceFilter inferencer.
[[nodiscard]] std::unique_ptr<FaceFilterInferencer> createOnnxFaceFilterInferencer();

