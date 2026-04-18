#pragma once

#include <tl/expected.hpp>

#include "common/constants/cv_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// FaceFilterInferencer — pure virtual interface for FaceMesh inference +
// texture warping operations (AR face filters).
//
// Pipeline per frame:
//   1. FaceMesh inference → 478×3 landmarks + 4×4 transform matrix
//   2. EMA smoothing across frames (reduce jitter)
//   3. Per-triangle affine warp of filter texture onto face region
//   4. Alpha-composite filtered overlay onto original frame
//   5. JPEG encode the composited result
//
// Concrete implementations:
//   OnnxFaceFilterInferencer  — ONNX Runtime + TensorRT EP (production)
//   StubFaceFilterInferencer  — passthrough JPEG encode (tests / CPU-only CI)
// ---------------------------------------------------------------------------


/// 478 3D landmarks from FaceMesh V2 (x, y, z in normalised coordinates).
struct FaceMeshLandmarks {
	float pts[kFaceMeshLandmarkCount][kFaceMeshCoordinateDims];  ///< [landmark_index][x/y/z], normalised to [0,1]
};

/// 4x4 facial transformation matrix from FaceMesh V2.
struct FacialTransformMatrix {
	float m[kFacialTransformMatrixDim][kFacialTransformMatrixDim];
};

/// Result of a single FaceMesh inference pass.
struct FaceMeshResult {
	FaceMeshLandmarks    landmarks;
	FacialTransformMatrix transform;
	bool                  faceDetected = false;
};

/// Metadata describing a loaded filter asset.
struct FilterInfo {
	std::string id;         ///< Unique filter identifier (e.g. "dog", "cat")
	std::string name;       ///< Human-readable name
	bool        loaded = false;
};

class FaceFilterInferencer {
public:
	virtual ~FaceFilterInferencer() = default;

	/** Load the FaceMesh ONNX model.
	 *  @param onnxPath  Path to the FaceMesh V2 ONNX model file.
	 *  @throws std::runtime_error on model load failure
	 *  (call only from initialize(); propagates to main as fatal).
	 */
	virtual void loadModel(const std::string& onnxPath) = 0;

	/** Load filter assets from a manifest file.
	 *  The manifest JSON lists available filters with their texture
	 *  atlases and UV maps.
	 *
	 *  @param manifestPath  Path to the filter manifest JSON file.
	 */
	virtual void loadFilterAssets(const std::string& manifestPath) = 0;

	/** Set the active filter by ID.
	 *  Pass an empty string or "none" to disable filtering (passthrough).
	 *
	 *  @param filterId  Filter ID from the manifest (e.g. "dog", "cat").
	 */
	virtual void setActiveFilter(const std::string& filterId) = 0;

	/** Run the full face filter pipeline on one BGR24 frame:
	 *  1. Preprocess: resize to 192x192, normalise
	 *  2. FaceMesh inference → 478 landmarks + transform matrix
	 *  3. EMA smooth landmarks across frames
	 *  4. Warp active filter texture onto face using per-triangle affine
	 *  5. Alpha-composite onto original frame
	 *  6. JPEG encode → write into outBuf
	 *
	 *  If no face is detected or no filter is active, encodes the
	 *  original frame as JPEG (passthrough).
	 *
	 *  @param bgrFrame      Pointer to BGR24 host bytes (width x height x 3)
	 *  @param width         Frame width in pixels
	 *  @param height        Frame height in pixels
	 *  @param outBuf        Destination buffer (caller owns, >= maxJpegBytes)
	 *  @param maxJpegBytes  Capacity of outBuf in bytes
	 *  @return              Actual JPEG byte count, or error string.
	 */
	[[nodiscard]] virtual tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* bgrFrame,
	             uint32_t       width,
	             uint32_t       height,
	             uint8_t*       outBuf,
	             std::size_t    maxJpegBytes) = 0;

	/** Run the face filter pipeline but return composited BGR24 instead of JPEG.
	 *
	 *  Same pipeline as processFrame() (FaceMesh -> warp -> composite)
	 *  but skips the JPEG encode step. Used in pipeline chaining mode
	 *  where the output feeds into another CV node via ShmCircularBuffer.
	 *
	 *  @param bgrFrame     Pointer to BGR24 host bytes (width * height * 3)
	 *  @param width        Frame width in pixels
	 *  @param height       Frame height in pixels
	 *  @param outBgrBuf    Destination buffer for BGR24 output (width * height * 3)
	 *  @param maxBgrBytes  Capacity of outBgrBuf in bytes
	 *  @return             Actual BGR24 byte count, or error string.
	 *
	 *  Default returns an error — concrete inferencers override when BGR24
	 *  output is supported.
	 */
	[[nodiscard]] virtual tl::expected<std::size_t, std::string>
	processFrameGetBgr(const uint8_t* bgrFrame,
	                   uint32_t       width,
	                   uint32_t       height,
	                   uint8_t*       outBgrBuf,
	                   std::size_t    maxBgrBytes)
	{
		(void)bgrFrame; (void)width; (void)height;
		(void)outBgrBuf; (void)maxBgrBytes;
		return tl::unexpected(std::string("BGR24 output not supported by this inferencer"));
	}

	/** Get the raw FaceMesh landmarks from the most recent inference.
	 *  Useful for debug overlay (draw landmarks on frame).
	 *  Returns an empty result if no inference has been run yet.
	 */
	[[nodiscard]] virtual FaceMeshResult lastLandmarks() const noexcept = 0;

	/** List available filter IDs from the loaded manifest. */
	[[nodiscard]] virtual std::vector<FilterInfo> availableFilters() const = 0;

	/** Unload model and free all GPU memory. */
	virtual void unload() noexcept = 0;

	/** VRAM used by this inferencer in bytes (for daemon eviction accounting). */
	[[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;

	/** Return the CUDA stream used by this inferencer for GPU work.
	 *  Used by CudaFence to record completion on the correct stream.
	 *  Default returns nullptr (default stream) for CPU-only stubs.
	 */
	[[nodiscard]] virtual cudaStream_t cudaStream() const noexcept { return nullptr; }
};

