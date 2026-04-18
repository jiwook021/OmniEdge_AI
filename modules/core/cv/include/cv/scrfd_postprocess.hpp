#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// SCRFD Post-Processing — anchor decoding, NMS, coordinate rescaling.
//
// Internal utility for OnnxFaceRecogInferencer. Not part of the public API.
//
// SCRFD outputs predictions at three stride levels (8, 16, 32).
// Each stride level produces:
//   - scores:    [1, num_anchors, 1]   — face confidence
//   - bboxes:    [1, num_anchors, 4]   — (dx, dy, dw, dh) offsets from anchor
//   - landmarks: [1, num_anchors, 10]  — 5 × (x, y) keypoint offsets
//
// The _kps variant of SCRFD fuses score+bbox+landmark into fewer outputs.
// We support both the 9-output and 3-output (fused) variants.
// ---------------------------------------------------------------------------


/// Raw face detection before NMS.
struct RawDetection {
	float x1, y1, x2, y2;     ///< Bounding box in input-space pixels
	float score;               ///< Detection confidence [0, 1]
	float landmarks[5][2];     ///< 5-point landmarks in input-space pixels
};

/// SCRFD stride levels and their anchor counts.
inline constexpr int kStrides[]    = {8, 16, 32};
inline constexpr int kStrideCount  = 3;
inline constexpr int kAnchorsPerCell = 2;  // SCRFD uses 2 anchors per cell

// ---------------------------------------------------------------------------
// Anchor grid generation for a single stride level.
// ---------------------------------------------------------------------------

inline void generateAnchors(int inputSize, int stride,
                            std::vector<float>& anchorCenters)
{
	const int gridSize = inputSize / stride;
	anchorCenters.clear();
	anchorCenters.reserve(
		static_cast<std::size_t>(gridSize * gridSize * kAnchorsPerCell * 2));

	for (int row = 0; row < gridSize; ++row) {
		for (int col = 0; col < gridSize; ++col) {
			const float cx = (static_cast<float>(col) + 0.5f) * static_cast<float>(stride);
			const float cy = (static_cast<float>(row) + 0.5f) * static_cast<float>(stride);
			for (int a = 0; a < kAnchorsPerCell; ++a) {
				anchorCenters.push_back(cx);
				anchorCenters.push_back(cy);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Decode one stride level (separate score/bbox/landmark tensors).
// ---------------------------------------------------------------------------

inline void decodeStride(
	const float* scores,        // [num_anchors, 1]
	const float* bboxes,        // [num_anchors, 4]
	const float* landmarks,     // [num_anchors, 10] or nullptr
	int numAnchors,
	int stride,
	int inputSize,
	float scoreThreshold,
	std::vector<RawDetection>& out)
{
	std::vector<float> anchorCenters;
	generateAnchors(inputSize, stride, anchorCenters);

	for (int i = 0; i < numAnchors; ++i) {
		const float conf = scores[i];
		if (conf < scoreThreshold) {
			continue;
		}

		const float cx = anchorCenters[static_cast<std::size_t>(i * 2)];
		const float cy = anchorCenters[static_cast<std::size_t>(i * 2 + 1)];
		const float s  = static_cast<float>(stride);

		// bbox: distance from anchor center to edges
		const float* b = bboxes + i * 4;
		const float x1 = cx - b[0] * s;
		const float y1 = cy - b[1] * s;
		const float x2 = cx + b[2] * s;
		const float y2 = cy + b[3] * s;

		RawDetection det{};
		det.x1    = x1;
		det.y1    = y1;
		det.x2    = x2;
		det.y2    = y2;
		det.score = conf;

		// Landmarks: offset from anchor center
		if (landmarks != nullptr) {
			const float* lm = landmarks + i * 10;
			for (int p = 0; p < 5; ++p) {
				det.landmarks[p][0] = cx + lm[p * 2]     * s;
				det.landmarks[p][1] = cy + lm[p * 2 + 1] * s;
			}
		}

		out.push_back(det);
	}
}

// ---------------------------------------------------------------------------
// IoU (Intersection over Union) for two bounding boxes.
// ---------------------------------------------------------------------------

inline float computeIoU(const RawDetection& a, const RawDetection& b) noexcept
{
	const float ix1 = std::max(a.x1, b.x1);
	const float iy1 = std::max(a.y1, b.y1);
	const float ix2 = std::min(a.x2, b.x2);
	const float iy2 = std::min(a.y2, b.y2);

	const float iw = std::max(0.0f, ix2 - ix1);
	const float ih = std::max(0.0f, iy2 - iy1);
	const float intersection = iw * ih;

	const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
	const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
	const float unionArea = areaA + areaB - intersection;

	return (unionArea > 0.0f) ? intersection / unionArea : 0.0f;
}

// ---------------------------------------------------------------------------
// Greedy Non-Maximum Suppression — sort by score, suppress overlapping.
// ---------------------------------------------------------------------------

inline void nms(std::vector<RawDetection>& dets, float iouThreshold)
{
	std::sort(dets.begin(), dets.end(),
	          [](const RawDetection& a, const RawDetection& b) {
		          return a.score > b.score;
	          });

	std::vector<bool> suppressed(dets.size(), false);
	std::vector<RawDetection> kept;
	kept.reserve(dets.size());

	for (std::size_t i = 0; i < dets.size(); ++i) {
		if (suppressed[i]) continue;
		kept.push_back(dets[i]);

		for (std::size_t j = i + 1; j < dets.size(); ++j) {
			if (!suppressed[j] && computeIoU(dets[i], dets[j]) > iouThreshold) {
				suppressed[j] = true;
			}
		}
	}

	dets = std::move(kept);
}

// ---------------------------------------------------------------------------
// Letterbox parameters — used for resize and coordinate rescaling.
// ---------------------------------------------------------------------------

struct LetterboxParams {
	float scale;    ///< Resize scale factor
	float padX;     ///< X offset (padding) in input-space pixels
	float padY;     ///< Y offset (padding) in input-space pixels
};

/// Compute letterbox parameters for resizing (origW, origH) -> (targetSize, targetSize).
inline LetterboxParams computeLetterbox(uint32_t origW, uint32_t origH,
                                        int targetSize) noexcept
{
	const float scaleX = static_cast<float>(targetSize) / static_cast<float>(origW);
	const float scaleY = static_cast<float>(targetSize) / static_cast<float>(origH);
	const float scale  = std::min(scaleX, scaleY);

	const float newW = static_cast<float>(origW) * scale;
	const float newH = static_cast<float>(origH) * scale;

	return {
		scale,
		(static_cast<float>(targetSize) - newW) * 0.5f,
		(static_cast<float>(targetSize) - newH) * 0.5f
	};
}

/// Rescale detection coordinates from letterboxed input space to original image space.
inline void rescaleToOriginal(std::vector<RawDetection>& dets,
                              const LetterboxParams& lb) noexcept
{
	const float invScale = 1.0f / lb.scale;

	for (auto& d : dets) {
		d.x1 = (d.x1 - lb.padX) * invScale;
		d.y1 = (d.y1 - lb.padY) * invScale;
		d.x2 = (d.x2 - lb.padX) * invScale;
		d.y2 = (d.y2 - lb.padY) * invScale;

		for (auto& pt : d.landmarks) {
			pt[0] = (pt[0] - lb.padX) * invScale;
			pt[1] = (pt[1] - lb.padY) * invScale;
		}
	}
}

/// Clamp detection coordinates to image bounds.
inline void clampToImage(std::vector<RawDetection>& dets,
                         uint32_t imgW, uint32_t imgH) noexcept
{
	const float maxX = static_cast<float>(imgW) - 1.0f;
	const float maxY = static_cast<float>(imgH) - 1.0f;

	for (auto& d : dets) {
		d.x1 = std::clamp(d.x1, 0.0f, maxX);
		d.y1 = std::clamp(d.y1, 0.0f, maxY);
		d.x2 = std::clamp(d.x2, 0.0f, maxX);
		d.y2 = std::clamp(d.y2, 0.0f, maxY);

		for (auto& pt : d.landmarks) {
			pt[0] = std::clamp(pt[0], 0.0f, maxX);
			pt[1] = std::clamp(pt[1], 0.0f, maxY);
		}
	}
}

