#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Face Alignment — 5-point similarity transform + CPU bilinear affine warp.
//
// Internal utility for OnnxFaceRecogInferencer. Not part of the public API.
//
// Aligns a detected face crop to the standard 112x112 ArcFace reference
// positions using a 2x3 affine matrix derived from 5-point landmarks
// (left eye, right eye, nose tip, left mouth corner, right mouth corner).
// ---------------------------------------------------------------------------


/// Standard ArcFace/AuraFace reference landmarks for 112x112 aligned face.
/// These are the target positions that all face recognition models expect.
inline constexpr float kRef112x112[5][2] = {
	{38.2946f, 51.6963f},   // left eye
	{73.5318f, 51.5014f},   // right eye
	{56.0252f, 71.7366f},   // nose tip
	{41.5493f, 92.3655f},   // left mouth corner
	{70.7299f, 92.2041f},   // right mouth corner
};

inline constexpr uint32_t kAlignedSize = 112;

/// 2x3 affine matrix: [a, b, tx; c, d, ty]
struct AffineMatrix {
	float a, b, tx;
	float c, d, ty;
};

// ---------------------------------------------------------------------------
// estimateSimilarityTransform — least-squares Umeyama estimation.
//
// Computes a similarity transform (rotation + uniform scale + translation)
// from 5 source landmarks to the reference 112x112 landmarks.
// Uses the closed-form Umeyama algorithm (no SVD, exploit 2D structure).
// ---------------------------------------------------------------------------

inline AffineMatrix estimateSimilarityTransform(
	const float src[5][2],
	const float dst[5][2] = kRef112x112) noexcept
{
	// Compute centroids
	float srcMeanX = 0.0f, srcMeanY = 0.0f;
	float dstMeanX = 0.0f, dstMeanY = 0.0f;
	for (int i = 0; i < 5; ++i) {
		srcMeanX += src[i][0];
		srcMeanY += src[i][1];
		dstMeanX += dst[i][0];
		dstMeanY += dst[i][1];
	}
	srcMeanX /= 5.0f; srcMeanY /= 5.0f;
	dstMeanX /= 5.0f; dstMeanY /= 5.0f;

	// Centered coordinates and covariance terms
	float srcVar = 0.0f;
	float cov00 = 0.0f, cov01 = 0.0f;
	float cov10 = 0.0f, cov11 = 0.0f;

	for (int i = 0; i < 5; ++i) {
		const float sx = src[i][0] - srcMeanX;
		const float sy = src[i][1] - srcMeanY;
		const float dx = dst[i][0] - dstMeanX;
		const float dy = dst[i][1] - dstMeanY;

		srcVar += sx * sx + sy * sy;
		cov00  += sx * dx;
		cov01  += sx * dy;
		cov10  += sy * dx;
		cov11  += sy * dy;
	}

	// Similarity transform: M = [s*cos(θ), -s*sin(θ), tx; s*sin(θ), s*cos(θ), ty]
	// For least-squares similarity (Umeyama without reflection):
	//   [a, b] = [cov00+cov11, cov01-cov10] / srcVar   (using complex number analogy)
	if (srcVar < 1e-6f) {
		// Degenerate: all source points coincide → identity transform
		return {1.0f, 0.0f, dstMeanX - srcMeanX,
		        0.0f, 1.0f, dstMeanY - srcMeanY};
	}

	const float a = (cov00 + cov11) / srcVar;
	const float b = (cov01 - cov10) / srcVar;

	const float tx = dstMeanX - (a * srcMeanX + b * srcMeanY);
	const float ty = dstMeanY - (-b * srcMeanX + a * srcMeanY);

	return {a, b, tx, -b, a, ty};
}

// ---------------------------------------------------------------------------
// invertAffine — compute the inverse of a similarity transform.
// For similarity transforms [a, b; -b, a] the inverse is:
//   [a, -b; b, a] / (a²+b²)  with adjusted translation.
// ---------------------------------------------------------------------------

inline AffineMatrix invertAffine(const AffineMatrix& m) noexcept
{
	const float det = m.a * m.d - m.b * m.c;
	if (std::fabs(det) < 1e-9f) {
		return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	}
	const float invDet = 1.0f / det;

	AffineMatrix inv{};
	inv.a  =  m.d * invDet;
	inv.b  = -m.b * invDet;
	inv.c  = -m.c * invDet;
	inv.d  =  m.a * invDet;
	inv.tx = -(inv.a * m.tx + inv.b * m.ty);
	inv.ty = -(inv.c * m.tx + inv.d * m.ty);
	return inv;
}

// ---------------------------------------------------------------------------
// warpAffine — CPU bilinear interpolation warp (BGR or RGB, 3 channels).
//
// Transforms the source image using the INVERSE of the given forward matrix,
// producing a (outW x outH) output. Pixels outside the source image are
// filled with grey (128, 128, 128).
// ---------------------------------------------------------------------------

inline void warpAffine(const uint8_t* src, uint32_t srcW, uint32_t srcH,
                       const AffineMatrix& fwdMatrix,
                       uint8_t* dst, uint32_t outW, uint32_t outH) noexcept
{
	const AffineMatrix inv = invertAffine(fwdMatrix);
	const uint32_t srcStride = srcW * 3;
	const uint32_t dstStride = outW * 3;

	for (uint32_t dy = 0; dy < outH; ++dy) {
		for (uint32_t dx = 0; dx < outW; ++dx) {
			// Map destination pixel back to source coordinates
			const float fx = static_cast<float>(dx);
			const float fy = static_cast<float>(dy);
			const float sx = inv.a * fx + inv.b * fy + inv.tx;
			const float sy = inv.c * fx + inv.d * fy + inv.ty;

			uint8_t* outPx = dst + dy * dstStride + dx * 3;

			// Bilinear interpolation bounds check
			const int sx0 = static_cast<int>(std::floor(sx));
			const int sy0 = static_cast<int>(std::floor(sy));
			const int sx1 = sx0 + 1;
			const int sy1 = sy0 + 1;

			if (sx0 < 0 || sy0 < 0 ||
			    sx1 >= static_cast<int>(srcW) ||
			    sy1 >= static_cast<int>(srcH)) {
				// Out of bounds → grey fill
				outPx[0] = 128;
				outPx[1] = 128;
				outPx[2] = 128;
				continue;
			}

			const float fx_frac = sx - static_cast<float>(sx0);
			const float fy_frac = sy - static_cast<float>(sy0);
			const float w00 = (1.0f - fx_frac) * (1.0f - fy_frac);
			const float w01 = fx_frac * (1.0f - fy_frac);
			const float w10 = (1.0f - fx_frac) * fy_frac;
			const float w11 = fx_frac * fy_frac;

			const uint8_t* p00 = src + static_cast<uint32_t>(sy0) * srcStride
			                   + static_cast<uint32_t>(sx0) * 3;
			const uint8_t* p01 = p00 + 3;
			const uint8_t* p10 = p00 + srcStride;
			const uint8_t* p11 = p10 + 3;

			for (int ch = 0; ch < 3; ++ch) {
				const float val = w00 * static_cast<float>(p00[ch])
				                + w01 * static_cast<float>(p01[ch])
				                + w10 * static_cast<float>(p10[ch])
				                + w11 * static_cast<float>(p11[ch]);
				outPx[ch] = static_cast<uint8_t>(
					std::clamp(val + 0.5f, 0.0f, 255.0f));
			}
		}
	}
}

// ---------------------------------------------------------------------------
// alignFace — full pipeline: estimate transform → warp → output 112x112.
//
// Input:  BGR image + 5-point landmarks (in original image coordinates)
// Output: 112x112 BGR aligned face in outBuf (must be 112*112*3 bytes)
// ---------------------------------------------------------------------------

inline void alignFace(const uint8_t* bgrSrc, uint32_t srcW, uint32_t srcH,
                      const float srcLandmarks[5][2],
                      uint8_t* outBuf) noexcept
{
	const AffineMatrix M = estimateSimilarityTransform(srcLandmarks, kRef112x112);
	warpAffine(bgrSrc, srcW, srcH, M, outBuf, kAlignedSize, kAlignedSize);
}

