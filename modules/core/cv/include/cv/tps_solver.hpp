#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Thin-Plate Spline (TPS) Weight Solver
//
// CPU-side solver for TPS face warping.  Given a set of control points
// (landmark subset) and displacement vectors (from BeautyParams sliders),
// computes the TPS weight matrix that the GPU warp kernel uses to deform
// the face image.
//
// The TPS basis function is: U(r) = r^2 * log(r)
// The system solves: [K  P] [w]   [v]
//                    [P' 0] [a] = [0]
// where K_ij = U(|p_i - p_j|), P is the affine part, w are weights,
// a are affine coefficients, and v are target displacements.
//
// Performance:
//   - 68 control points → 71×71 linear system → ~0.1 ms on CPU
//   - Recomputed only when slider values change (cached between frames)
//   - Uploaded to GPU once per recomputation
//
// Usage:
//   TpsSolver solver;
//   solver.computeWeights(landmarks478, params, frameW, frameH);
//   // Upload solver.weightsX(), solver.weightsY() to device
//   // Call launchTpsWarp() kernel with the weights
// ---------------------------------------------------------------------------

#include "cv/beauty_landmark_regions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>


namespace oe::beauty {

// ---------------------------------------------------------------------------
// Control point subset — 68 landmarks for TPS warping.
// Combines jawline (17) + left eye (16) + right eye (16) + nose (9) + lips (20).
// Excludes eyebrows and under-eye (those are for skin effects, not warping).
// ---------------------------------------------------------------------------

inline constexpr std::size_t kTpsControlPointCount =
    kJawlineCount + kLeftEyeCount + kRightEyeCount + kNoseBridgeCount + kLipsCount;

/// Build the TPS control point index array from the landmark region arrays.
/// Returns indices into the FaceMesh 478-landmark array.
inline std::vector<uint16_t> buildTpsControlIndices()
{
    std::vector<uint16_t> indices;
    indices.reserve(kTpsControlPointCount);

    for (std::size_t i = 0; i < kJawlineCount; ++i)    indices.push_back(kJawline[i]);
    for (std::size_t i = 0; i < kLeftEyeCount; ++i)    indices.push_back(kLeftEye[i]);
    for (std::size_t i = 0; i < kRightEyeCount; ++i)   indices.push_back(kRightEye[i]);
    for (std::size_t i = 0; i < kNoseBridgeCount; ++i) indices.push_back(kNoseBridge[i]);
    for (std::size_t i = 0; i < kLipsCount; ++i)       indices.push_back(kLips[i]);

    return indices;
}


// ---------------------------------------------------------------------------
// TpsSolver — computes TPS weights from landmarks + beauty slider params
// ---------------------------------------------------------------------------

class TpsSolver {
public:
    TpsSolver()
        : controlIndices_(buildTpsControlIndices())
    {}

    /// Number of control points (N).
    [[nodiscard]] std::size_t numPoints() const noexcept
    {
        return controlIndices_.size();
    }

    /// Compute TPS weights from current landmarks and beauty parameters.
    ///
    /// @param landmarks478  FaceMesh V2 output: 478×3 floats (x,y,z), normalized [0,1]
    /// @param faceSlim      Jawline inward push (0-100)
    /// @param eyeEnlarge    Eye contour expansion (0-100)
    /// @param noseNarrow    Nose bridge narrowing (0-100)
    /// @param jawReshape    Chin/jaw vertical shift (0-100)
    /// @param frameW        Frame width in pixels
    /// @param frameH        Frame height in pixels
    ///
    /// After this call, sourceX/Y() and weightsX/Y() contain the data
    /// needed by the GPU TPS warp kernel.
    void computeWeights(const float* landmarks478,
                        float faceSlim, float eyeEnlarge,
                        float noseNarrow, float jawReshape,
                        uint32_t frameW, uint32_t frameH)
    {
        const std::size_t N = controlIndices_.size();

        // Extract source control points in pixel coordinates
        srcX_.resize(N);
        srcY_.resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            const uint16_t idx = controlIndices_[i];
            srcX_[i] = landmarks478[idx * 3]     * static_cast<float>(frameW);
            srcY_[i] = landmarks478[idx * 3 + 1] * static_cast<float>(frameH);
        }

        // Compute face center X (nose tip, landmark 1) — used for jawline inward push
        const float faceCenterX = landmarks478[kFaceCenter * 3] * static_cast<float>(frameW);

        // Compute face width for displacement magnitude clamping
        float faceMinX = 1e6f, faceMaxX = -1e6f;
        for (std::size_t i = 0; i < kFaceOvalCount; ++i) {
            const float px = landmarks478[kFaceOval[i] * 3] * static_cast<float>(frameW);
            faceMinX = std::min(faceMinX, px);
            faceMaxX = std::max(faceMaxX, px);
        }
        const float faceWidth = faceMaxX - faceMinX;
        const float maxDisplacement = faceWidth * 0.15f;  // clamp to 15% of face width

        // Compute eye centers for eye enlargement
        float leftEyeCX = 0, leftEyeCY = 0;
        for (std::size_t i = 0; i < kLeftEyeCount; ++i) {
            leftEyeCX += landmarks478[kLeftEye[i] * 3]     * static_cast<float>(frameW);
            leftEyeCY += landmarks478[kLeftEye[i] * 3 + 1] * static_cast<float>(frameH);
        }
        leftEyeCX /= static_cast<float>(kLeftEyeCount);
        leftEyeCY /= static_cast<float>(kLeftEyeCount);

        float rightEyeCX = 0, rightEyeCY = 0;
        for (std::size_t i = 0; i < kRightEyeCount; ++i) {
            rightEyeCX += landmarks478[kRightEye[i] * 3]     * static_cast<float>(frameW);
            rightEyeCY += landmarks478[kRightEye[i] * 3 + 1] * static_cast<float>(frameH);
        }
        rightEyeCX /= static_cast<float>(kRightEyeCount);
        rightEyeCY /= static_cast<float>(kRightEyeCount);

        // Compute nose center line X for nose narrowing
        float noseCenterX = 0;
        for (std::size_t i = 0; i < kNoseBridgeCount; ++i) {
            noseCenterX += landmarks478[kNoseBridge[i] * 3] * static_cast<float>(frameW);
        }
        noseCenterX /= static_cast<float>(kNoseBridgeCount);

        // Build displacement vectors for each control point
        std::vector<float> dstX(N), dstY(N);
        std::size_t offset = 0;

        // Jawline points: push inward toward face center (faceSlim)
        // + vertical shift (jawReshape)
        for (std::size_t i = 0; i < kJawlineCount; ++i) {
            float dx = 0.0f, dy = 0.0f;

            if (faceSlim > 0.5f) {
                const float toCenter = faceCenterX - srcX_[offset + i];
                dx = toCenter * (faceSlim / 100.0f) * 0.15f;
            }
            if (std::abs(jawReshape) > 0.5f) {
                dy = -(jawReshape / 100.0f) * faceWidth * 0.05f;
            }

            dx = std::clamp(dx, -maxDisplacement, maxDisplacement);
            dy = std::clamp(dy, -maxDisplacement, maxDisplacement);
            dstX[offset + i] = srcX_[offset + i] + dx;
            dstY[offset + i] = srcY_[offset + i] + dy;
        }
        offset += kJawlineCount;

        // Left eye points: expand outward from eye center (eyeEnlarge)
        for (std::size_t i = 0; i < kLeftEyeCount; ++i) {
            float dx = 0.0f, dy = 0.0f;
            if (eyeEnlarge > 0.5f) {
                const float fromCX = srcX_[offset + i] - leftEyeCX;
                const float fromCY = srcY_[offset + i] - leftEyeCY;
                const float scale = (eyeEnlarge / 100.0f) * 0.15f;
                dx = fromCX * scale;
                dy = fromCY * scale;
            }
            dx = std::clamp(dx, -maxDisplacement, maxDisplacement);
            dy = std::clamp(dy, -maxDisplacement, maxDisplacement);
            dstX[offset + i] = srcX_[offset + i] + dx;
            dstY[offset + i] = srcY_[offset + i] + dy;
        }
        offset += kLeftEyeCount;

        // Right eye points: expand outward from eye center
        for (std::size_t i = 0; i < kRightEyeCount; ++i) {
            float dx = 0.0f, dy = 0.0f;
            if (eyeEnlarge > 0.5f) {
                const float fromCX = srcX_[offset + i] - rightEyeCX;
                const float fromCY = srcY_[offset + i] - rightEyeCY;
                const float scale = (eyeEnlarge / 100.0f) * 0.15f;
                dx = fromCX * scale;
                dy = fromCY * scale;
            }
            dx = std::clamp(dx, -maxDisplacement, maxDisplacement);
            dy = std::clamp(dy, -maxDisplacement, maxDisplacement);
            dstX[offset + i] = srcX_[offset + i] + dx;
            dstY[offset + i] = srcY_[offset + i] + dy;
        }
        offset += kRightEyeCount;

        // Nose bridge points: push toward center line (noseNarrow)
        for (std::size_t i = 0; i < kNoseBridgeCount; ++i) {
            float dx = 0.0f;
            if (noseNarrow > 0.5f) {
                const float toCenter = noseCenterX - srcX_[offset + i];
                dx = toCenter * (noseNarrow / 100.0f) * 0.2f;
            }
            dx = std::clamp(dx, -maxDisplacement, maxDisplacement);
            dstX[offset + i] = srcX_[offset + i] + dx;
            dstY[offset + i] = srcY_[offset + i];
        }
        offset += kNoseBridgeCount;

        // Lips: no displacement (anchor points to prevent distortion)
        for (std::size_t i = 0; i < kLipsCount; ++i) {
            dstX[offset + i] = srcX_[offset + i];
            dstY[offset + i] = srcY_[offset + i];
        }

        // Solve TPS system
        solveTps(dstX, dstY);
    }

    /// Source control point X coordinates (pixel space). Size = N.
    [[nodiscard]] const std::vector<float>& sourceX() const noexcept { return srcX_; }
    [[nodiscard]] const std::vector<float>& sourceY() const noexcept { return srcY_; }

    /// TPS weights for X displacement. Size = N + 3 (N weights + 3 affine).
    [[nodiscard]] const std::vector<float>& weightsX() const noexcept { return wxOut_; }
    [[nodiscard]] const std::vector<float>& weightsY() const noexcept { return wyOut_; }

private:
    /// TPS radial basis function: U(r) = r^2 * log(r), with U(0) = 0.
    static float tpsBasis(float r)
    {
        if (r < 1e-6f) return 0.0f;
        return r * r * std::log(r);
    }

    /// Solve the TPS linear system for X and Y displacements.
    void solveTps(const std::vector<float>& dstX,
                  const std::vector<float>& dstY)
    {
        const int N = static_cast<int>(srcX_.size());
        const int M = N + 3;  // N weights + 3 affine (a0, ax, ay)

        // Build TPS system matrix L (M × M)
        //   L = [K  P]   where K_ij = U(|p_i - p_j|)
        //       [P' 0]   P = [1, x_i, y_i] for each control point
        Eigen::MatrixXf L = Eigen::MatrixXf::Zero(M, M);

        // Fill K block (N × N)
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const float dx = srcX_[i] - srcX_[j];
                const float dy = srcY_[i] - srcY_[j];
                const float r = std::sqrt(dx * dx + dy * dy);
                const float u = tpsBasis(r);
                L(i, j) = u;
                L(j, i) = u;
            }
        }

        // Fill P block (N × 3) and P' block (3 × N)
        for (int i = 0; i < N; ++i) {
            L(i, N)     = 1.0f;
            L(i, N + 1) = srcX_[i];
            L(i, N + 2) = srcY_[i];
            L(N, i)     = 1.0f;
            L(N + 1, i) = srcX_[i];
            L(N + 2, i) = srcY_[i];
        }

        // Add small regularization to K diagonal for numerical stability
        for (int i = 0; i < N; ++i) {
            L(i, i) += 1e-3f;
        }

        // RHS vectors: target displacements relative to source
        Eigen::VectorXf rhsX = Eigen::VectorXf::Zero(M);
        Eigen::VectorXf rhsY = Eigen::VectorXf::Zero(M);
        for (int i = 0; i < N; ++i) {
            rhsX(i) = dstX[i];
            rhsY(i) = dstY[i];
        }

        // Solve via LU decomposition
        Eigen::PartialPivLU<Eigen::MatrixXf> solver(L);
        Eigen::VectorXf solX = solver.solve(rhsX);
        Eigen::VectorXf solY = solver.solve(rhsY);

        // Extract weights: [w_0 ... w_{N-1}, a_0, a_x, a_y]
        wxOut_.resize(M);
        wyOut_.resize(M);
        for (int i = 0; i < M; ++i) {
            wxOut_[i] = solX(i);
            wyOut_[i] = solY(i);
        }
    }

    std::vector<uint16_t> controlIndices_;
    std::vector<float>    srcX_, srcY_;
    std::vector<float>    wxOut_, wyOut_;
};

} // namespace oe::beauty
