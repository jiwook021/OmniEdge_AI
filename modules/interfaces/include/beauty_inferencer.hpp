#pragma once

#include <tl/expected.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// BeautyInferencer — pure virtual interface for real-time face beautification.
//
// Pipeline per frame:
//   1. FaceMesh V2 inference → 478 landmarks
//   2. Skin mask: YCrCb threshold within face landmark convex hull
//   3. Bilateral filter on skin region (smoothing)
//   4. Regional brightness adjustment (dark circles, tone evening)
//   5. Thin-plate spline warp (face/feature reshaping)
//   6. ISP: warmth, shadow fill, highlight, sharpen
//   7. JPEG encode the result
//
// Concrete implementations:
//   OnnxBeautyInferencer   — ONNX Runtime + TensorRT EP + CUDA kernels (production)
//   StubBeautyInferencer   — passthrough JPEG encode (tests / CPU-only CI)
// ---------------------------------------------------------------------------


/// Beauty adjustment parameters forwarded from the frontend sliders.
struct BeautyParams {
    // ── Skin tab ──────────────────────────────────────────────────────────
    float smoothing   = 0.0f;   ///< Bilateral filter strength (0-100)
    float toneEven    = 0.0f;   ///< Skin tone evening (0-100)
    float darkCircles = 0.0f;   ///< Under-eye brightening (0-100)
    float blemish     = 0.0f;   ///< Blemish removal strength (0-100)
    float sharpen     = 0.0f;   ///< Sharpening amount (0-100)

    // ── Shape tab ─────────────────────────────────────────────────────────
    float faceSlim   = 0.0f;    ///< Jawline inward push (0-100)
    float eyeEnlarge = 0.0f;    ///< Eye contour expansion (0-100)
    float noseNarrow = 0.0f;    ///< Nose bridge narrowing (0-100)
    float jawReshape = 0.0f;    ///< Chin/jaw vertical shift (0-100)

    // ── Light tab ─────────────────────────────────────────────────────────
    float brightness  = 0.0f;   ///< Face region brightness (-100 to +100)
    float warmth      = 0.0f;   ///< Color temperature shift (-100 to +100)
    float shadowFill  = 0.0f;   ///< Shadow area brightening (0-100)
    float highlight   = 0.0f;   ///< Specular highlight boost (0-100)

    // ── BG tab ────────────────────────────────────────────────────────────
    enum class BgMode : uint8_t { kNone = 0, kBlur = 1, kColor = 2, kImage = 3 };
    BgMode  bgMode   = BgMode::kNone;
    uint8_t bgColorR = 0;            ///< Solid BG red   component (kColor mode)
    uint8_t bgColorG = 0;            ///< Solid BG green component (kColor mode)
    uint8_t bgColorB = 0;            ///< Solid BG blue  component (kColor mode)

    /// Returns true when all parameters are at their default (no-op) values.
    [[nodiscard]] bool isIdentity() const noexcept {
        return std::abs(smoothing)   < 1e-6f
            && std::abs(toneEven)    < 1e-6f
            && std::abs(darkCircles) < 1e-6f
            && std::abs(blemish)     < 1e-6f
            && std::abs(sharpen)     < 1e-6f
            && std::abs(faceSlim)    < 1e-6f
            && std::abs(eyeEnlarge)  < 1e-6f
            && std::abs(noseNarrow)  < 1e-6f
            && std::abs(jawReshape)  < 1e-6f
            && std::abs(brightness)  < 1e-6f
            && std::abs(warmth)      < 1e-6f
            && std::abs(shadowFill)  < 1e-6f
            && std::abs(highlight)   < 1e-6f
            && bgMode == BgMode::kNone;
    }
};


class BeautyInferencer {
public:
    virtual ~BeautyInferencer() = default;

    /** Load the FaceMesh V2 ONNX model and allocate GPU buffers.
     *  @param faceMeshOnnxPath  Path to the FaceMesh V2 ONNX model file.
     *  @throws std::runtime_error on model load failure
     *  (call only from initialize(); propagates to main as fatal).
     */
    virtual void loadModel(const std::string& faceMeshOnnxPath) = 0;

    /** Run the full beauty pipeline on one BGR24 frame:
     *  1. H2D copy via PinnedStagingBuffer
     *  2. FaceMesh inference → 478 landmarks
     *  3. Skin mask + bilateral filter + region adjustments
     *  4. TPS warp (if shape params non-zero)
     *  5. ISP: warmth/shadow/highlight/sharpen
     *  6. JPEG encode → write into outBuf
     *
     *  If no face is detected or all params are identity, encodes the
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

    /** Run the beauty pipeline but return composited BGR24 instead of JPEG.
     *  Same as processFrame() but skips JPEG encode. Used for pipeline chaining.
     *  Default returns error — concrete inferencers override when supported.
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

    /** Update beauty parameters from frontend sliders.
     *  Thread-safe: the inferencer copies the struct atomically.
     */
    virtual void setBeautyParams(const BeautyParams& params) noexcept = 0;

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
