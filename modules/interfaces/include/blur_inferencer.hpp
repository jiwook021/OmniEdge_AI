#pragma once

#include <tl/expected.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// BlurInferencer — pure virtual interface for person segmentation + composite.
//
// Concrete implementations:
//   OnnxSelfieSegBlurInferencer — MediaPipe Selfie Seg ONNX + nvJPEG (production)
//   StubBlurInferencer          — returns a stub (no-ONNX-Runtime fallback / CPU-only CI)
// ---------------------------------------------------------------------------


// ISP = Image Signal Processing (brightness, contrast, saturation, sharpness).
/** ISP adjustment parameters forwarded from the frontend sliders. */
struct IspParams {
    float brightness = 0.0f;   ///< additive offset (−100 to +100)
    float contrast   = 1.0f;   ///< multiplicative (0.5 to 3.0)
    float saturation = 1.0f;   ///< 0.0 = grayscale, 1.0 = unchanged, 2.0 = double
    float sharpness  = 0.0f;   ///< 0 = none, up to 10

    [[nodiscard]] bool isIdentity() const noexcept {
        return std::abs(brightness - 0.0f) < 1e-6f
            && std::abs(contrast   - 1.0f) < 1e-6f
            && std::abs(saturation - 1.0f) < 1e-6f
            && std::abs(sharpness  - 0.0f) < 1e-6f;
    }
};

class BlurInferencer {
public:
    virtual ~BlurInferencer() = default;

    /** Load the segmentation model and allocate GPU buffers.
     *  @throws std::runtime_error on model load or CUDA allocation failure
     *  (call only from initialize(); propagates to main as fatal).
     */
    virtual void loadEngine(const std::string& enginePath,
                            uint32_t           inputWidth,
                            uint32_t           inputHeight) = 0;

    /** Run full pipeline on one BGR24 frame:
     *  1. Copy src frame to GPU via pinned staging buffer
     *  2. Preprocess → Selfie Seg inference → mask upscale
     *  3. Composite sharp foreground over Gaussian-blurred background
     *  4. nvJPEG encode → write result into outBuf (caller owns buffer)
     *
     *  @param bgrFrame     Pointer to BGR24 host bytes (width × height × 3)
     *  @param width        Frame width in pixels
     *  @param height       Frame height in pixels
     *  @param outBuf       Destination buffer (must be ≥ maxJpegBytes bytes)
     *  @param maxJpegBytes Capacity of outBuf in bytes
     *  @return             Actual JPEG byte count, or error string.
     */
    [[nodiscard]] virtual tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame,
                 uint32_t       width,
                 uint32_t       height,
                 uint8_t*       outBuf,
                 std::size_t    maxJpegBytes) = 0;

    /** Run the GPU pipeline but return composited BGR24 instead of JPEG.
     *
     *  Same pipeline as processFrame() (ISP → Selfie Seg → blur → composite)
     *  but skips the nvJPEG encode step. Used in pipeline chaining mode
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

    /** Unload engine and free all GPU memory. */
    virtual void unload() noexcept = 0;

    /** VRAM used by this inferencer in bytes (for daemon eviction accounting). */
    [[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;

    /** Update ISP parameters applied between H2D copy and inference.
     *  Default implementation is a no-op (for MockBlurInferencer).
     */
    virtual void setIspParams(const IspParams& /*params*/) noexcept {}

    /** Return the CUDA stream used by this inferencer for GPU work.
     *  Used by CudaFence to record completion on the correct stream.
     *  Default returns nullptr (default stream) for CPU-only stubs.
     */
    [[nodiscard]] virtual cudaStream_t cudaStream() const noexcept { return nullptr; }
};

