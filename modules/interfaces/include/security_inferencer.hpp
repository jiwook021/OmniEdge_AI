#pragma once

#include <tl/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// SecurityInferencer — pure virtual interface for YOLO detection + annotation
// operations used by SecurityCameraNode.
//
// Concrete implementations:
//   TensorRTSecurityInferencer  — TensorRT YOLO engine (production)
//   StubSecurityInferencer      — returns empty detections (tests / CPU-only CI)
// ---------------------------------------------------------------------------

class SecurityInferencer {
public:
    virtual ~SecurityInferencer() = default;

    /** Run the GPU pipeline but return composited BGR24 instead of JPEG.
     *
     *  Same pipeline as the JPEG annotator path but skips the JPEG encode
     *  step. Used in pipeline chaining mode where the output feeds into
     *  another CV node via ShmCircularBuffer.
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

    /** Return the CUDA stream used by this inferencer for GPU work.
     *  Used by CudaFence to record completion on the correct stream.
     *  Default returns nullptr (default stream) for CPU-only stubs.
     */
    [[nodiscard]] virtual cudaStream_t cudaStream() const noexcept { return nullptr; }
};
