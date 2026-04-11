// tensorrt_blur_inferencer_stub.cpp — Stub inferencer for omniedge_bg_blur.
//
// PURPOSE: Provides the createTensorRTBlurInferencer() factory without requiring
// TensorRT or nvJPEG.  The binary links and launches, but processFrame()
// returns an explicit error.  Replace this file with the real TensorRT
// implementation when the YOLOv8-seg engine and nvJPEG pipeline are ready.

#include "blur_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <memory>


namespace {

class StubBlurInferencer final : public BlurInferencer {
public:
    void loadEngine(const std::string& /*enginePath*/,
                    uint32_t           /*inputWidth*/,
                    uint32_t           /*inputHeight*/) override
    {
        OE_LOG_WARN("blur_stub_load: TensorRT blur inferencer not implemented — using stub");
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* /*bgrFrame*/,
                 uint32_t       /*width*/,
                 uint32_t       /*height*/,
                 uint8_t*       /*outBuf*/,
                 std::size_t    /*maxJpegBytes*/) override
    {
        OE_ZONE_SCOPED;
        return tl::unexpected(
            std::string("TensorRT blur inferencer not available: built with stub"));
    }

    void unload() noexcept override {}

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
    {
        return 0;
    }

    void setIspParams(const IspParams& /*params*/) noexcept override {}
};

} // anonymous namespace

std::unique_ptr<BlurInferencer> createTensorRTBlurInferencer()
{
    return std::make_unique<StubBlurInferencer>();
}

