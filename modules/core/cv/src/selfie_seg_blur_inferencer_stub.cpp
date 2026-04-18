#include "blur_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <memory>


namespace {

class StubBlurInferencer final : public BlurInferencer {
public:
    void loadEngine(const std::string& /*modelPath*/,
                    uint32_t           /*inputWidth*/,
                    uint32_t           /*inputHeight*/) override
    {
        OE_LOG_WARN("blur_stub_load: MediaPipe Selfie Seg inferencer not built — using stub");
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
            std::string("Blur inferencer not available: built without ONNX Runtime"));
    }

    void unload() noexcept override {}

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
    {
        return 0;
    }

    void setIspParams(const IspParams& /*params*/) noexcept override {}
};

} // anonymous namespace


std::unique_ptr<BlurInferencer> createBlurInferencer()
{
    return std::make_unique<StubBlurInferencer>();
}
