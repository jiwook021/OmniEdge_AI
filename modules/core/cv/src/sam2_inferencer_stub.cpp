#include "cv/onnx_sam2_inferencer.hpp"
#include "common/oe_tracy.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// StubSam2Inferencer — CPU-only stub for testing and CI without a GPU.
//
// Returns deterministic synthetic masks that depend on the prompt type and
// location, proving the pipeline wiring is correct without requiring ONNX
// Runtime or a GPU.
// ---------------------------------------------------------------------------


/// Stub implementation used when ONNX Runtime is not available.
/// Linked by omniedge_sam2 executable in non-ONNX builds.
class StubSam2InferencerImpl : public Sam2Inferencer {
public:
    void loadModel(const std::string& encoderPath,
                   const std::string& decoderPath) override
    {
        spdlog::info("[SAM2-Stub] loadModel({}, {})", encoderPath, decoderPath);
        loaded_ = true;
    }

    [[nodiscard]] tl::expected<void, std::string>
    encodeImage(const uint8_t* bgrFrame, uint32_t width, uint32_t height) override
    {
        if (!loaded_) return tl::unexpected(std::string("model not loaded"));
        if (bgrFrame == nullptr) return tl::unexpected(std::string("null frame"));
        width_  = width;
        height_ = height;
        firstByte_ = bgrFrame[0];
        encoded_ = true;
        return {};
    }

    [[nodiscard]] tl::expected<Sam2Result, std::string>
    segmentWithPrompt(const Sam2Prompt& prompt) override
    {
        if (!encoded_) return tl::unexpected(std::string("no image encoded"));

        Sam2Result result;
        result.maskWidth  = width_;
        result.maskHeight = height_;
        result.mask.resize(
            static_cast<std::size_t>(width_) * height_, 0);

        // Fill a simple region based on prompt type
        const uint32_t cx = width_ / 2;
        const uint32_t cy = height_ / 2;
        const uint32_t r  = std::min(width_, height_) / 6;

        if (prompt.type == Sam2PromptType::kPoint && !prompt.points.empty()) {
            const auto px = static_cast<uint32_t>(prompt.points[0].x * static_cast<float>(width_));
            const auto py = static_cast<uint32_t>(prompt.points[0].y * static_cast<float>(height_));
            fillCircle(result.mask, width_, height_, px, py, r);
        } else if (prompt.type == Sam2PromptType::kBox) {
            fillBox(result.mask, width_, height_, prompt.box);
        } else {
            fillCircle(result.mask, width_, height_, cx, cy, r);
        }

        result.iouScore  = 0.95f;
        result.stability = 0.90f;
        return result;
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame, uint32_t width, uint32_t height,
                 const Sam2Prompt& prompt, uint8_t* outBuf, std::size_t maxJpegBytes) override
    {
        auto enc = encodeImage(bgrFrame, width, height);
        if (!enc) return tl::unexpected(enc.error());

        auto seg = segmentWithPrompt(prompt);
        if (!seg) return tl::unexpected(seg.error());

        // Synthetic JPEG
        const std::size_t jpegSize = std::min<std::size_t>(
            static_cast<std::size_t>(width) * height * 3 / 10, maxJpegBytes);
        if (jpegSize < 4) return tl::unexpected(std::string("buffer too small"));

        outBuf[0] = 0xFF; outBuf[1] = 0xD8;
        const uint8_t fill = (bgrFrame != nullptr) ? bgrFrame[0] : 0x42;
        std::memset(outBuf + 2, fill, jpegSize - 4);
        outBuf[jpegSize - 2] = 0xFF; outBuf[jpegSize - 1] = 0xD9;
        return jpegSize;
    }

    void unload() noexcept override {
        loaded_ = false; encoded_ = false;
    }

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
        return 0;
    }

private:
    bool     loaded_    = false;
    bool     encoded_   = false;
    uint32_t width_     = 0;
    uint32_t height_    = 0;
    uint8_t  firstByte_ = 0;

    static void fillCircle(std::vector<uint8_t>& mask, uint32_t w, uint32_t h,
                           uint32_t cx, uint32_t cy, uint32_t radius)
    {
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                const auto dx = static_cast<int32_t>(x) - static_cast<int32_t>(cx);
                const auto dy = static_cast<int32_t>(y) - static_cast<int32_t>(cy);
                if (dx * dx + dy * dy <= static_cast<int32_t>(radius * radius)) {
                    mask[static_cast<std::size_t>(y) * w + x] = 255;
                }
            }
        }
    }

    static void fillBox(std::vector<uint8_t>& mask, uint32_t w, uint32_t h,
                        const Sam2BoxPrompt& box)
    {
        const auto x1 = static_cast<uint32_t>(box.x1 * static_cast<float>(w));
        const auto y1 = static_cast<uint32_t>(box.y1 * static_cast<float>(h));
        const auto x2 = static_cast<uint32_t>(box.x2 * static_cast<float>(w));
        const auto y2 = static_cast<uint32_t>(box.y2 * static_cast<float>(h));
        for (uint32_t y = y1; y < y2 && y < h; ++y) {
            for (uint32_t x = x1; x < x2 && x < w; ++x) {
                mask[static_cast<std::size_t>(y) * w + x] = 255;
            }
        }
    }
};

// The factory function in the stub build creates the stub implementation.
std::unique_ptr<Sam2Inferencer> createOnnxSam2Inferencer() {
    return std::make_unique<StubSam2InferencerImpl>();
}

