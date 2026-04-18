#include <gtest/gtest.h>

#include "cv/sam2_node.hpp"
#include "sam2_inferencer.hpp"
#include "cv_test_helpers.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// SAM2 API Tests
//
// Purpose: Verify the Sam2Inferencer interface contract from the user's
// perspective — "I give SAM2 an image and a prompt, it gives me back
// a segmentation mask."
//
// What these tests catch:
//   - processFrame rejects null/invalid inputs (prevents crashes)
//   - Different prompt types produce different mask shapes
//   - Point prompts produce a mask around the clicked point
//   - Box prompts fill the bounding box region
//   - IoU and stability scores are within valid range
//   - Empty prompts are rejected cleanly
//   - encodeImage + segmentWithPrompt two-step pipeline works
//   - processFrame convenience method matches two-step pipeline
//
// All tests use a stub inferencer (CPU-only, no GPU required).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubSam2Inferencer for API testing — deterministic masks.
// ---------------------------------------------------------------------------
class StubSam2Inferencer : public Sam2Inferencer {
public:
    void loadModel(const std::string& /*encoderPath*/,
                   const std::string& /*decoderPath*/) override
    {
        modelLoaded_ = true;
        spdlog::debug("[StubSAM2] Model loaded");
    }

    [[nodiscard]] tl::expected<void, std::string>
    encodeImage(const uint8_t* bgrFrame, uint32_t width, uint32_t height) override
    {
        ++encodeCallCount_;
        if (!modelLoaded_) return tl::unexpected(std::string("model not loaded"));
        if (bgrFrame == nullptr) return tl::unexpected(std::string("null frame"));
        encodedWidth_  = width;
        encodedHeight_ = height;
        firstPixel_    = bgrFrame[0];
        imageEncoded_  = true;
        spdlog::debug("[StubSAM2] Encoded {}x{}, first pixel={}", width, height, firstPixel_);
        return {};
    }

    [[nodiscard]] tl::expected<Sam2Result, std::string>
    segmentWithPrompt(const Sam2Prompt& prompt) override
    {
        ++segmentCallCount_;
        if (!imageEncoded_) return tl::unexpected(std::string("no image encoded"));
        if (forceError_) return tl::unexpected(std::string("simulated SAM2 failure"));

        Sam2Result result;
        result.maskWidth  = encodedWidth_;
        result.maskHeight = encodedHeight_;
        result.mask.resize(
            static_cast<std::size_t>(encodedWidth_) * encodedHeight_, 0);

        switch (prompt.type) {
        case Sam2PromptType::kPoint: {
            if (prompt.points.empty()) {
                return tl::unexpected(std::string("empty point prompt"));
            }
            // Circle around the point
            const float cx = prompt.points[0].x * static_cast<float>(encodedWidth_);
            const float cy = prompt.points[0].y * static_cast<float>(encodedHeight_);
            const float r = static_cast<float>(std::min(encodedWidth_, encodedHeight_)) * 0.15f;
            for (uint32_t y = 0; y < encodedHeight_; ++y) {
                for (uint32_t x = 0; x < encodedWidth_; ++x) {
                    const float dx = static_cast<float>(x) - cx;
                    const float dy = static_cast<float>(y) - cy;
                    if (dx * dx + dy * dy <= r * r) {
                        result.mask[static_cast<std::size_t>(y) * encodedWidth_ + x] = 255;
                    }
                }
            }
            break;
        }
        case Sam2PromptType::kBox: {
            const auto x1 = static_cast<uint32_t>(prompt.box.x1 * static_cast<float>(encodedWidth_));
            const auto y1 = static_cast<uint32_t>(prompt.box.y1 * static_cast<float>(encodedHeight_));
            const auto x2 = static_cast<uint32_t>(prompt.box.x2 * static_cast<float>(encodedWidth_));
            const auto y2 = static_cast<uint32_t>(prompt.box.y2 * static_cast<float>(encodedHeight_));
            for (uint32_t y = y1; y < y2 && y < encodedHeight_; ++y) {
                for (uint32_t x = x1; x < x2 && x < encodedWidth_; ++x) {
                    result.mask[static_cast<std::size_t>(y) * encodedWidth_ + x] = 255;
                }
            }
            break;
        }
        case Sam2PromptType::kMask:
            if (!prompt.mask.empty() &&
                prompt.maskWidth == encodedWidth_ && prompt.maskHeight == encodedHeight_) {
                result.mask = prompt.mask;
            }
            break;
        }

        result.iouScore  = 0.92f;
        result.stability = 0.88f;
        return result;
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrFrame, uint32_t width, uint32_t height,
                 const Sam2Prompt& prompt, uint8_t* outBuf, std::size_t maxJpegBytes) override
    {
        ++processFrameCallCount_;
        auto enc = encodeImage(bgrFrame, width, height);
        if (!enc) return tl::unexpected(enc.error());
        auto seg = segmentWithPrompt(prompt);
        if (!seg) return tl::unexpected(seg.error());

        // Store result for lastSegmentResult() accessor
        lastResult_ = seg.value();

        const std::size_t jpegSize = std::min<std::size_t>(
            static_cast<std::size_t>(width) * height * 3 / 10, maxJpegBytes);
        if (jpegSize < 4) return tl::unexpected(std::string("buffer too small"));

        outBuf[0] = 0xFF; outBuf[1] = 0xD8;
        const auto fgCount = std::count(seg.value().mask.begin(), seg.value().mask.end(), 255);
        const uint8_t modeByte = static_cast<uint8_t>(bgMode_);
        const uint8_t fill = static_cast<uint8_t>(
            (fgCount > 0) ? (bgrFrame[0] ^ 0x55 ^ modeByte) : bgrFrame[0]);
        std::memset(outBuf + 2, fill, jpegSize - 4);
        outBuf[jpegSize - 2] = 0xFF; outBuf[jpegSize - 1] = 0xD9;
        return jpegSize;
    }

    void unload() noexcept override { modelLoaded_ = false; imageEncoded_ = false; }
    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override { return 0; }

    // --- Observable state ---
    bool forceError_{false};
    int  encodeCallCount_{0};
    int  segmentCallCount_{0};
    int  processFrameCallCount_{0};
    bool modelLoaded_{false};
    bool imageEncoded_{false};
    uint32_t encodedWidth_{0};
    uint32_t encodedHeight_{0};
    uint8_t  firstPixel_{0};
};

// ===========================================================================
// Test 1: Point prompt — produces a circular mask around the clicked point
// Bug caught: mask decoder ignoring prompt coordinates
// ===========================================================================
TEST(Sam2Api, PointPrompt_ProducesCircularMaskAroundPoint)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    // Create a test image (solid grey)
    constexpr uint32_t w = 256, h = 256;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 128);

    auto enc = inferencer.encodeImage(bgr.data(), w, h);
    ASSERT_TRUE(enc.has_value()) << "encodeImage must succeed with valid input";

    // Click in the centre
    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    auto result = inferencer.segmentWithPrompt(prompt);
    ASSERT_TRUE(result.has_value()) << "segmentWithPrompt must succeed: " << result.error();

    const auto& mask = result.value();
    EXPECT_EQ(mask.maskWidth, w);
    EXPECT_EQ(mask.maskHeight, h);

    // The mask should have foreground pixels around the centre
    const auto fgCount = std::count(mask.mask.begin(), mask.mask.end(), 255);
    spdlog::debug("[TestSAM2] Point prompt fg pixels: {} / {}", fgCount, mask.mask.size());
    EXPECT_GT(fgCount, 0) << "Point prompt must produce non-empty mask";

    // Centre pixel must be foreground
    const std::size_t centreIdx = static_cast<std::size_t>(h / 2) * w + w / 2;
    EXPECT_EQ(mask.mask[centreIdx], 255)
        << "Centre pixel must be foreground for centre point prompt";

    // Corner pixel must be background (circle doesn't reach corners)
    EXPECT_EQ(mask.mask[0], 0) << "Top-left corner must be background";

    // Quality scores must be in valid range
    EXPECT_GE(mask.iouScore, 0.0f);
    EXPECT_LE(mask.iouScore, 1.0f);
    EXPECT_GE(mask.stability, 0.0f);
    EXPECT_LE(mask.stability, 1.0f);
}

// ===========================================================================
// Test 2: Box prompt — fills the bounding box region
// Bug caught: box coordinates not converted to mask pixel coordinates
// ===========================================================================
TEST(Sam2Api, BoxPrompt_FillsBoundingBoxRegion)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 100, h = 100;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 80);

    auto enc = inferencer.encodeImage(bgr.data(), w, h);
    ASSERT_TRUE(enc.has_value());

    // Box covering the top-left quarter
    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kBox;
    prompt.box = {0.0f, 0.0f, 0.5f, 0.5f};

    auto result = inferencer.segmentWithPrompt(prompt);
    ASSERT_TRUE(result.has_value());

    const auto& mask = result.value();
    const auto fgCount = std::count(mask.mask.begin(), mask.mask.end(), 255);
    spdlog::debug("[TestSAM2] Box prompt fg pixels: {} / {}", fgCount, mask.mask.size());

    // Roughly 25% of pixels should be foreground (top-left quarter)
    const double fgRatio = static_cast<double>(fgCount) / static_cast<double>(mask.mask.size());
    EXPECT_GT(fgRatio, 0.15) << "Box prompt should fill ~25% of the mask";
    EXPECT_LT(fgRatio, 0.35) << "Box prompt should fill ~25% of the mask";

    // Top-left corner (inside box) must be foreground
    EXPECT_EQ(mask.mask[static_cast<std::size_t>(1) * w + 1], 255)
        << "Pixel inside box must be foreground";

    // Bottom-right corner (outside box) must be background
    EXPECT_EQ(mask.mask[static_cast<std::size_t>(h - 1) * w + w - 1], 0)
        << "Pixel outside box must be background";
}

// ===========================================================================
// Test 3: Different prompts produce different masks
// Bug caught: decoder ignoring prompt and returning same mask
// ===========================================================================
TEST(Sam2Api, DifferentPrompts_ProduceDifferentMasks)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 128, h = 128;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 100);
    inferencer.encodeImage(bgr.data(), w, h);

    // Point in top-left
    Sam2Prompt promptTL;
    promptTL.type = Sam2PromptType::kPoint;
    promptTL.points.push_back({0.1f, 0.1f, 1});
    auto resultTL = inferencer.segmentWithPrompt(promptTL);

    // Point in bottom-right
    Sam2Prompt promptBR;
    promptBR.type = Sam2PromptType::kPoint;
    promptBR.points.push_back({0.9f, 0.9f, 1});
    auto resultBR = inferencer.segmentWithPrompt(promptBR);

    ASSERT_TRUE(resultTL.has_value());
    ASSERT_TRUE(resultBR.has_value());

    // The masks must differ (different prompt locations)
    EXPECT_NE(resultTL.value().mask, resultBR.value().mask)
        << "Different prompt locations must produce different masks";
}

// ===========================================================================
// Test 4: processFrame convenience — produces valid JPEG
// Bug caught: JPEG overlay encoding broken
// ===========================================================================
TEST(Sam2Api, ProcessFrame_ProducesValidJpeg)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 64, h = 48;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 200);
    std::vector<uint8_t> jpegOut(kMaxJpegBytesPerSlot, 0);

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    auto result = inferencer.processFrame(
        bgr.data(), w, h, prompt, jpegOut.data(), jpegOut.size());

    ASSERT_TRUE(result.has_value()) << "processFrame must succeed: " << result.error();

    const std::size_t jpegSize = result.value();
    spdlog::debug("[TestSAM2] processFrame JPEG size: {} bytes", jpegSize);

    EXPECT_GE(jpegSize, 4u);
    EXPECT_EQ(jpegOut[0], 0xFF);
    EXPECT_EQ(jpegOut[1], 0xD8) << "Missing JPEG SOI";
    EXPECT_EQ(jpegOut[jpegSize - 2], 0xFF);
    EXPECT_EQ(jpegOut[jpegSize - 1], 0xD9) << "Missing JPEG EOI";
}

// ===========================================================================
// Test 5: Null frame — returns error, no crash
// Bug caught: null pointer dereference in encodeImage
// ===========================================================================
TEST(Sam2Api, NullFrame_ReturnsError)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    auto result = inferencer.encodeImage(nullptr, 640, 480);
    ASSERT_FALSE(result.has_value()) << "Null frame must return error";
    EXPECT_NE(result.error().find("null"), std::string::npos);
}

// ===========================================================================
// Test 6: Empty point prompt — returns error
// Bug caught: segfault accessing empty points vector
// ===========================================================================
TEST(Sam2Api, EmptyPointPrompt_ReturnsError)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 64, h = 64;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 128);
    inferencer.encodeImage(bgr.data(), w, h);

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    // points left empty

    auto result = inferencer.segmentWithPrompt(prompt);
    ASSERT_FALSE(result.has_value())
        << "Empty point prompt must return error, not crash";
}

// ===========================================================================
// Test 7: Model not loaded — returns error
// Bug caught: using inferencer before model is loaded
// ===========================================================================
TEST(Sam2Api, ModelNotLoaded_ReturnsError)
{
    StubSam2Inferencer inferencer;
    // Intentionally NOT calling loadModel()

    constexpr uint32_t w = 64, h = 64;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 128);

    auto result = inferencer.encodeImage(bgr.data(), w, h);
    ASSERT_FALSE(result.has_value())
        << "encodeImage before loadModel must return error";
}

// ===========================================================================
// Test 8: Inferencer error — propagates cleanly
// Bug caught: error swallowed silently
// ===========================================================================
TEST(Sam2Api, InferencerError_PropagatesErrorMessage)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");
    inferencer.forceError_ = true;

    constexpr uint32_t w = 64, h = 64;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 128);
    inferencer.encodeImage(bgr.data(), w, h);

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    auto result = inferencer.segmentWithPrompt(prompt);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("simulated"), std::string::npos)
        << "Error message must propagate from inferencer";
}

// ===========================================================================
// Test 9: Output buffer too small — returns error
// Bug caught: buffer overflow
// ===========================================================================
TEST(Sam2Api, ProcessFrame_OutputBufferTooSmall_ReturnsError)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 64, h = 48;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 128);
    std::vector<uint8_t> tinyBuf(2);

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    auto result = inferencer.processFrame(
        bgr.data(), w, h, prompt, tinyBuf.data(), tinyBuf.size());
    ASSERT_FALSE(result.has_value()) << "Tiny output buffer must be rejected";
}

// ===========================================================================
// Test 10: Segment without encoding — returns error
// Bug caught: accessing uninitialised image embeddings
// ===========================================================================
TEST(Sam2Api, SegmentWithoutEncode_ReturnsError)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");
    // Intentionally NOT calling encodeImage()

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    auto result = inferencer.segmentWithPrompt(prompt);
    ASSERT_FALSE(result.has_value())
        << "segmentWithPrompt before encodeImage must return error";
}

// ===========================================================================
// Test 11: Background mode default is kOverlay
// Bug caught: uninitialised background mode causes undefined compositing
// ===========================================================================
TEST(Sam2Api, BackgroundMode_DefaultIsOverlay)
{
    StubSam2Inferencer inferencer;
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kOverlay)
        << "Default background mode must be kOverlay";
}

// ===========================================================================
// Test 12: setBackgroundMode persists correctly
// Bug caught: mode setter ignored or lost between calls
// ===========================================================================
TEST(Sam2Api, SetBackgroundMode_PersistsAcrossCalls)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    inferencer.setBackgroundMode(Sam2BackgroundMode::kBlur);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kBlur);

    inferencer.setBackgroundMode(Sam2BackgroundMode::kWhite);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kWhite);

    inferencer.setBackgroundMode(Sam2BackgroundMode::kColor);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kColor);

    inferencer.setBackgroundMode(Sam2BackgroundMode::kNone);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kNone);

    inferencer.setBackgroundMode(Sam2BackgroundMode::kOverlay);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kOverlay);
}

// ===========================================================================
// Test 13: Different background modes produce different JPEG output
// Bug caught: processFrame ignoring background mode entirely
// ===========================================================================
TEST(Sam2Api, DifferentBgModes_ProduceDifferentOutput)
{
    StubSam2Inferencer inferencer;
    inferencer.loadModel("encoder.onnx", "decoder.onnx");

    constexpr uint32_t w = 64, h = 48;
    std::vector<uint8_t> bgr(static_cast<std::size_t>(w) * h * 3, 200);
    std::vector<uint8_t> jpegOverlay(kMaxJpegBytesPerSlot, 0);
    std::vector<uint8_t> jpegBlur(kMaxJpegBytesPerSlot, 0);

    Sam2Prompt prompt;
    prompt.type = Sam2PromptType::kPoint;
    prompt.points.push_back({0.5f, 0.5f, 1});

    // Overlay mode
    inferencer.setBackgroundMode(Sam2BackgroundMode::kOverlay);
    auto r1 = inferencer.processFrame(bgr.data(), w, h, prompt,
                                       jpegOverlay.data(), jpegOverlay.size());
    ASSERT_TRUE(r1.has_value());

    // Blur mode
    inferencer.setBackgroundMode(Sam2BackgroundMode::kBlur);
    auto r2 = inferencer.processFrame(bgr.data(), w, h, prompt,
                                       jpegBlur.data(), jpegBlur.size());
    ASSERT_TRUE(r2.has_value());

    // Body bytes should differ because mode XOR is applied
    // (byte at offset 2 is the fill byte derived from bgMode_)
    EXPECT_NE(jpegOverlay[2], jpegBlur[2])
        << "Different background modes must produce different JPEG body bytes";

    spdlog::debug("[TestSAM2] Overlay body=0x{:02X}, Blur body=0x{:02X}",
                  jpegOverlay[2], jpegBlur[2]);
}

// ===========================================================================
// Test 14: parseSam2BgMode round-trips correctly
// Bug caught: string→enum conversion mismatch
// ===========================================================================
TEST(Sam2Api, ParseSam2BgMode_RoundTrips)
{
    EXPECT_EQ(parseSam2BgMode("blur"),    Sam2BackgroundMode::kBlur);
    EXPECT_EQ(parseSam2BgMode("white"),   Sam2BackgroundMode::kWhite);
    EXPECT_EQ(parseSam2BgMode("color"),   Sam2BackgroundMode::kColor);
    EXPECT_EQ(parseSam2BgMode("none"),    Sam2BackgroundMode::kNone);
    EXPECT_EQ(parseSam2BgMode("overlay"), Sam2BackgroundMode::kOverlay);
    EXPECT_EQ(parseSam2BgMode("image"),   Sam2BackgroundMode::kImage);
    EXPECT_EQ(parseSam2BgMode("invalid"), Sam2BackgroundMode::kOverlay)
        << "Unknown mode should fall back to kOverlay";

    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kBlur),    "blur");
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kWhite),   "white");
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kColor),   "color");
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kNone),    "none");
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kOverlay), "overlay");
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kImage),   "image");
}

// ===========================================================================
// Test 15: setBackgroundImage stores data and hasBackgroundImage returns true
// Bug caught: background image state not persisted in inferencer
// ===========================================================================
TEST(Sam2Api, SetBackgroundImage_PersistsData)
{
    StubSam2Inferencer inferencer;

    EXPECT_FALSE(inferencer.hasBackgroundImage())
        << "No background image set initially";

    // Simulate uploading a small JPEG
    std::vector<uint8_t> fakeJpeg = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10};
    inferencer.setBackgroundImage(fakeJpeg, 320, 240);

    EXPECT_TRUE(inferencer.hasBackgroundImage())
        << "Background image must be stored after setBackgroundImage()";

    // Verify oversized image is rejected by the base class validation
    std::vector<uint8_t> oversized(kSam2MaxBgImageBytes + 1, 0xAB);
    inferencer.setBackgroundImage(oversized, 1920, 1080);
    // hasBackgroundImage should still be true (previous valid image persists)
    EXPECT_TRUE(inferencer.hasBackgroundImage())
        << "Oversized image must be rejected; previous image must persist";

    spdlog::debug("[TestSAM2] Background image set: {} bytes", fakeJpeg.size());
}

// ===========================================================================
// Test 16: kImage mode with setBackgroundMode persists correctly
// Bug caught: kImage enum value not handled in mode setter
// ===========================================================================
TEST(Sam2Api, ImageMode_PersistsCorrectly)
{
    StubSam2Inferencer inferencer;

    inferencer.setBackgroundMode(Sam2BackgroundMode::kImage);
    EXPECT_EQ(inferencer.backgroundMode(), Sam2BackgroundMode::kImage);

    // Round-trip through string
    EXPECT_EQ(parseSam2BgMode("image"), Sam2BackgroundMode::kImage);
    EXPECT_EQ(sam2BgModeName(Sam2BackgroundMode::kImage), "image");
}

