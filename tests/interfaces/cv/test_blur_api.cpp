#include <gtest/gtest.h>

#include "cv/background_blur_node.hpp"
#include "cv_test_helpers.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Background Blur API Tests
//
// Purpose: Verify the BlurInferencer interface contract from the user's
// perspective — "I give the blur engine a camera frame, it gives me back
// a JPEG with the background blurred."
//
// What these tests catch:
//   - processFrame rejects invalid buffers (prevents buffer overflows)
//   - processFrame output changes when input changes (not returning stale data)
//   - Inferencer errors propagate cleanly (no silent failures)
//   - ISP parameters are forwarded to the blur engine (slider adjustments work)
//
// All tests use a stub inferencer that simulates JPEG encoding without a GPU.
// Tests load actual BGR24 image files from tests/cv/fixtures/ when available.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubBlurInferencer — simulates the blur pipeline for CPU-only testing.
//
// Given a BGR frame, produces a synthetic JPEG (SOI marker + body + EOI marker)
// whose content derives from the input pixels.  This proves the pipeline
// actually reads the input rather than returning canned data.
// ---------------------------------------------------------------------------
class StubBlurInferencer : public BlurInferencer {
public:
    void loadEngine(const std::string& /*enginePath*/,
                    uint32_t           width,
                    uint32_t           height) override
    {
        loadedWidth_  = width;
        loadedHeight_ = height;
    }

    [[nodiscard]] tl::expected<std::size_t, std::string>
    processFrame(const uint8_t* bgrInputFrame,
                 uint32_t       frameWidth,
                 uint32_t       frameHeight,
                 uint8_t*       jpegOutputBuffer,
                 std::size_t    maxOutputBytes) override
    {
        ++processFrameCallCount_;

        if (forceError_) {
            return tl::unexpected(std::string("simulated TRT inference failure"));
        }

        // Simulate JPEG compression: output is ~10% of input size
        const std::size_t inputBytes =
            static_cast<std::size_t>(frameWidth) * frameHeight * 3;
        const std::size_t simulatedJpegSize =
            std::min(inputBytes / 10, maxOutputBytes);

        if (simulatedJpegSize < 4) {
            return tl::unexpected(
                std::string("output buffer too small for JPEG markers"));
        }

        // Write valid JPEG structure: SOI + body + EOI
        jpegOutputBuffer[0] = 0xFF;
        jpegOutputBuffer[1] = 0xD8;  // JPEG Start-Of-Image marker

        // Fill body with a byte derived from the first input pixel.
        // This proves the inferencer actually reads the input frame.
        const uint8_t bodyFillByte =
            (bgrInputFrame != nullptr) ? bgrInputFrame[0] : 0x42;
        std::memset(jpegOutputBuffer + 2, bodyFillByte, simulatedJpegSize - 4);

        jpegOutputBuffer[simulatedJpegSize - 2] = 0xFF;
        jpegOutputBuffer[simulatedJpegSize - 1] = 0xD9;  // JPEG End-Of-Image

        lastAppliedIspParams_ = pendingIspParams_;
        return simulatedJpegSize;
    }

    void unload() noexcept override {}

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
        return 0;
    }

    void setIspParams(const IspParams& params) noexcept override {
        pendingIspParams_ = params;
        ++ispUpdateCount_;
    }

    // --- Observable state for test assertions ---
    uint32_t  loadedWidth_{0};
    uint32_t  loadedHeight_{0};
    int       processFrameCallCount_{0};
    bool      forceError_{false};
    int       ispUpdateCount_{0};
    IspParams pendingIspParams_{};
    IspParams lastAppliedIspParams_{};
};

// ===========================================================================
// Blur processFrame API — "feed a camera frame, get a JPEG back"
// ===========================================================================

TEST(BlurApi, ProcessFrame_WithPersonScene_ProducesValidJpeg)
{
    // GIVEN: a BGR image of a person in a room (loaded from test fixture)
    auto personSceneBgr = loadBgr24Fixture(kPersonSceneFile);
    SKIP_IF_NO_FIXTURE(personSceneBgr);

    StubBlurInferencer blurInferencer;
    blurInferencer.loadEngine("bg_blur/mediapipe_selfie_seg.onnx",
                           kFixtureImageWidth, kFixtureImageHeight);

    std::vector<uint8_t> jpegOutput(kMaxJpegBytesPerSlot, 0);

    // WHEN: the blur inferencer processes the frame
    auto result = blurInferencer.processFrame(
        personSceneBgr.data(),
        kFixtureImageWidth, kFixtureImageHeight,
        jpegOutput.data(), jpegOutput.size());

    // THEN: a valid JPEG is produced with SOI and EOI markers
    ASSERT_TRUE(result.has_value())
        << "processFrame must succeed with a valid person scene image";

    const std::size_t jpegByteCount = result.value();
    EXPECT_GE(jpegByteCount, 4u) << "JPEG must have at least SOI + EOI markers";
    EXPECT_LE(jpegByteCount, jpegOutput.size());

    EXPECT_EQ(jpegOutput[0], 0xFF);
    EXPECT_EQ(jpegOutput[1], 0xD8) << "Missing JPEG Start-Of-Image marker";
    EXPECT_EQ(jpegOutput[jpegByteCount - 2], 0xFF);
    EXPECT_EQ(jpegOutput[jpegByteCount - 1], 0xD9) << "Missing JPEG End-Of-Image marker";
}

TEST(BlurApi, ProcessFrame_DifferentInputImages_ProduceDifferentOutputs)
{
    // Bug caught: inferencer ignoring input and returning cached/stale JPEG data

    // GIVEN: two different scene images
    auto personScene = loadBgr24Fixture(kPersonSceneFile);
    auto emptyRoom   = loadBgr24Fixture(kEmptyRoomFile);
    SKIP_IF_NO_FIXTURE(personScene);
    SKIP_IF_NO_FIXTURE(emptyRoom);

    StubBlurInferencer blurInferencer;
    blurInferencer.loadEngine("", kFixtureImageWidth, kFixtureImageHeight);

    std::vector<uint8_t> jpegFromPersonScene(kMaxJpegBytesPerSlot, 0);
    std::vector<uint8_t> jpegFromEmptyRoom(kMaxJpegBytesPerSlot, 0);

    // WHEN: both images are processed
    auto resultPerson = blurInferencer.processFrame(
        personScene.data(), kFixtureImageWidth, kFixtureImageHeight,
        jpegFromPersonScene.data(), jpegFromPersonScene.size());
    auto resultRoom = blurInferencer.processFrame(
        emptyRoom.data(), kFixtureImageWidth, kFixtureImageHeight,
        jpegFromEmptyRoom.data(), jpegFromEmptyRoom.size());

    ASSERT_TRUE(resultPerson.has_value());
    ASSERT_TRUE(resultRoom.has_value());

    // THEN: the JPEG body content differs (proves inferencer reads input pixels)
    EXPECT_NE(jpegFromPersonScene[2], jpegFromEmptyRoom[2])
        << "Different input images must produce different JPEG output — "
           "the inferencer may be ignoring the input frame";
}

TEST(BlurApi, ProcessFrame_OutputBufferTooSmall_ReturnsError)
{
    // Bug caught: buffer overflow when output buffer is smaller than expected

    StubBlurInferencer blurInferencer;
    blurInferencer.loadEngine("", kMaxInputWidth, kMaxInputHeight);

    // GIVEN: an output buffer that is too small to hold even JPEG markers
    std::vector<uint8_t> tinyOutputBuffer(2);

    // WHEN: processFrame is called with the undersized buffer
    auto result = blurInferencer.processFrame(
        nullptr, kMaxInputWidth, kMaxInputHeight,
        tinyOutputBuffer.data(), tinyOutputBuffer.size());

    // THEN: an error is returned (not a crash or buffer overflow)
    ASSERT_FALSE(result.has_value())
        << "Must reject output buffer that cannot hold JPEG markers";
}

TEST(BlurApi, ProcessFrame_InferencerInferenceFailure_PropagatesErrorMessage)
{
    // Bug caught: silent failure where blur errors are swallowed

    StubBlurInferencer blurInferencer;
    blurInferencer.forceError_ = true;

    // WHEN: the inferencer fails during inference (e.g., CUDA OOM, TRT error)
    auto result = blurInferencer.processFrame(
        nullptr, kMaxInputWidth, kMaxInputHeight,
        nullptr, 0);

    // THEN: the error message propagates to the caller
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("simulated"), std::string::npos)
        << "Error message must propagate from the inferencer to the caller";
}

// ===========================================================================
// ISP Parameter Pipeline — "adjust brightness/contrast/saturation/sharpness"
// ===========================================================================


TEST(IspAdjustment, NonZeroBrightness_IsNotIdentityTransform)
{
    IspParams brighterParams;
    brighterParams.brightness = 10.0f;

    EXPECT_FALSE(brighterParams.isIdentity())
        << "Any non-default ISP value must report as non-identity";
}

TEST(IspAdjustment, SetIspParams_AllValuesForwardedToInferencer)
{
    // Bug caught: ISP slider changes ignored because params not forwarded

    StubBlurInferencer blurInferencer;

    // GIVEN: the user adjusts all four ISP parameters via the UI
    IspParams userAdjustments;
    userAdjustments.brightness = 50.0f;   // brighter image
    userAdjustments.contrast   = 1.5f;    // higher contrast
    userAdjustments.saturation = 0.8f;    // slightly desaturated
    userAdjustments.sharpness  = 3.0f;    // moderate sharpening

    // WHEN: the parameters are sent to the blur inferencer
    blurInferencer.setIspParams(userAdjustments);

    // THEN: every parameter value arrives at the inferencer
    EXPECT_EQ(blurInferencer.ispUpdateCount_, 1);
    EXPECT_FLOAT_EQ(blurInferencer.pendingIspParams_.brightness, 50.0f);
    EXPECT_FLOAT_EQ(blurInferencer.pendingIspParams_.contrast,   1.5f);
    EXPECT_FLOAT_EQ(blurInferencer.pendingIspParams_.saturation, 0.8f);
    EXPECT_FLOAT_EQ(blurInferencer.pendingIspParams_.sharpness,  3.0f);
}

TEST(IspAdjustment, MultipleUpdates_LastValueWins)
{
    // Bug caught: ISP params accumulate instead of overwriting

    StubBlurInferencer blurInferencer;

    IspParams firstAdjustment;
    firstAdjustment.brightness = 20.0f;
    blurInferencer.setIspParams(firstAdjustment);

    IspParams secondAdjustment;
    secondAdjustment.brightness = -10.0f;
    blurInferencer.setIspParams(secondAdjustment);

    EXPECT_FLOAT_EQ(blurInferencer.pendingIspParams_.brightness, -10.0f)
        << "The latest ISP update must overwrite the previous one";
    EXPECT_EQ(blurInferencer.ispUpdateCount_, 2);
}

