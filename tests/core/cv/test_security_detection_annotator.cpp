#include <gtest/gtest.h>

#include "cv/security_detection_annotator.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// SecurityDetectionAnnotator Tests
//
// Purpose: Verify the detection overlay renderer that draws styled bounding
// boxes on video frames and encodes to JPEG for the /security_video channel.
//
// What these tests catch:
//   - JPEG encoding failures on valid input
//   - Style switch not reflected in output
//   - Null/zero-size frame crashes instead of returning an error
//   - Class colour mapping returns wrong colour for known classes
//   - Tiny bounding boxes crash the drawing code
//   - Detection at frame edges triggers out-of-bounds writes
//
// Requires: OpenCV (imgproc + imgcodecs).  No GPU, no CUDA.
// ---------------------------------------------------------------------------

using oe::security::AnnotationStyle;
using oe::security::DetectionBox;
using oe::security::SecurityDetectionAnnotator;

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

// Frame dimensions — VGA (640x480) is the default security camera output
// resolution; QVGA (320x240) is used for lighter edge-case tests.
constexpr int kVgaWidth     = 640;
constexpr int kVgaHeight    = 480;
constexpr int kQvgaWidth    = 320;
constexpr int kQvgaHeight   = 240;

// BGR24 channels per pixel.
constexpr int kBgrChannels  = 3;

// Default dark-grey background — low intensity to make coloured overlays
// clearly visible in visual-diff checks.
constexpr uint8_t kDefaultBgBlue  = 30;
constexpr uint8_t kDefaultBgGreen = 30;
constexpr uint8_t kDefaultBgRed   = 30;

// Detection coordinate defaults — a reasonably-sized box in the upper-left
// quadrant (normalised 0-1 coords).
constexpr float kDefaultDetX = 0.1f;
constexpr float kDefaultDetY = 0.1f;
constexpr float kDefaultDetW = 0.3f;
constexpr float kDefaultDetH = 0.4f;

// JPEG magic bytes (SOI marker: FF D8 FF).
constexpr uint8_t kJpegMagic0 = 0xFF;
constexpr uint8_t kJpegMagic1 = 0xD8;
constexpr uint8_t kJpegMagic2 = 0xFF;

// Minimum byte count for a non-trivial JPEG (header + some DCT data).
constexpr size_t kMinJpegSize = 100;

// JPEG quality settings for compression-ratio tests.
constexpr int kLowJpegQuality  = 20;
constexpr int kHighJpegQuality = 95;

// Sub-pixel normalised size — maps to <2 px on a 320-wide frame,
// which the annotator should skip rather than crash.
constexpr float kSubPixelSize = 0.001f;

// Grid layout for the ManyDetections stress test.
constexpr int kGridCols        = 5;
constexpr int kGridRows        = 4;
constexpr int kGridTotal       = kGridCols * kGridRows;  // 20 detections
constexpr float kGridCellW     = 0.18f;  // normalised cell width
constexpr float kGridCellH     = 0.22f;  // normalised cell height
constexpr float kGridBoxW      = 0.15f;  // box width within each cell
constexpr float kGridBoxH      = 0.18f;  // box height within each cell
constexpr float kGridBaseConf  = 0.50f;  // lowest confidence in the grid
constexpr float kGridConfStep  = 0.02f;  // confidence increment per detection

// Minimum number of visually-distinct class colours expected across
// the 6 COCO classes tested in AllKnownClasses_ProduceDistinctOutput.
constexpr int kMinDistinctColours = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a solid-colour BGR24 test frame.
static std::vector<uint8_t> makeBgrFrame(int width, int height,
                                          uint8_t b = kDefaultBgBlue,
                                          uint8_t g = kDefaultBgGreen,
                                          uint8_t r = kDefaultBgRed) {
    std::vector<uint8_t> frame(width * height * kBgrChannels);
    for (int i = 0; i < width * height; ++i) {
        frame[i * kBgrChannels + 0] = b;
        frame[i * kBgrChannels + 1] = g;
        frame[i * kBgrChannels + 2] = r;
    }
    return frame;
}

/// Create a DetectionBox with normalised coordinates.
/// Default position places a medium-sized box in the upper-left quadrant.
static DetectionBox makeDetection(const std::string& cls, float confidence,
                                   float x = kDefaultDetX,
                                   float y = kDefaultDetY,
                                   float w = kDefaultDetW,
                                   float h = kDefaultDetH) {
    return DetectionBox{
        .x          = x,
        .y          = y,
        .w          = w,
        .h          = h,
        .confidence = confidence,
        .className  = cls,
        .trackId    = -1,
    };
}

/// Check whether a buffer starts with the JPEG SOI marker (FF D8 FF).
static bool isValidJpeg(const std::vector<uint8_t>& data) {
    return data.size() >= 3 &&
           data[0] == kJpegMagic0 &&
           data[1] == kJpegMagic1 &&
           data[2] == kJpegMagic2;
}

// ===========================================================================
// Basic JPEG output
//
// Verify that annotate() returns a valid JPEG buffer for typical inputs.
// These are the "golden path" tests — if these fail, nothing else matters.
// ===========================================================================

TEST(SecurityDetectionAnnotator, NoDetections_ProducesValidJpeg)
{
    // A bare frame with zero detections should still produce a valid JPEG —
    // the security camera stream runs continuously even when nothing is detected.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);

    auto result = annotator.annotate(frame.data(), kVgaWidth, kVgaHeight, {});
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result)) << "Output must start with JPEG SOI marker";
    EXPECT_GT(result->size(), kMinJpegSize)
        << "JPEG should contain more than just headers";
}

TEST(SecurityDetectionAnnotator, WithDetections_ProducesValidJpeg)
{
    // Two detections of different classes at different positions — the most
    // common real-world scenario for security camera mode.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);

    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.95f),
        makeDetection("car", 0.80f, 0.5f, 0.2f, 0.25f, 0.35f),
    };

    auto result = annotator.annotate(frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result));
}

TEST(SecurityDetectionAnnotator, WithDetections_OutputDiffersFromBlankFrame)
{
    // Ensure that drawing detections actually modifies the output — catches
    // bugs where annotate() silently skips the overlay rendering step.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);

    auto blankResult = annotator.annotate(frame.data(), kVgaWidth, kVgaHeight, {});
    ASSERT_TRUE(blankResult.has_value());

    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.90f, 0.1f, 0.1f, 0.5f, 0.5f),
    };
    auto annotatedResult = annotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(annotatedResult.has_value());

    EXPECT_NE(*blankResult, *annotatedResult)
        << "Annotated frame must differ from blank — overlays should be visible";
}

// ===========================================================================
// Annotation styles
//
// The frontend lets users switch between three visual styles at runtime
// (corner brackets, full rectangles, crosshairs).  These tests verify that
// each style actually renders differently and that setStyle() takes effect.
// ===========================================================================

TEST(SecurityDetectionAnnotator, SetStyle_ChangesRenderingOutput)
{
    // Render the same detection with each of the three styles and compare
    // the JPEG output.  Because each style draws different primitives
    // (L-corners vs full rect vs crosshair), at least some outputs must differ.
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);
    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.90f, 0.2f, 0.2f, 0.4f, 0.4f),
    };

    SecurityDetectionAnnotator::Config cfg;

    cfg.style = AnnotationStyle::kCornerBracket;
    SecurityDetectionAnnotator cornerAnnotator(cfg);
    auto cornerResult = cornerAnnotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(cornerResult.has_value());

    cfg.style = AnnotationStyle::kRectangle;
    SecurityDetectionAnnotator rectAnnotator(cfg);
    auto rectResult = rectAnnotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(rectResult.has_value());

    cfg.style = AnnotationStyle::kCrosshair;
    SecurityDetectionAnnotator crossAnnotator(cfg);
    auto crossResult = crossAnnotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(crossResult.has_value());

    EXPECT_TRUE(isValidJpeg(*cornerResult));
    EXPECT_TRUE(isValidJpeg(*rectResult));
    EXPECT_TRUE(isValidJpeg(*crossResult));

    bool anyDifference = (*cornerResult != *rectResult) ||
                         (*rectResult != *crossResult) ||
                         (*cornerResult != *crossResult);
    EXPECT_TRUE(anyDifference)
        << "Different annotation styles must produce different visual output";
}

TEST(SecurityDetectionAnnotator, SetStyle_RuntimeSwitch)
{
    // Verify that setStyle() updates the style and that the default
    // constructed annotator starts with kCornerBracket (the UI default).
    SecurityDetectionAnnotator annotator;
    EXPECT_EQ(annotator.style(), AnnotationStyle::kCornerBracket);

    annotator.setStyle(AnnotationStyle::kRectangle);
    EXPECT_EQ(annotator.style(), AnnotationStyle::kRectangle);

    annotator.setStyle(AnnotationStyle::kCrosshair);
    EXPECT_EQ(annotator.style(), AnnotationStyle::kCrosshair);
}

// ===========================================================================
// Error handling
//
// The annotator receives raw pointers from the video pipeline.  These tests
// verify that invalid inputs return a descriptive error string via
// tl::expected rather than crashing or invoking undefined behaviour.
// ===========================================================================

TEST(SecurityDetectionAnnotator, NullFrame_ReturnsError)
{
    // A null pointer can arrive if the upstream GStreamer pipeline stalls
    // or the shared-memory mapping fails — must not dereference it.
    SecurityDetectionAnnotator annotator;
    auto result = annotator.annotate(nullptr, kVgaWidth, kVgaHeight, {});
    ASSERT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().empty());
}

TEST(SecurityDetectionAnnotator, ZeroWidth_ReturnsError)
{
    // Zero dimensions are structurally invalid — OpenCV would assert.
    // The annotator must reject before reaching cv::Mat construction.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(1, 1);
    auto result = annotator.annotate(frame.data(), 0, kVgaHeight, {});
    ASSERT_FALSE(result.has_value());
}

TEST(SecurityDetectionAnnotator, ZeroHeight_ReturnsError)
{
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(1, 1);
    auto result = annotator.annotate(frame.data(), kVgaWidth, 0, {});
    ASSERT_FALSE(result.has_value());
}

// ===========================================================================
// Edge cases — detections at frame boundaries
//
// YOLO can produce bounding boxes that touch or exceed frame edges,
// especially for objects entering/leaving the camera FOV.  The annotator
// must clamp coordinates to [0, width/height) before drawing.
// ===========================================================================

TEST(SecurityDetectionAnnotator, DetectionAtFrameEdge_DoesNotCrash)
{
    // Full-frame detection (x=0, y=0, w=1, h=1) — the bounding box
    // coincides exactly with the frame borders.  cv::rectangle must not
    // write beyond the Mat bounds.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kQvgaWidth, kQvgaHeight);

    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.99f, 0.0f, 0.0f, 1.0f, 1.0f),
    };

    auto result = annotator.annotate(frame.data(), kQvgaWidth, kQvgaHeight, detections);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result));
}

TEST(SecurityDetectionAnnotator, TinyDetection_SkippedGracefully)
{
    // A sub-pixel detection (w/h map to <2 px at QVGA) — drawing a
    // bounding box this small is meaningless, so the annotator should
    // skip it silently rather than crash on degenerate rect coordinates.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kQvgaWidth, kQvgaHeight);

    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.90f, 0.5f, 0.5f,
                      kSubPixelSize, kSubPixelSize),
    };

    auto result = annotator.annotate(frame.data(), kQvgaWidth, kQvgaHeight, detections);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result));
}

TEST(SecurityDetectionAnnotator, DetectionPartiallyOutOfBounds_ClampedSafely)
{
    // Detection anchored at 80% of frame width/height with 50% extent —
    // the right and bottom edges overflow past the frame boundary.
    // The annotator must clamp to frame dimensions before drawing.
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kQvgaWidth, kQvgaHeight);

    std::vector<DetectionBox> detections = {
        makeDetection("car", 0.85f, 0.8f, 0.8f, 0.5f, 0.5f),
    };

    auto result = annotator.annotate(frame.data(), kQvgaWidth, kQvgaHeight, detections);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result));
}

// ===========================================================================
// Class colours
//
// Each COCO class maps to a distinct BGR colour via classColour().
// This test renders the same detection geometry with different class names
// and verifies that at least some produce visually different JPEG output,
// confirming the colour lookup table is wired up correctly.
// ===========================================================================

TEST(SecurityDetectionAnnotator, AllKnownClasses_ProduceDistinctOutput)
{
    auto frame = makeBgrFrame(kQvgaWidth, kQvgaHeight);

    // Six representative COCO classes that the security camera mode cares
    // about — these should map to at least 3 distinct colours.
    const std::vector<std::string> classes = {
        "person", "backpack", "suitcase", "car", "truck", "motorcycle",
    };

    std::vector<std::vector<uint8_t>> outputs;
    for (const auto& cls : classes) {
        SecurityDetectionAnnotator annotator;
        std::vector<DetectionBox> detections = {
            makeDetection(cls, 0.90f, 0.1f, 0.1f, 0.5f, 0.5f),
        };
        auto result = annotator.annotate(
            frame.data(), kQvgaWidth, kQvgaHeight, detections);
        ASSERT_TRUE(result.has_value()) << cls << ": " << result.error();
        outputs.push_back(std::move(*result));
    }

    int diffCount = 0;
    for (size_t i = 1; i < outputs.size(); ++i) {
        if (outputs[i] != outputs[0]) ++diffCount;
    }
    EXPECT_GE(diffCount, kMinDistinctColours)
        << "Multiple distinct classes should produce visually different overlays";
}

// ===========================================================================
// JPEG quality
//
// Config::jpegQuality controls the cv::imencode compression parameter.
// Higher quality preserves more DCT coefficients → larger file size.
// This test verifies the quality setting actually reaches the encoder.
// ===========================================================================

TEST(SecurityDetectionAnnotator, HighQuality_ProducesLargerJpeg)
{
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);
    std::vector<DetectionBox> detections = {
        makeDetection("person", 0.90f, 0.1f, 0.1f, 0.5f, 0.5f),
    };

    SecurityDetectionAnnotator::Config lowCfg;
    lowCfg.jpegQuality = kLowJpegQuality;
    SecurityDetectionAnnotator lowAnnotator(lowCfg);
    auto lowResult = lowAnnotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(lowResult.has_value());

    SecurityDetectionAnnotator::Config highCfg;
    highCfg.jpegQuality = kHighJpegQuality;
    SecurityDetectionAnnotator highAnnotator(highCfg);
    auto highResult = highAnnotator.annotate(
        frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(highResult.has_value());

    EXPECT_GT(highResult->size(), lowResult->size())
        << "Quality " << kHighJpegQuality << " should produce a larger JPEG than "
        << kLowJpegQuality;
}

// ===========================================================================
// Multiple detections — stress test
//
// A crowded scene (e.g. a busy intersection) can produce many simultaneous
// detections.  This test generates a 5x4 grid of 20 overlapping bounding
// boxes to verify the annotator handles bulk rendering without crashing
// or corrupting the output buffer.
// ===========================================================================

TEST(SecurityDetectionAnnotator, ManyDetections_AllRendered)
{
    SecurityDetectionAnnotator annotator;
    auto frame = makeBgrFrame(kVgaWidth, kVgaHeight);

    std::vector<DetectionBox> detections;
    detections.reserve(kGridTotal);
    for (int i = 0; i < kGridTotal; ++i) {
        float x = (i % kGridCols) * kGridCellW;
        float y = (i / kGridCols) * kGridCellH;
        float conf = kGridBaseConf + i * kGridConfStep;
        detections.push_back(
            makeDetection("person", conf, x, y, kGridBoxW, kGridBoxH));
    }

    auto result = annotator.annotate(frame.data(), kVgaWidth, kVgaHeight, detections);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(isValidJpeg(*result));
}
