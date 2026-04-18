#pragma once

#include <tl/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/constants/cv_constants.hpp"

// ---------------------------------------------------------------------------
// Sam2Inferencer — pure virtual interface for SAM2 (Segment Anything Model 2)
// image/video segmentation with interactive prompts.
//
// SAM2 takes an image plus one or more prompts (point, bounding box, or mask)
// and produces a pixel-perfect binary segmentation mask.
//
// Pipeline per frame:
//   1. Image encoder: ViT backbone encodes the full image into embeddings
//   2. Prompt encoder: encodes user prompts (points / boxes / masks)
//   3. Mask decoder: fuses image + prompt embeddings → segmentation mask
//   4. Post-process: threshold logits → binary mask, JPEG-encode overlay
//
// Concrete implementations:
//   OnnxSam2Inferencer  — ONNX Runtime + TensorRT EP (production)
//   StubSam2Inferencer  — deterministic synthetic mask (tests / CPU-only CI)
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Prompt types for SAM2 interactive segmentation
// ---------------------------------------------------------------------------

/// A single 2D point prompt with a label (foreground=1, background=0).
struct Sam2PointPrompt {
    float x = 0.0f;  ///< Normalised x coordinate [0, 1]
    float y = 0.0f;  ///< Normalised y coordinate [0, 1]
    int   label = 1; ///< 1 = foreground, 0 = background
};

/// A bounding box prompt (top-left and bottom-right corners).
struct Sam2BoxPrompt {
    float x1 = 0.0f;  ///< Top-left x, normalised [0, 1]
    float y1 = 0.0f;  ///< Top-left y, normalised [0, 1]
    float x2 = 0.0f;  ///< Bottom-right x, normalised [0, 1]
    float y2 = 0.0f;  ///< Bottom-right y, normalised [0, 1]
};

/// Prompt type enum for dispatching.
enum class Sam2PromptType : uint8_t {
    kPoint = 0,  ///< One or more point prompts
    kBox   = 1,  ///< A bounding box prompt
    kMask  = 2,  ///< A prior mask prompt (refinement)
};

/// Combined prompt container — holds exactly one prompt type per inference call.
struct Sam2Prompt {
    Sam2PromptType                type = Sam2PromptType::kPoint;
    std::vector<Sam2PointPrompt>  points;   ///< Used when type == kPoint
    Sam2BoxPrompt                 box;      ///< Used when type == kBox
    std::vector<uint8_t>          mask;     ///< Used when type == kMask (binary mask, 1 byte/pixel)
    uint32_t                      maskWidth  = 0;  ///< Width of the input mask
    uint32_t                      maskHeight = 0;  ///< Height of the input mask
};

/// Result of a SAM2 segmentation pass.
struct Sam2Result {
    std::vector<uint8_t> mask;         ///< Binary mask (0 or 255 per pixel, H×W)
    uint32_t             maskWidth  = 0;
    uint32_t             maskHeight = 0;
    float                iouScore   = 0.0f;  ///< Model's predicted IoU confidence
    float                stability  = 0.0f;  ///< Stability score (mask quality metric)
};

class Sam2Inferencer {
public:
    virtual ~Sam2Inferencer() = default;

    /** Load the SAM2 ONNX model (image encoder + mask decoder).
     *  @param encoderPath  Path to the SAM2 image encoder ONNX file.
     *  @param decoderPath  Path to the SAM2 mask decoder ONNX file.
     *  @throws std::runtime_error on model load failure
     *  (call only from initialize(); propagates to main as fatal).
     */
    virtual void loadModel(const std::string& encoderPath,
                           const std::string& decoderPath) = 0;

    /** Encode an image — call once per new image, then run segmentWithPrompt()
     *  for each prompt without re-encoding.
     *
     *  @param bgrFrame  Pointer to BGR24 host bytes (width × height × 3)
     *  @param width     Frame width in pixels
     *  @param height    Frame height in pixels
     *  @return void on success, or error string.
     */
    [[nodiscard]] virtual tl::expected<void, std::string>
    encodeImage(const uint8_t* bgrFrame,
                uint32_t       width,
                uint32_t       height) = 0;

    /** Run segmentation on the previously encoded image with the given prompt.
     *
     *  @param prompt    User prompt (point, box, or mask).
     *  @return          Sam2Result with binary mask and quality scores, or error.
     */
    [[nodiscard]] virtual tl::expected<Sam2Result, std::string>
    segmentWithPrompt(const Sam2Prompt& prompt) = 0;

    /** Run the full pipeline (encode + segment) in one call.
     *  Convenience method for single-shot segmentation.
     *
     *  @param bgrFrame      Pointer to BGR24 host bytes (width × height × 3)
     *  @param width         Frame width in pixels
     *  @param height        Frame height in pixels
     *  @param prompt        User prompt (point, box, or mask).
     *  @param outBuf        Destination buffer for JPEG-encoded overlay
     *  @param maxJpegBytes  Capacity of outBuf in bytes
     *  @return              Actual JPEG byte count, or error string.
     */
    [[nodiscard]] virtual tl::expected<std::size_t, std::string>
    processFrame(const uint8_t*    bgrFrame,
                 uint32_t          width,
                 uint32_t          height,
                 const Sam2Prompt& prompt,
                 uint8_t*          outBuf,
                 std::size_t       maxJpegBytes) = 0;

    /** Run the full pipeline but return composited BGR24 instead of JPEG.
     *
     *  Same pipeline as processFrame() (encode + segment + overlay)
     *  but skips the JPEG encode step. Used in pipeline chaining mode
     *  where the output feeds into another CV node via ShmCircularBuffer.
     *
     *  @param bgrFrame      Pointer to BGR24 host bytes (width * height * 3)
     *  @param width         Frame width in pixels
     *  @param height        Frame height in pixels
     *  @param prompt        User prompt (point, box, or mask).
     *  @param outBgrBuf     Destination buffer for BGR24 output (width * height * 3)
     *  @param maxBgrBytes   Capacity of outBgrBuf in bytes
     *  @return              Actual BGR24 byte count, or error string.
     *
     *  Default returns an error — concrete inferencers override when BGR24
     *  output is supported.
     */
    [[nodiscard]] virtual tl::expected<std::size_t, std::string>
    processFrameGetBgr(const uint8_t*    bgrFrame,
                       uint32_t          width,
                       uint32_t          height,
                       const Sam2Prompt& prompt,
                       uint8_t*          outBgrBuf,
                       std::size_t       maxBgrBytes)
    {
        (void)bgrFrame; (void)width; (void)height;
        (void)prompt; (void)outBgrBuf; (void)maxBgrBytes;
        return tl::unexpected(std::string("BGR24 output not supported by this inferencer"));
    }

    /** Set the background compositing mode for segmentation output.
     *  Default: kOverlay (blue tint).
     */
    virtual void setBackgroundMode(Sam2BackgroundMode mode) noexcept {
        bgMode_ = mode;
    }

    /** Set the custom background color for kColor mode (RGB, 0-255). */
    virtual void setBackgroundColor(uint8_t r, uint8_t g, uint8_t b) noexcept {
        bgColorR_ = r; bgColorG_ = g; bgColorB_ = b;
    }

    /** Set a custom background image for kImage mode.
     *  @param jpegData  JPEG or PNG encoded image bytes.
     *  @param width     Image width in pixels (0 = auto-detect from header).
     *  @param height    Image height in pixels (0 = auto-detect from header).
     */
    virtual void setBackgroundImage(const std::vector<uint8_t>& jpegData,
                                    uint32_t width, uint32_t height) noexcept {
        if (jpegData.size() > kSam2MaxBgImageBytes) return;
        bgImageData_   = jpegData;
        bgImageWidth_  = width;
        bgImageHeight_ = height;
    }

    /** Check if a background image has been uploaded. */
    [[nodiscard]] bool hasBackgroundImage() const noexcept {
        return !bgImageData_.empty();
    }

    /** Get the current background mode. */
    [[nodiscard]] Sam2BackgroundMode backgroundMode() const noexcept { return bgMode_; }

    /** Get the last segmentation result (IoU score, stability, mask dimensions).
     *  processFrame() stores its result internally — callers use this instead
     *  of re-calling segmentWithPrompt() to avoid redundant GPU work.
     */
    [[nodiscard]] const Sam2Result& lastSegmentResult() const noexcept {
        return lastResult_;
    }

    /** Unload model and free all GPU memory. */
    virtual void unload() noexcept = 0;

    /** VRAM used by this inferencer in bytes (for daemon eviction accounting). */
    [[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;

protected:
    Sam2BackgroundMode bgMode_ = Sam2BackgroundMode::kOverlay;
    uint8_t bgColorR_ = kSam2DefaultBgColorR;
    uint8_t bgColorG_ = kSam2DefaultBgColorG;
    uint8_t bgColorB_ = kSam2DefaultBgColorB;

    // Last segmentation result — stored by processFrame() for callers
    // to retrieve metadata (IoU, stability) without re-running inference.
    Sam2Result lastResult_;

    // Background image for kImage mode
    std::vector<uint8_t> bgImageData_;
    uint32_t bgImageWidth_  = 0;
    uint32_t bgImageHeight_ = 0;
};

