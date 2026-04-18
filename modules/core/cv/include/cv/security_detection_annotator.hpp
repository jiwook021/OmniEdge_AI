#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// SecurityDetectionAnnotator — Draw detection overlays on video frames
//
// Supports multiple annotation styles:
//   kCornerBracket — L-shaped corner brackets with confidence bar (default)
//   kRectangle     — traditional full bounding-box rectangle
//   kCrosshair     — center crosshair with corner ticks
//
// Accepts BGR24 frame data and a list of detections, draws coloured geometry
// with class labels and confidence percentages, then encodes to JPEG.
// ---------------------------------------------------------------------------

namespace oe::security {

/// Visual annotation style for detection overlays.
enum class AnnotationStyle : uint8_t {
    kCornerBracket = 0,   ///< L-shaped corners + confidence bar (default)
    kRectangle     = 1,   ///< Full bounding-box rectangle (classic)
    kCrosshair     = 2,   ///< Center crosshair + corner ticks
};

/// A single detection to annotate on the frame.
struct DetectionBox {
    float       x;              ///< Top-left X (normalised 0-1)
    float       y;              ///< Top-left Y (normalised 0-1)
    float       w;              ///< Width (normalised 0-1)
    float       h;              ///< Height (normalised 0-1)
    float       confidence;     ///< Detection confidence [0.0, 1.0]
    std::string className;      ///< COCO class name (e.g. "person")
    int         trackId = -1;   ///< Optional tracker ID (-1 = untracked)
};

class SecurityDetectionAnnotator {
public:
    struct Config {
        int jpegQuality          = 85;
        int bboxThickness        = 3;
        AnnotationStyle style    = AnnotationStyle::kCornerBracket;
    };

    SecurityDetectionAnnotator() : SecurityDetectionAnnotator(Config{}) {}
    explicit SecurityDetectionAnnotator(const Config& config);
    ~SecurityDetectionAnnotator() = default;

    /// Annotate a BGR24 frame with detection overlays and encode to JPEG.
    [[nodiscard]] tl::expected<std::vector<uint8_t>, std::string>
    annotate(const uint8_t* bgr24, int width, int height,
             const std::vector<DetectionBox>& detections);

    /// Change the annotation style at runtime (called from UI command).
    void setStyle(AnnotationStyle style) noexcept { config_.style = style; }

    /// Get the current style.
    [[nodiscard]] AnnotationStyle style() const noexcept { return config_.style; }

    /// Set the ROI polygon (normalised 0-1 coordinates).
    /// Pass an empty vector to clear the ROI.
    void setRoi(std::vector<std::pair<float, float>> points) noexcept {
        roiPoints_ = std::move(points);
    }

    /// Get the current ROI polygon (empty = no ROI active).
    [[nodiscard]] const std::vector<std::pair<float, float>>& roi() const noexcept {
        return roiPoints_;
    }

    /// Test whether a normalised point (nx, ny) is inside the ROI polygon.
    /// Returns true if the ROI is empty (no filtering).
    [[nodiscard]] bool isInsideRoi(float nx, float ny) const;

private:
    /// Map a class name to a BGR colour for the bounding box.
    static void classColour(const std::string& className,
                            uint8_t& b, uint8_t& g, uint8_t& r);

    /// Map a class name to a short hex colour string for the frontend.
    static const char* classHexColour(const std::string& className);

    Config config_;
    std::vector<std::pair<float, float>> roiPoints_;  ///< Normalised ROI polygon vertices
};

}  // namespace oe::security
