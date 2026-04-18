#include "cv/security_detection_annotator.hpp"

#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "common/constants/security_constants.hpp"
#include "common/oe_tracy.hpp"

namespace oe::security {

SecurityDetectionAnnotator::SecurityDetectionAnnotator(const Config& config)
    : config_(config)
{
    SPDLOG_INFO("[SecurityDetectionAnnotator] jpegQuality={} bboxThickness={} style={}",
                config_.jpegQuality, config_.bboxThickness,
                static_cast<int>(config_.style));
}

// ── Corner Bracket Drawing ─────────────────────────────────────────────────

/// Draw an L-shaped bracket at one corner of the bounding box.
///   corner: 0=TL, 1=TR, 2=BR, 3=BL
static void drawCornerBracket(cv::Mat& img, int cx, int cy, int armLen,
                               int thickness, cv::Scalar colour, int corner) {
    int dx = (corner == 0 || corner == 3) ? 1 : -1;  // right or left
    int dy = (corner == 0 || corner == 1) ? 1 : -1;  // down or up

    // Horizontal arm.
    cv::line(img, cv::Point(cx, cy), cv::Point(cx + dx * armLen, cy),
             colour, thickness, cv::LINE_AA);
    // Vertical arm.
    cv::line(img, cv::Point(cx, cy), cv::Point(cx, cy + dy * armLen),
             colour, thickness, cv::LINE_AA);
}

/// Draw a confidence bar below the bounding box.
static void drawConfidenceBar(cv::Mat& img, int x1, int y2, int boxWidth,
                               float confidence, cv::Scalar colour) {
    int barY   = std::min(y2 + 2, img.rows - kSecurityConfidenceBarHeight - 1);
    int barLen = static_cast<int>(boxWidth * confidence);

    // Background (dark).
    cv::rectangle(img,
                  cv::Point(x1, barY),
                  cv::Point(x1 + boxWidth, barY + kSecurityConfidenceBarHeight),
                  cv::Scalar(40, 40, 40), cv::FILLED);
    // Filled portion.
    if (barLen > 0) {
        cv::rectangle(img,
                      cv::Point(x1, barY),
                      cv::Point(x1 + barLen, barY + kSecurityConfidenceBarHeight),
                      colour, cv::FILLED);
    }
}

/// Draw a crosshair + corner ticks.
static void drawCrosshair(cv::Mat& img, int x1, int y1, int x2, int y2,
                           int thickness, cv::Scalar colour, int armLen) {
    int cx = (x1 + x2) / 2;
    int cy = (y1 + y2) / 2;
    int crossLen = armLen;

    // Center crosshair.
    cv::line(img, cv::Point(cx - crossLen, cy), cv::Point(cx + crossLen, cy),
             colour, thickness, cv::LINE_AA);
    cv::line(img, cv::Point(cx, cy - crossLen), cv::Point(cx, cy + crossLen),
             colour, thickness, cv::LINE_AA);

    // Corner ticks (short marks at each corner).
    int tickLen = armLen / 2;
    drawCornerBracket(img, x1, y1, tickLen, thickness, colour, 0);
    drawCornerBracket(img, x2, y1, tickLen, thickness, colour, 1);
    drawCornerBracket(img, x2, y2, tickLen, thickness, colour, 2);
    drawCornerBracket(img, x1, y2, tickLen, thickness, colour, 3);
}

// ── Main Annotate Method ───────────────────────────────────────────────────

tl::expected<std::vector<uint8_t>, std::string>
SecurityDetectionAnnotator::annotate(const uint8_t* bgr24, int width, int height,
                                      const std::vector<DetectionBox>& detections) {
    OE_ZONE_SCOPED;

    if (!bgr24 || width <= 0 || height <= 0) {
        return tl::unexpected(std::string("Invalid frame data"));
    }

    // Wrap raw BGR24 data in a cv::Mat (no copy — shared pointer).
    cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(bgr24));

    // Clone the frame to avoid modifying the original SHM data.
    cv::Mat annotated = frame.clone();

    for (const auto& det : detections) {
        // Convert normalised coords to pixel coords.
        int x1 = static_cast<int>(det.x * width);
        int y1 = static_cast<int>(det.y * height);
        int x2 = static_cast<int>((det.x + det.w) * width);
        int y2 = static_cast<int>((det.y + det.h) * height);

        // Clamp to frame bounds.
        x1 = std::clamp(x1, 0, width  - 1);
        y1 = std::clamp(y1, 0, height - 1);
        x2 = std::clamp(x2, 0, width  - 1);
        y2 = std::clamp(y2, 0, height - 1);

        int boxW = x2 - x1;
        int boxH = y2 - y1;
        if (boxW < 2 || boxH < 2) continue;

        uint8_t b, g, r;
        classColour(det.className, b, g, r);
        cv::Scalar colour(b, g, r);

        // Compute corner bracket arm length.
        int shortSide = std::min(boxW, boxH);
        int armLen = std::max(
            static_cast<int>(shortSide * kSecurityCornerLengthFraction),
            kSecurityCornerLengthMinPx);

        switch (config_.style) {
        case AnnotationStyle::kCornerBracket: {
            // Four L-shaped corner brackets.
            drawCornerBracket(annotated, x1, y1, armLen, config_.bboxThickness, colour, 0);
            drawCornerBracket(annotated, x2, y1, armLen, config_.bboxThickness, colour, 1);
            drawCornerBracket(annotated, x2, y2, armLen, config_.bboxThickness, colour, 2);
            drawCornerBracket(annotated, x1, y2, armLen, config_.bboxThickness, colour, 3);

            // Confidence bar below the box.
            drawConfidenceBar(annotated, x1, y2, boxW, det.confidence, colour);
            break;
        }
        case AnnotationStyle::kRectangle: {
            // Full bounding-box rectangle.
            cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2),
                          colour, config_.bboxThickness, cv::LINE_AA);
            break;
        }
        case AnnotationStyle::kCrosshair: {
            drawCrosshair(annotated, x1, y1, x2, y2,
                          config_.bboxThickness, colour, armLen);
            break;
        }
        }

        // ── Label (all styles) ─────────────────────────────────────────
        char labelBuf[64];
        std::snprintf(labelBuf, sizeof(labelBuf), "%s %.0f%%",
                      det.className.c_str(), det.confidence * 100.0f);
        std::string label(labelBuf);

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.55, 1, &baseline);
        int labelY = std::max(y1 - textSize.height - 6, 0);

        // Semi-transparent label background.
        cv::Mat roi = annotated(cv::Rect(
            x1, labelY,
            std::min(textSize.width + 8, width - x1),
            std::min(textSize.height + 8, height - labelY)));
        cv::Mat overlay;
        roi.copyTo(overlay);
        cv::rectangle(overlay, cv::Point(0, 0),
                      cv::Point(overlay.cols, overlay.rows),
                      cv::Scalar(b / 3, g / 3, r / 3), cv::FILLED);
        cv::addWeighted(overlay, 0.7, roi, 0.3, 0, roi);

        cv::putText(annotated, label,
                     cv::Point(x1 + 4, labelY + textSize.height + 3),
                     cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                     cv::LINE_AA);
    }

    // Draw ROI polygon overlay (if set).
    if (!roiPoints_.empty()) {
        std::vector<cv::Point> pts;
        pts.reserve(roiPoints_.size());
        for (const auto& [px, py] : roiPoints_) {
            pts.emplace_back(static_cast<int>(px * width),
                             static_cast<int>(py * height));
        }
        cv::Scalar roiColour(kSecurityRoiB, kSecurityRoiG, kSecurityRoiR);
        cv::polylines(annotated, pts, /*isClosed=*/true, roiColour,
                      kSecurityRoiThickness, cv::LINE_AA);
        for (const auto& pt : pts) {
            cv::circle(annotated, pt, 4, roiColour, cv::FILLED, cv::LINE_AA);
        }
    }

    // Encode to JPEG.
    std::vector<uint8_t> jpegBuf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config_.jpegQuality};
    if (!cv::imencode(".jpg", annotated, jpegBuf, params)) {
        return tl::unexpected(std::string("JPEG encoding failed"));
    }

    SPDLOG_DEBUG("[SecurityDetectionAnnotator] annotated {} detections, JPEG {} bytes",
                 detections.size(), jpegBuf.size());
    return jpegBuf;
}

// ── ROI Point-in-Polygon ───────────────────────────────────────────────────

bool SecurityDetectionAnnotator::isInsideRoi(float nx, float ny) const {
    if (roiPoints_.empty()) return true;  // No ROI = everything passes.

    // Convert normalised ROI points to cv::Point2f and use pointPolygonTest.
    std::vector<cv::Point2f> contour;
    contour.reserve(roiPoints_.size());
    for (const auto& [px, py] : roiPoints_) {
        contour.emplace_back(px, py);
    }

    double dist = cv::pointPolygonTest(contour, cv::Point2f(nx, ny), /*measureDist=*/false);
    return dist >= 0;  // >= 0 means inside or on edge.
}

// ── Colour Mapping ─────────────────────────────────────────────────────────

void SecurityDetectionAnnotator::classColour(const std::string& className,
                                              uint8_t& b, uint8_t& g, uint8_t& r) {
    if (className == "person") {
        b = kSecurityBboxPersonB;
        g = kSecurityBboxPersonG;
        r = kSecurityBboxPersonR;
    } else if (className == "backpack" || className == "suitcase") {
        b = kSecurityBboxPackageB;
        g = kSecurityBboxPackageG;
        r = kSecurityBboxPackageR;
    } else if (className == "car") {
        b = kSecurityBboxCarB;
        g = kSecurityBboxCarG;
        r = kSecurityBboxCarR;
    } else if (className == "truck" || className == "bus") {
        b = kSecurityBboxTruckB;
        g = kSecurityBboxTruckG;
        r = kSecurityBboxTruckR;
    } else if (className == "motorcycle" || className == "bicycle") {
        b = kSecurityBboxBikeB;
        g = kSecurityBboxBikeG;
        r = kSecurityBboxBikeR;
    } else {
        // Default: yellow for unknown classes.
        b = 0; g = 255; r = 255;
    }
}

const char* SecurityDetectionAnnotator::classHexColour(const std::string& className) {
    if (className == "person")                                         return "#00ff00";
    if (className == "backpack" || className == "suitcase")              return "#ffa500";
    if (className == "car")                                            return "#00ffff";
    if (className == "truck" || className == "bus")                    return "#ff8c00";
    if (className == "motorcycle" || className == "bicycle")           return "#ff00ff";
    return "#ffff00";
}

}  // namespace oe::security
