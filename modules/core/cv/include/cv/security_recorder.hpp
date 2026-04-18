#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>

// Forward-declare GStreamer types to avoid polluting headers.
struct _GstElement;
typedef struct _GstElement GstElement;

// ---------------------------------------------------------------------------
// SecurityRecorder — GStreamer NVENC MP4 recording pipeline for Security Mode
//
// Push-model pipeline: caller feeds BGR24 frames via pushFrame(), GStreamer
// encodes with hardware H.264 (nvv4l2h264enc) and writes segmented MP4 files
// via splitmuxsink.  Falls back to x264enc on systems without NVENC (WSL2).
//
// Thread safety: pushFrame() may be called from any single thread.
// Internal GStreamer threads handle encoding and I/O asynchronously.
// ---------------------------------------------------------------------------

namespace oe::security {

/// Metadata about a recorded MP4 segment.
struct RecordingInfo {
    std::string file;        ///< File name (e.g. "security_20260412_140000.mp4")
    std::string startTime;   ///< ISO-8601 UTC start time
    std::string endTime;     ///< ISO-8601 UTC end time (empty if recording)
    double      sizeMB;      ///< File size in megabytes
    int         eventCount;  ///< Number of detection events during this segment
};

class SecurityRecorder {
public:
    struct Config {
        std::filesystem::path recordingDir = "recordings/";
        int                   fps          = 20;
        int                   width        = 1920;
        int                   height       = 1080;
        int                   bitrate      = 5'000'000;  ///< H.264 bitrate (bps)
        int                   segmentDurationMin = 30;    ///< splitmuxsink segment length
    };

    SecurityRecorder() = default;
    ~SecurityRecorder();

    SecurityRecorder(const SecurityRecorder&) = delete;
    SecurityRecorder& operator=(const SecurityRecorder&) = delete;

    /// Build and start the GStreamer pipeline.
    [[nodiscard]] tl::expected<void, std::string> start(const Config& config);

    /// Feed a single BGR24 frame.  Must match config width×height×3 bytes.
    /// timestampNs is the monotonic capture timestamp from VideoIngest.
    [[nodiscard]] tl::expected<void, std::string>
    pushFrame(const uint8_t* bgr24, std::size_t size, uint64_t timestampNs);

    /// Flush buffers, finalise the current MP4, and tear down the pipeline.
    void stop();

    /// Is the pipeline currently recording?
    [[nodiscard]] bool isRecording() const noexcept;

    /// Path of the current (or most recent) recording segment.
    [[nodiscard]] std::string currentFile() const;

    /// List all MP4 recordings in the recording directory.
    [[nodiscard]] static std::vector<RecordingInfo>
    listRecordings(const std::filesystem::path& dir);

    /// Delete MP4 recordings older than maxAgeDays.  Returns the number of
    /// files removed.  Safe to call from any thread (no instance state).
    [[nodiscard]] static int purgeOldRecordings(const std::filesystem::path& dir,
                                                int maxAgeDays);

private:
    /// Detect whether NVENC hardware encoder is available.
    [[nodiscard]] static bool hasNvenc();

    /// Build the GStreamer pipeline string.
    [[nodiscard]] std::string buildPipelineString() const;

    Config                config_;
    GstElement*           pipeline_ = nullptr;
    GstElement*           appsrc_   = nullptr;
    std::atomic<bool>     recording_{false};
    std::string           currentFile_;
    mutable std::mutex    mutex_;
    uint64_t              frameCount_ = 0;
    bool                  useNvenc_   = false;
};

}  // namespace oe::security
