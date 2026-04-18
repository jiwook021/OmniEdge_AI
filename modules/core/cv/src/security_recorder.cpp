#include "cv/security_recorder.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <spdlog/spdlog.h>

#include "common/oe_tracy.hpp"

namespace oe::security {

// ── Destruction ─────────────────────────────────────────────────────────────

SecurityRecorder::~SecurityRecorder() {
    stop();
}

// ── Public API ──────────────────────────────────────────────────────────────

tl::expected<void, std::string> SecurityRecorder::start(const Config& config) {
    OE_ZONE_SCOPED;
    std::lock_guard lock(mutex_);

    if (recording_.load(std::memory_order_relaxed)) {
        return tl::unexpected(std::string("Already recording"));
    }

    config_ = config;
    useNvenc_ = hasNvenc();
    SPDLOG_INFO("[SecurityRecorder] encoder: {}", useNvenc_ ? "NVENC (hardware)" : "x264 (software fallback)");

    // Ensure output directory exists.
    std::error_code ec;
    std::filesystem::create_directories(config_.recordingDir, ec);
    if (ec) {
        std::string err = "Failed to create recording dir: " + config_.recordingDir.string();
        SPDLOG_ERROR("[SecurityRecorder] {}", err);
        return tl::unexpected(std::move(err));
    }

    // Initialise GStreamer if needed.
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    // Build and launch the pipeline.
    std::string pipelineStr = buildPipelineString();
    SPDLOG_INFO("[SecurityRecorder] pipeline: {}", pipelineStr);

    GError* error = nullptr;
    pipeline_ = gst_parse_launch(pipelineStr.c_str(), &error);
    if (error) {
        std::string err = std::string("GStreamer pipeline error: ") + error->message;
        SPDLOG_ERROR("[SecurityRecorder] {}", err);
        g_error_free(error);
        return tl::unexpected(std::move(err));
    }

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "security_src");
    if (!appsrc_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return tl::unexpected(std::string("Failed to find appsrc element"));
    }

    // Configure appsrc caps.
    GstCaps* caps = gst_caps_new_simple("video/x-raw",
        "format",    G_TYPE_STRING,  "BGR",
        "width",     G_TYPE_INT,     config_.width,
        "height",    G_TYPE_INT,     config_.height,
        "framerate", GST_TYPE_FRACTION, config_.fps, 1,
        nullptr);
    g_object_set(appsrc_, "caps", caps, "format", GST_FORMAT_TIME,
                 "is-live", TRUE, nullptr);
    gst_caps_unref(caps);

    // Start pipeline.
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        gst_object_unref(appsrc_);
        gst_object_unref(pipeline_);
        appsrc_ = nullptr;
        pipeline_ = nullptr;
        return tl::unexpected(std::string("Failed to start GStreamer pipeline"));
    }

    recording_.store(true, std::memory_order_release);
    frameCount_ = 0;

    SPDLOG_INFO("[SecurityRecorder] recording started — dir={}, {}x{} @ {} fps",
                config_.recordingDir.string(), config_.width, config_.height, config_.fps);
    return {};
}

tl::expected<void, std::string>
SecurityRecorder::pushFrame(const uint8_t* bgr24, std::size_t size, uint64_t timestampNs) {
    OE_ZONE_SCOPED;

    if (!recording_.load(std::memory_order_acquire)) {
        return tl::unexpected(std::string("Not recording"));
    }

    std::size_t expectedSize = static_cast<std::size_t>(config_.width) * config_.height * 3;
    if (size != expectedSize) {
        return tl::unexpected("Frame size mismatch: expected " +
                              std::to_string(expectedSize) + ", got " + std::to_string(size));
    }

    // Allocate a GstBuffer and copy frame data.
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    std::memcpy(map.data, bgr24, size);
    gst_buffer_unmap(buffer, &map);

    // Set buffer timestamp.
    GST_BUFFER_PTS(buffer) = timestampNs;
    GST_BUFFER_DTS(buffer) = timestampNs;
    GST_BUFFER_DURATION(buffer) = static_cast<uint64_t>(GST_SECOND) / config_.fps;

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (ret != GST_FLOW_OK) {
        SPDLOG_WARN("[SecurityRecorder] push_buffer returned {}", static_cast<int>(ret));
        return tl::unexpected(std::string("gst_app_src_push_buffer failed"));
    }

    ++frameCount_;
    if (frameCount_ % 100 == 0) {
        SPDLOG_DEBUG("[SecurityRecorder] pushed {} frames", frameCount_);
    }

    return {};
}

void SecurityRecorder::stop() {
    OE_ZONE_SCOPED;
    std::lock_guard lock(mutex_);

    if (!recording_.load(std::memory_order_acquire)) return;

    recording_.store(false, std::memory_order_release);

    if (appsrc_) {
        gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        gst_object_unref(appsrc_);
        appsrc_ = nullptr;
    }

    if (pipeline_) {
        // Wait for pipeline to finish processing EOS.
        GstBus* bus = gst_element_get_bus(pipeline_);
        if (bus) {
            gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
                                       static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
            gst_object_unref(bus);
        }

        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    SPDLOG_INFO("[SecurityRecorder] stopped — {} frames written", frameCount_);
}

bool SecurityRecorder::isRecording() const noexcept {
    return recording_.load(std::memory_order_acquire);
}

std::string SecurityRecorder::currentFile() const {
    std::lock_guard lock(mutex_);
    return currentFile_;
}

std::vector<RecordingInfo>
SecurityRecorder::listRecordings(const std::filesystem::path& dir) {
    OE_ZONE_SCOPED;
    std::vector<RecordingInfo> recordings;

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return recordings;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".mp4") continue;

        RecordingInfo info;
        info.file = entry.path().filename().string();
        info.sizeMB = static_cast<double>(entry.file_size(ec)) / (1024.0 * 1024.0);
        info.eventCount = 0;  // Populated by caller from event log cross-reference.

        // Extract start time from filename: security_YYYYMMDD_HHMMSS.mp4
        std::string stem = entry.path().stem().string();
        if (stem.size() >= 24 && stem.substr(0, 9) == "security_") {
            // Parse YYYYMMDD_HHMMSS
            std::string dateStr = stem.substr(9, 8);
            std::string timeStr = stem.substr(18, 6);
            info.startTime = dateStr.substr(0, 4) + "-" + dateStr.substr(4, 2) + "-" +
                             dateStr.substr(6, 2) + "T" + timeStr.substr(0, 2) + ":" +
                             timeStr.substr(2, 2) + ":" + timeStr.substr(4, 2) + "Z";
        }

        recordings.push_back(std::move(info));
    }

    // Sort by filename (chronological).
    std::sort(recordings.begin(), recordings.end(),
              [](const RecordingInfo& a, const RecordingInfo& b) {
                  return a.file < b.file;
              });

    SPDLOG_DEBUG("[SecurityRecorder] listed {} recordings in {}", recordings.size(), dir.string());
    return recordings;
}

int SecurityRecorder::purgeOldRecordings(const std::filesystem::path& dir,
                                         int maxAgeDays) {
    OE_ZONE_SCOPED;
    int removed = 0;

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return 0;

    auto cutoff = std::filesystem::file_time_type::clock::now()
                  - std::chrono::hours(maxAgeDays * 24);

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".mp4") continue;

        auto mtime = entry.last_write_time(ec);
        if (ec) continue;

        if (mtime < cutoff) {
            std::string filename = entry.path().filename().string();
            if (std::filesystem::remove(entry.path(), ec)) {
                SPDLOG_INFO("[SecurityRecorder] purged old recording: {}", filename);
                ++removed;
            } else {
                SPDLOG_WARN("[SecurityRecorder] failed to delete {}: {}", filename, ec.message());
            }
        }
    }

    if (removed > 0) {
        SPDLOG_INFO("[SecurityRecorder] purged {} recordings older than {} days", removed, maxAgeDays);
    }
    return removed;
}

// ── Private ─────────────────────────────────────────────────────────────────

bool SecurityRecorder::hasNvenc() {
    // Probe for the nvv4l2h264enc GStreamer element.
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    GstElementFactory* factory = gst_element_factory_find("nvv4l2h264enc");
    if (factory) {
        gst_object_unref(factory);
        return true;
    }

    // Also try nvh264enc (NVIDIA plugin on desktop Linux).
    factory = gst_element_factory_find("nvh264enc");
    if (factory) {
        gst_object_unref(factory);
        return true;
    }

    return false;
}

std::string SecurityRecorder::buildPipelineString() const {
    // Generate file location pattern for splitmuxsink.
    std::string locationPattern = (config_.recordingDir / "security_%05d.mp4").string();

    // Segment max duration in nanoseconds.
    uint64_t segmentNs = static_cast<uint64_t>(config_.segmentDurationMin) * 60 * GST_SECOND;

    std::string encoder;
    if (useNvenc_) {
        encoder = "nvv4l2h264enc bitrate=" + std::to_string(config_.bitrate) +
                  " preset-level=4 profile=4";
    } else {
        encoder = "x264enc bitrate=" + std::to_string(config_.bitrate / 1000) +
                  " speed-preset=ultrafast tune=zerolatency";
    }

    return "appsrc name=security_src format=time is-live=true "
           "! videoconvert "
           "! video/x-raw,format=I420 "
           "! " + encoder + " "
           "! h264parse "
           "! splitmuxsink location=" + locationPattern +
           " max-size-time=" + std::to_string(segmentNs) +
           " muxer-factory=mp4mux";
}

}  // namespace oe::security
