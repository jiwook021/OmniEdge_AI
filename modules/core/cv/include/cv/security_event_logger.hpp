#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// SecurityEventLogger — Append-only JSON Lines event logger for Security Mode
//
// Each detection event is serialised as a single JSON line and flushed
// immediately to guarantee durability on power loss.  File rotation occurs
// when the log exceeds kSecurityEventLogMaxSizeMB.
//
// Thread safety: all public methods are internally synchronised (mutex).
// ---------------------------------------------------------------------------

namespace oe::security {

/// A single detection event persisted to the log file.
struct SecurityEvent {
    std::string timestamp;       ///< ISO-8601 UTC (e.g. "2026-04-12T14:30:15.123Z")
    std::string detectedClass;   ///< COCO class name (e.g. "person", "suitcase")
    float       confidence;      ///< Detection confidence [0.0, 1.0]
    float       bboxX;           ///< Bounding box top-left X (normalised 0-1)
    float       bboxY;           ///< Bounding box top-left Y (normalised 0-1)
    float       bboxW;           ///< Bounding box width  (normalised 0-1)
    float       bboxH;           ///< Bounding box height (normalised 0-1)
    std::string recordingFile;   ///< Active MP4 file at time of detection
    uint64_t    frameSeq;        ///< Frame sequence number from VideoIngest
};

class SecurityEventLogger {
public:
    /// Construct with a log directory path.  The log file is created on
    /// first call to appendEvent().
    explicit SecurityEventLogger(std::filesystem::path logDir,
                                  std::string_view fileName = "security_events.jsonl",
                                  int maxSizeMB = 100);

    ~SecurityEventLogger();

    SecurityEventLogger(const SecurityEventLogger&) = delete;
    SecurityEventLogger& operator=(const SecurityEventLogger&) = delete;

    /// Append a detection event as a single JSON line.  Flushes immediately.
    /// Returns an error string on I/O failure.
    [[nodiscard]] tl::expected<void, std::string> appendEvent(const SecurityEvent& event);

    /// Query events within a time range (ISO-8601 strings, inclusive).
    /// Reads the current log file(s) and returns matching events.
    [[nodiscard]] tl::expected<std::vector<SecurityEvent>, std::string>
    queryEvents(std::string_view fromTime, std::string_view toTime) const;

    /// Current log file path.
    [[nodiscard]] std::filesystem::path currentLogPath() const;

    /// Total number of events written since construction (or last rotation).
    [[nodiscard]] uint64_t eventCount() const noexcept;

private:
    /// Open or rotate the log file if it exceeds the size limit.
    [[nodiscard]] tl::expected<void, std::string> ensureFileOpen();

    /// Rotate: close current file, rename with timestamp suffix.
    [[nodiscard]] tl::expected<void, std::string> rotateIfNeeded();

    /// Serialise a SecurityEvent to a JSON object.
    static nlohmann::json eventToJson(const SecurityEvent& event);

    /// Deserialise a JSON line to a SecurityEvent.
    static tl::expected<SecurityEvent, std::string> jsonToEvent(std::string_view line);

    std::filesystem::path logDir_;
    std::string           fileName_;
    int                   maxSizeMB_;
    std::filesystem::path currentPath_;

    mutable std::mutex    mutex_;
    std::ofstream*        file_ = nullptr;  ///< Raw pointer; owned, closed in dtor
    uint64_t              eventCount_ = 0;
};

}  // namespace oe::security
