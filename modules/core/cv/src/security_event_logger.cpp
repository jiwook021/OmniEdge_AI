#include "cv/security_event_logger.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "common/oe_tracy.hpp"

namespace oe::security {

// ── Construction / Destruction ──────────────────────────────────────────────

SecurityEventLogger::SecurityEventLogger(std::filesystem::path logDir,
                                          std::string_view fileName,
                                          int maxSizeMB)
    : logDir_(std::move(logDir))
    , fileName_(fileName)
    , maxSizeMB_(maxSizeMB)
    , currentPath_(logDir_ / fileName_)
{
    SPDLOG_INFO("[SecurityEventLogger] logDir={}, fileName={}, maxSizeMB={}",
                logDir_.string(), fileName_, maxSizeMB_);
}

SecurityEventLogger::~SecurityEventLogger() {
    std::lock_guard lock(mutex_);
    if (file_) {
        file_->flush();
        delete file_;
        file_ = nullptr;
    }
    SPDLOG_INFO("[SecurityEventLogger] closed, total events written: {}", eventCount_);
}

// ── Public API ──────────────────────────────────────────────────────────────

tl::expected<void, std::string> SecurityEventLogger::appendEvent(const SecurityEvent& event) {
    OE_ZONE_SCOPED;
    std::lock_guard lock(mutex_);

    if (auto r = ensureFileOpen(); !r) return r;
    if (auto r = rotateIfNeeded(); !r) return r;

    nlohmann::json j = eventToJson(event);
    std::string line = j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);

    (*file_) << line << '\n';
    file_->flush();

    if (file_->fail()) {
        std::string err = "Failed to write to log file: " + currentPath_.string();
        SPDLOG_ERROR("[SecurityEventLogger] {}", err);
        return tl::unexpected(std::move(err));
    }

    ++eventCount_;
    SPDLOG_DEBUG("[SecurityEventLogger] event #{}: class={} conf={:.2f} recording={}",
                 eventCount_, event.detectedClass, event.confidence, event.recordingFile);
    return {};
}

tl::expected<std::vector<SecurityEvent>, std::string>
SecurityEventLogger::queryEvents(std::string_view fromTime, std::string_view toTime) const {
    OE_ZONE_SCOPED;
    std::lock_guard lock(mutex_);

    std::vector<SecurityEvent> results;

    // Read the current log file line by line.
    std::ifstream in(currentPath_);
    if (!in.is_open()) {
        return results;  // No log file yet — return empty.
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        auto parsed = jsonToEvent(line);
        if (!parsed) {
            SPDLOG_WARN("[SecurityEventLogger] skipping malformed line: {}", parsed.error());
            continue;
        }

        // ISO-8601 string comparison works for range filtering.
        if (parsed->timestamp >= fromTime && parsed->timestamp <= toTime) {
            results.push_back(std::move(*parsed));
        }
    }

    SPDLOG_DEBUG("[SecurityEventLogger] queryEvents [{} .. {}] returned {} events",
                 fromTime, toTime, results.size());
    return results;
}

std::filesystem::path SecurityEventLogger::currentLogPath() const {
    std::lock_guard lock(mutex_);
    return currentPath_;
}

uint64_t SecurityEventLogger::eventCount() const noexcept {
    return eventCount_;
}

// ── Private ─────────────────────────────────────────────────────────────────

tl::expected<void, std::string> SecurityEventLogger::ensureFileOpen() {
    if (file_ && file_->is_open()) return {};

    // Create directory if needed.
    std::error_code ec;
    std::filesystem::create_directories(logDir_, ec);
    if (ec) {
        std::string err = "Failed to create log directory: " + logDir_.string() + " — " + ec.message();
        SPDLOG_ERROR("[SecurityEventLogger] {}", err);
        return tl::unexpected(std::move(err));
    }

    file_ = new std::ofstream(currentPath_, std::ios::app | std::ios::binary);
    if (!file_->is_open()) {
        std::string err = "Failed to open log file: " + currentPath_.string();
        SPDLOG_ERROR("[SecurityEventLogger] {}", err);
        delete file_;
        file_ = nullptr;
        return tl::unexpected(std::move(err));
    }

    SPDLOG_INFO("[SecurityEventLogger] opened log file: {}", currentPath_.string());
    return {};
}

tl::expected<void, std::string> SecurityEventLogger::rotateIfNeeded() {
    if (!file_) return {};

    std::error_code ec;
    auto fileSize = std::filesystem::file_size(currentPath_, ec);
    if (ec) return {};  // Can't stat — skip rotation.

    auto maxBytes = static_cast<std::uintmax_t>(maxSizeMB_) * 1024 * 1024;
    if (fileSize < maxBytes) return {};

    // Close current file.
    file_->flush();
    delete file_;
    file_ = nullptr;

    // Rename with timestamp suffix.
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

    auto rotatedName = logDir_ / (fileName_ + "." + buf);
    std::filesystem::rename(currentPath_, rotatedName, ec);
    if (ec) {
        SPDLOG_WARN("[SecurityEventLogger] rotation rename failed: {}", ec.message());
    } else {
        SPDLOG_INFO("[SecurityEventLogger] rotated log to {}", rotatedName.string());
    }

    // Open a fresh file.
    return ensureFileOpen();
}

nlohmann::json SecurityEventLogger::eventToJson(const SecurityEvent& event) {
    return nlohmann::json{
        {"timestamp",      event.timestamp},
        {"class",          event.detectedClass},
        {"confidence",     event.confidence},
        {"bbox",           {{"x", event.bboxX}, {"y", event.bboxY},
                            {"w", event.bboxW}, {"h", event.bboxH}}},
        {"recording_file", event.recordingFile},
        {"frame_seq",      event.frameSeq},
    };
}

tl::expected<SecurityEvent, std::string>
SecurityEventLogger::jsonToEvent(std::string_view line) {
    try {
        auto j = nlohmann::json::parse(line);
        SecurityEvent ev;
        ev.timestamp     = j.at("timestamp").get<std::string>();
        ev.detectedClass = j.at("class").get<std::string>();
        ev.confidence    = j.at("confidence").get<float>();
        ev.bboxX         = j.at("bbox").at("x").get<float>();
        ev.bboxY         = j.at("bbox").at("y").get<float>();
        ev.bboxW         = j.at("bbox").at("w").get<float>();
        ev.bboxH         = j.at("bbox").at("h").get<float>();
        ev.recordingFile = j.value("recording_file", "");
        ev.frameSeq      = j.value("frame_seq", uint64_t{0});
        return ev;
    } catch (const std::exception& e) {
        return tl::unexpected(std::string("JSON parse error: ") + e.what());
    }
}

}  // namespace oe::security
