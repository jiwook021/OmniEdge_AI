#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — spdlog-Based Consolidated Logger
//
// Output format (one line per event):
//   2026-04-01 12:34:56.789 [INFO ] [module] [function] message here
//
// All modules write to a single consolidated log file:
//   <project_root>/logs/omniedge.log
//
// Singleton: OeLogger::instance().  Thread-safe via spdlog internals.
// ---------------------------------------------------------------------------

#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <string_view>


inline constexpr std::string_view kConsolidatedLogFile = "omniedge.log";
inline constexpr std::string_view kDefaultLogDir       = "logs";
inline constexpr std::string_view kLogPattern =
    "%Y-%m-%d %H:%M:%S.%e [%-5l] [%n] [%!] %v";

/** Thin singleton wrapper around spdlog providing module-scoped initialization. */
class OeLogger {
public:
    /** @brief Access the global logger singleton. */
    [[nodiscard]] static OeLogger& instance();

    /**
     * @brief Set the module name used in all subsequent log lines.
     * @param moduleName  Short identifier (e.g. "omniedge_stt").
     */
    void setModule(std::string_view moduleName);

    /**
     * @brief Initialize the log file sink.
     *
     * Creates the log directory if it does not exist and opens the
     * consolidated log file in append mode.  If already initialized,
     * logs an "attached" notice and returns.
     *
     * @param logDir   Override directory (empty = use OE_LOG_DIR env or cwd/logs).
     * @param fileName Override file name (empty = "omniedge.log").
     */
    void initFile(std::string_view logDir  = "",
                  std::string_view fileName = "");

    /** @brief Resolve the log directory from OE_LOG_DIR env or cwd/logs. */
    [[nodiscard]] static std::string resolveLogDir();

    /**
     * @brief Set the runtime log level.
     * @param level  spdlog level enum value.
     */
    void setLevel(spdlog::level::level_enum level);

    /**
     * @brief Set the runtime log level from a string.
     *
     * Accepted values: "trace", "debug", "info", "warn", "err", "critical", "off".
     * Unknown strings fall back to "info" with a logged warning.
     */
    void setLevel(std::string_view levelStr);

    /**
     * @brief Read the [logging] section from an INI file and apply the
     *        effective log level for this module.
     *
     * Resolution order: per-module override > default key > hardcoded debug.
     *
     * @param iniPath     Path to the INI file (e.g. "config/omniedge.ini").
     * @param moduleName  Short module name matching INI keys (e.g. "llm").
     */
    void applyLogLevelFromIni(std::string_view iniPath,
                              std::string_view moduleName);

    /** @brief Get the underlying spdlog logger for direct use. */
    [[nodiscard]] std::shared_ptr<spdlog::logger> get() const noexcept
    { return logger_; }

    /** @brief Path of the opened log file (empty if not yet initialized). */
    [[nodiscard]] const std::string& logFilePath() const noexcept
    { return logOutputPath_; }

    OeLogger(const OeLogger&)            = delete;
    OeLogger& operator=(const OeLogger&) = delete;

private:
    OeLogger();
    ~OeLogger() = default;

    /** @brief Apply log pattern, level=debug, and flush policy to logger_. */
    void applyDefaults(spdlog::level::level_enum flushLevel);

    std::shared_ptr<spdlog::logger> logger_;
    std::string moduleName_{"omniedge"};
    std::string logOutputPath_;
    bool        fileInitialized_{false};
};


// ---------------------------------------------------------------------------
// Convenience macros — delegate to spdlog.
//
// These preserve the call-site __FILE__, __LINE__, and __func__ via spdlog's
// SPDLOG_LOGGER_* macros.  Format strings use fmt::format syntax ({}, {:.3f}).
//
// NOTE: OE_LOG_TRACE is compiled out unless the build defines
//       SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE (spdlog compile-time gate).
// ---------------------------------------------------------------------------
#define OE_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE   (::OeLogger::instance().get(), __VA_ARGS__)
#define OE_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG   (::OeLogger::instance().get(), __VA_ARGS__)
#define OE_LOG_INFO(...)     SPDLOG_LOGGER_INFO    (::OeLogger::instance().get(), __VA_ARGS__)
#define OE_LOG_WARN(...)     SPDLOG_LOGGER_WARN    (::OeLogger::instance().get(), __VA_ARGS__)
#define OE_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR   (::OeLogger::instance().get(), __VA_ARGS__)
#define OE_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(::OeLogger::instance().get(), __VA_ARGS__)
