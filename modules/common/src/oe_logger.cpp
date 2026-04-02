#include "common/oe_logger.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "INIReader.h"
#include "common/oe_tracy.hpp"


namespace {

/// Session separator banner written once per init.
constexpr std::string_view kLogSessionSeparator =
    "\n"
    "========================================================================\n"
    "  OmniEdge_AI — New Session\n";

} // anonymous namespace

// ---------------------------------------------------------------------------
// Private helper — single source of truth for logger defaults.
// ---------------------------------------------------------------------------

void OeLogger::applyDefaults(spdlog::level::level_enum flushLevel)
{
	logger_->set_level(spdlog::level::debug);
	logger_->set_pattern(std::string{kLogPattern});
	logger_->flush_on(flushLevel);
}

// ---------------------------------------------------------------------------
// Construction / singleton
// ---------------------------------------------------------------------------

OeLogger::OeLogger()
{
	// Bootstrap with a stdout-only logger until initFile() is called.
	auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	logger_ = std::make_shared<spdlog::logger>(moduleName_, consoleSink);
	applyDefaults(spdlog::level::warn);
}

OeLogger& OeLogger::instance()
{
	static OeLogger sInstance;
	return sInstance;
}

// ---------------------------------------------------------------------------
// Module identity
// ---------------------------------------------------------------------------

void OeLogger::setModule(std::string_view moduleName)
{
	moduleName_ = moduleName;

	// Recreate the logger with the new name so %n reflects the module.
	auto sinks = logger_->sinks();
	logger_ = std::make_shared<spdlog::logger>(
	    std::string{moduleName}, sinks.begin(), sinks.end());
	applyDefaults(spdlog::level::warn);
}

// ---------------------------------------------------------------------------
// Log directory resolution
// ---------------------------------------------------------------------------

std::string OeLogger::resolveLogDir()
{
	const char* envDir = std::getenv("OE_LOG_DIR");
	if (envDir != nullptr && envDir[0] != '\0') {
		return std::string{envDir};
	}
	return (std::filesystem::current_path() / std::string{kDefaultLogDir}).string();
}

// ---------------------------------------------------------------------------
// File sink initialization
// ---------------------------------------------------------------------------

void OeLogger::initFile(std::string_view logDir, std::string_view fileName)
{
	if (fileInitialized_ && !logOutputPath_.empty()) {
		logger_->info("Module attached: {} (PID {})", moduleName_, getpid());
		return;
	}

	std::string resolvedDir = logDir.empty() ? resolveLogDir()
	                                         : std::string{logDir};
	std::filesystem::path dir{resolvedDir};
	std::filesystem::create_directories(dir);

	std::filesystem::path filePath = dir / std::string{
	    fileName.empty() ? kConsolidatedLogFile : fileName};
	logOutputPath_ = filePath.string();

	// Build the logger with file sink.  Only add the console sink when
	// stdout is an interactive terminal — when run_all.sh launches modules
	// it redirects stdout to the same omniedge.log, which would duplicate
	// every line (once via the file sink, once via the redirected console sink).
	try {
		auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
		    logOutputPath_, /*truncate=*/false);

		std::vector<spdlog::sink_ptr> sinks;
		if (::isatty(STDOUT_FILENO)) {
			sinks.push_back(
			    std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
		}
		sinks.push_back(fileSink);
		logger_ = std::make_shared<spdlog::logger>(
		    std::string{moduleName_}, sinks.begin(), sinks.end());

		// Flush every message — multiple processes share this file and
		// buffered writes cause interleaved/duplicated lines on WSL2.
		applyDefaults(spdlog::level::debug);

		fileInitialized_ = true;

		// Write session banner.
		logger_->info("{}", kLogSessionSeparator);
		logger_->info("  Started — Module: {} | PID: {} | Log: {}",
		              moduleName_, getpid(), logOutputPath_);
		logger_->info(
		    "========================================================================");
		logger_->flush();

	} catch (const spdlog::spdlog_ex& ex) {
		std::cerr << "[OeLogger] WARN: failed to open log file: "
		          << logOutputPath_ << " (" << ex.what() << ")\n";
		logOutputPath_.clear();
	}
}

// ---------------------------------------------------------------------------
// Level management
// ---------------------------------------------------------------------------

void OeLogger::setLevel(spdlog::level::level_enum level)
{
	logger_->set_level(level);
	// Keep aggressive flushing for debug/trace (WSL2 multi-process safety).
	// Otherwise flush on warn to reduce I/O overhead.
	if (level <= spdlog::level::debug) {
		logger_->flush_on(level);
	} else {
		logger_->flush_on(spdlog::level::warn);
	}
}

void OeLogger::setLevel(std::string_view levelStr)
{
	auto level = spdlog::level::from_str(std::string{levelStr});
	// from_str returns off for unrecognised strings — catch that mistake.
	if (level == spdlog::level::off && levelStr != "off") {
		logger_->warn("unknown_log_level: '{}' — defaulting to info", levelStr);
		level = spdlog::level::info;
	}
	setLevel(level);
}

// ---------------------------------------------------------------------------
// INI-based log level configuration
// ---------------------------------------------------------------------------

void OeLogger::applyLogLevelFromIni(std::string_view iniPath,
                                     std::string_view moduleName)
{
	INIReader reader{std::string{iniPath}};
	if (reader.ParseError() < 0) {
		logger_->info(
		    "log_config: ini not found ({}), keeping bootstrap level",
		    iniPath);
		return;
	}

	// Priority: per-module override > default > keep bootstrap (debug).
	std::string level = reader.Get("logging", std::string{moduleName}, "");
	if (level.empty()) {
		level = reader.Get("logging", "default", "");
	}
	if (level.empty()) {
		logger_->info("log_config: no level configured, keeping bootstrap level");
		return;
	}

	setLevel(std::string_view{level});
	logger_->info("log_config: module={}, level={}", moduleName, level);
}

