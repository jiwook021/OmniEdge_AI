#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include "common/oe_logger.hpp"

// ---------------------------------------------------------------------------
// Unit tests for OeLogger.
//
// Each test that creates files on disk uses a unique temporary directory
// and cleans it up in TearDown to avoid cross-test pollution.
// ---------------------------------------------------------------------------


namespace fs = std::filesystem;

// ===========================================================================
// Test fixture — provides a unique temp directory per test
// ===========================================================================

class OeLoggerTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		// Build a unique temp path: /tmp/oe_logger_test_<test_name>
		const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
		tempDir_ = fs::temp_directory_path()
		           / ("oe_logger_test_" + std::to_string(getpid()) + "_"
		              + std::string(info->name()));
		// Ensure clean slate.
		fs::remove_all(tempDir_);
		SPDLOG_DEBUG("SetUp: tempDir={}", tempDir_.string());
	}

	void TearDown() override
	{
		// Clean up temp files.
		std::error_code ec;
		fs::remove_all(tempDir_, ec);
	}

	fs::path tempDir_;
};

// ===========================================================================
// kLogPattern is accessible and non-empty
// ===========================================================================

TEST_F(OeLoggerTest, LogPatternConstantIsAccessible)
{
	EXPECT_FALSE(kLogPattern.empty());
	// Pattern must contain the five format fields we rely on.
	EXPECT_NE(kLogPattern.find("%Y"), std::string_view::npos) << "missing date";
	EXPECT_NE(kLogPattern.find("[%n]"), std::string_view::npos) << "missing logger name";
	EXPECT_NE(kLogPattern.find("[%!]"), std::string_view::npos) << "missing function name";
	SPDLOG_DEBUG("kLogPattern = '{}'", kLogPattern);
}

// ===========================================================================
// SingletonReturnsSameInstance
// ===========================================================================

TEST_F(OeLoggerTest, SingletonReturnsSameInstance)
{
	auto& a = OeLogger::instance();
	auto& b = OeLogger::instance();
	EXPECT_EQ(&a, &b) << "OeLogger::instance() must return the same object";
}

// ===========================================================================
// SetModuleUpdatesModuleName
// ===========================================================================

TEST_F(OeLoggerTest, SetModuleUpdatesModuleName)
{
	OeLogger::instance().setModule("test_module_xyz");
	// The spdlog logger name should now reflect the module name.
	auto logger = OeLogger::instance().get();
	ASSERT_NE(logger, nullptr);
	EXPECT_EQ(logger->name(), "test_module_xyz");
	SPDLOG_DEBUG("Logger name after setModule: '{}'", logger->name());

	// Restore a sensible default so other tests are not affected.
	OeLogger::instance().setModule("test");
}

// ===========================================================================
// InitFileCreatesLogFile
// ===========================================================================

TEST_F(OeLoggerTest, InitFileCreatesLogFile)
{
	const std::string logFileName = "test_init_file.log";

	OeLogger::instance().setModule("test_init");
	OeLogger::instance().initFile(tempDir_.string(), logFileName);

	const fs::path expectedPath = tempDir_ / logFileName;
	EXPECT_TRUE(fs::exists(expectedPath))
	    << "initFile() should create " << expectedPath;
	SPDLOG_DEBUG("Log file exists: {}", fs::exists(expectedPath));

	// Restore module name.
	OeLogger::instance().setModule("test");
}

// ===========================================================================
// DebugLogWritesToFile
// ===========================================================================

TEST_F(OeLoggerTest, DebugLogWritesToFile)
{
	const std::string logFileName = "test_debug_write.log";

	OeLogger::instance().setModule("test_debug");
	OeLogger::instance().initFile(tempDir_.string(), logFileName);

	const std::string marker = "UNIQUE_DEBUG_MARKER_42";
	OeLogger::instance().get()->debug(marker);
	OeLogger::instance().get()->flush();

	const fs::path logPath = tempDir_ / logFileName;
	ASSERT_TRUE(fs::exists(logPath)) << "Log file must exist after write";

	// Read the file and search for the marker string.
	std::ifstream fin(logPath);
	ASSERT_TRUE(fin.is_open()) << "Could not open log file for reading";

	std::string contents((std::istreambuf_iterator<char>(fin)),
	                      std::istreambuf_iterator<char>());
	EXPECT_NE(contents.find(marker), std::string::npos)
	    << "Debug message should appear in the log file";
	SPDLOG_DEBUG("Log contents length: {} bytes, marker found: {}",
	             contents.size(), contents.find(marker) != std::string::npos);

	// Restore module name.
	OeLogger::instance().setModule("test");
}

// ===========================================================================
// SetLevelEnum
// ===========================================================================

TEST_F(OeLoggerTest, SetLevelEnum)
{
	OeLogger::instance().setLevel(spdlog::level::warn);
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::warn);
	SPDLOG_DEBUG("Level after setLevel(warn): {}",
	             static_cast<int>(OeLogger::instance().get()->level()));

	// Restore.
	OeLogger::instance().setLevel(spdlog::level::debug);
}

// ===========================================================================
// SetLevelString
// ===========================================================================

TEST_F(OeLoggerTest, SetLevelString)
{
	OeLogger::instance().setLevel("err");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::err);

	OeLogger::instance().setLevel("info");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::info);
	SPDLOG_DEBUG("Level after setLevel('info'): {}",
	             static_cast<int>(OeLogger::instance().get()->level()));

	// Restore.
	OeLogger::instance().setLevel(spdlog::level::debug);
}

// ===========================================================================
// SetLevelInvalidStringFallsBackToInfo
// ===========================================================================

TEST_F(OeLoggerTest, SetLevelInvalidStringFallsBackToInfo)
{
	OeLogger::instance().setLevel("bogus_level");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::info)
	    << "Unknown level string should fall back to info, not off";
	SPDLOG_DEBUG("Level after bogus string: {}",
	             static_cast<int>(OeLogger::instance().get()->level()));

	// Restore.
	OeLogger::instance().setLevel(spdlog::level::debug);
}

// ===========================================================================
// ApplyLogLevelFromIniPerModule
// ===========================================================================

TEST_F(OeLoggerTest, ApplyLogLevelFromIniPerModule)
{
	// Write a temp INI with a per-module override.
	fs::create_directories(tempDir_);
	const fs::path iniPath = tempDir_ / "test.ini";
	{
		std::ofstream out(iniPath);
		out << "[logging]\n"
		    << "default = warn\n"
		    << "test_mod = debug\n";
	}
	SPDLOG_DEBUG("INI written to: {}", iniPath.string());

	OeLogger::instance().applyLogLevelFromIni(iniPath.string(), "test_mod");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::debug)
	    << "Per-module override should take precedence over default";

	// Restore.
	OeLogger::instance().setLevel(spdlog::level::debug);
}

// ===========================================================================
// ApplyLogLevelFromIniDefault
// ===========================================================================

TEST_F(OeLoggerTest, ApplyLogLevelFromIniDefault)
{
	fs::create_directories(tempDir_);
	const fs::path iniPath = tempDir_ / "test.ini";
	{
		std::ofstream out(iniPath);
		out << "[logging]\n"
		    << "default = warn\n";
	}
	SPDLOG_DEBUG("INI written to: {}", iniPath.string());

	OeLogger::instance().applyLogLevelFromIni(iniPath.string(), "unknown_mod");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::warn)
	    << "Module not listed should fall back to default";

	// Restore.
	OeLogger::instance().setLevel(spdlog::level::debug);
}

// ===========================================================================
// ApplyLogLevelFromIniMissingFile
// ===========================================================================

TEST_F(OeLoggerTest, ApplyLogLevelFromIniMissingFile)
{
	// Set a known level first so we can verify it's unchanged.
	OeLogger::instance().setLevel(spdlog::level::debug);

	OeLogger::instance().applyLogLevelFromIni("/nonexistent/path.ini", "mod");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::debug)
	    << "Missing INI should keep the bootstrap level unchanged";
}

// ===========================================================================
// ApplyLogLevelFromIniEmptyLoggingSection
// ===========================================================================

TEST_F(OeLoggerTest, ApplyLogLevelFromIniEmptyLoggingSection)
{
	fs::create_directories(tempDir_);
	const fs::path iniPath = tempDir_ / "empty_logging.ini";
	{
		std::ofstream out(iniPath);
		out << "[logging]\n"
		    << "; no keys at all\n";
	}

	OeLogger::instance().setLevel(spdlog::level::debug);
	OeLogger::instance().applyLogLevelFromIni(iniPath.string(), "any_mod");
	EXPECT_EQ(OeLogger::instance().get()->level(), spdlog::level::debug)
	    << "Empty [logging] section should keep bootstrap level";
	SPDLOG_DEBUG("Level unchanged after empty [logging] section: {}",
	             static_cast<int>(OeLogger::instance().get()->level()));
}
