#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "common/validated_config.hpp"

// ---------------------------------------------------------------------------
// Unit tests for ConfigValidator — the accumulating-error helper used by
// every module's Config::validate() factory.
// ---------------------------------------------------------------------------


TEST(ConfigValidatorTest, AllChecksPassing)
{
	ConfigValidator v;
	v.requireRange("port", 5561, 1024, 65535);
	v.requirePositive("timeout", 100);
	v.requireRangeF("temperature", 0.7f, 0.0f, 2.0f);
	v.requireNonEmpty("path", "/some/path");
	v.requirePort("pubPort", 5563);
	EXPECT_TRUE(v.ok());
	EXPECT_TRUE(v.finish().empty());
}

TEST(ConfigValidatorTest, RequireRangeRejectsOutOfRange)
{
	ConfigValidator v;
	v.requireRange("port", 0, 1024, 65535);
	EXPECT_FALSE(v.ok());
	EXPECT_NE(v.finish().find("port"), std::string::npos);
}

TEST(ConfigValidatorTest, RequirePortRejectsBelowMinimum)
{
	ConfigValidator v;
	v.requirePort("p", 1023);
	EXPECT_FALSE(v.ok());
}

TEST(ConfigValidatorTest, RequirePortRejectsAboveMaximum)
{
	ConfigValidator v;
	v.requirePort("p", 65536);
	EXPECT_FALSE(v.ok());
}

TEST(ConfigValidatorTest, RequirePositiveRejectsZeroAndNegative)
{
	ConfigValidator v;
	v.requirePositive("a", 0);
	v.requirePositive("b", -5);
	EXPECT_FALSE(v.ok());
	auto err = v.finish();
	// Both errors should be present
	EXPECT_NE(err.find("a"), std::string::npos);
	EXPECT_NE(err.find("b"), std::string::npos);
}

TEST(ConfigValidatorTest, RequireRangeFRejectsOutOfRange)
{
	ConfigValidator v;
	v.requireRangeF("temp", 3.0f, 0.0f, 2.0f);
	EXPECT_FALSE(v.ok());
}

TEST(ConfigValidatorTest, RequireRangeFRejectsBelowRange)
{
	ConfigValidator v;
	v.requireRangeF("temp", -0.1f, 0.0f, 2.0f);
	EXPECT_FALSE(v.ok());
}

TEST(ConfigValidatorTest, RequireNonEmptyRejectsEmpty)
{
	ConfigValidator v;
	v.requireNonEmpty("path", "");
	EXPECT_FALSE(v.ok());
	EXPECT_NE(v.finish().find("path"), std::string::npos);
}

TEST(ConfigValidatorTest, MultipleErrorsAccumulate)
{
	ConfigValidator v;
	v.requirePort("port1", 0);
	v.requirePositive("timeout", -1);
	v.requireNonEmpty("path", "");
	auto err = v.finish();
	SPDLOG_DEBUG("Accumulated errors: {}", err);
	// Should contain semicolons separating errors
	EXPECT_NE(err.find(";"), std::string::npos);
	// All three fields mentioned
	EXPECT_NE(err.find("port1"), std::string::npos);
	EXPECT_NE(err.find("timeout"), std::string::npos);
	EXPECT_NE(err.find("path"), std::string::npos);
}
