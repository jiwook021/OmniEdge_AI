#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Validated Config Pattern
//
// Factory-based config validation that prevents invalid runtime configurations
// from propagating silently.  Each module's Config struct provides a static
// validate() method returning tl::expected<Config, std::string>.  Bad values
// (negative timeouts, out-of-range ports, empty paths) are caught at
// construction time, not at first use deep in the pipeline.
//
// Usage:
//   auto result = DaemonConfig::validate(raw);
//   if (!result) { spdlog::error("Config: {}", result.error()); return 1; }
//   OmniEdgeDaemon daemon(std::move(*result));
//
// The validate() function is the ONLY way to construct a validated config.
// Plain construction from a Raw struct is private (enforced by friend).
// ---------------------------------------------------------------------------

#include <string>
#include <vector>

#include <tl/expected.hpp>


/// Accumulates validation errors instead of failing on the first one.
/// Usage:
///   ConfigValidator v;
///   v.requireRange("port", port, 1024, 65535);
///   v.requirePositive("watchdogPollMs", watchdogPollMs);
///   v.requireNonEmpty("configFile", configFile);
///   if (auto err = v.finish(); !err.empty()) return tl::unexpected(err);
class ConfigValidator {
public:
	/// Check that value is within [minVal, maxVal].
	void requireRange(std::string_view field, int value, int minVal, int maxVal)
	{
		if (value < minVal || value > maxVal) {
			errors_.emplace_back(
				std::string{field} + " = " + std::to_string(value)
				+ " out of range [" + std::to_string(minVal)
				+ ", " + std::to_string(maxVal) + "]");
		}
	}

	/// Check that value is > 0.
	void requirePositive(std::string_view field, int value)
	{
		if (value <= 0) {
			errors_.emplace_back(
				std::string{field} + " = " + std::to_string(value)
				+ " must be positive");
		}
	}

	/// Check that a floating-point value is within [minVal, maxVal].
	/// Also rejects NaN (NaN comparisons silently pass range checks).
	void requireRangeF(std::string_view field, float value,
	                   float minVal, float maxVal)
	{
		if (std::isnan(value) || value < minVal || value > maxVal) {
			errors_.emplace_back(
				std::string{field} + " = " + std::to_string(value)
				+ " out of range [" + std::to_string(minVal)
				+ ", " + std::to_string(maxVal) + "]");
		}
	}

	/// Check that a string is non-empty.
	void requireNonEmpty(std::string_view field, std::string_view value)
	{
		if (value.empty()) {
			errors_.emplace_back(
				std::string{field} + " must not be empty");
		}
	}

	/// Check that a ZMQ port is in the valid range for OmniEdge (1024-65535).
	void requirePort(std::string_view field, int value)
	{
		requireRange(field, value, 1024, 65535);
	}

	/// Return accumulated errors as a single newline-separated string.
	/// Empty string means all checks passed.
	[[nodiscard]] std::string finish() const
	{
		if (errors_.empty()) return {};
		std::string result;
		for (std::size_t i = 0; i < errors_.size(); ++i) {
			if (i > 0) result += "; ";
			result += errors_[i];
		}
		return result;
	}

	/// True if no errors have been recorded.
	[[nodiscard]] bool ok() const noexcept { return errors_.empty(); }

private:
	std::vector<std::string> errors_;
};

