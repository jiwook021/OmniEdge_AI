#pragma once

#include <chrono>
#include <cstdint>


[[nodiscard]] inline int64_t steadyClockNanoseconds() noexcept
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline double steadyClockMilliseconds() noexcept
{
	return static_cast<double>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count()) * 1e-6;
}

