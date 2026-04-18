#include <gtest/gtest.h>

#include "gpu/gpu_profiler.hpp"

namespace {

// Tag GPU-dependent tests so CI can filter them.
class GpuProfilerTest : public ::testing::Test {
protected:
	GpuProfiler::Config makeConfig()
	{
		return {
			.deviceId        = 0,
			.overrideProfile = "",
			.headroomMb      = 500,
		};
	}
};

// ── probe (requires CUDA GPU) ────────────────────────────────────────────

TEST_F(GpuProfilerTest, ProbeSucceeds)
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
		GTEST_SKIP() << "No CUDA device available";
	}

	auto cfg = makeConfig();
	GpuProfiler profiler(cfg);

	EXPECT_NO_THROW(profiler.probe());
	EXPECT_NE(profiler.tier(), GpuTier::kMinimal);
	EXPECT_GT(profiler.vramInfo().totalMb(), 0U);
}

TEST_F(GpuProfilerTest, LiveFreeMbReturnsPositive)
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
		GTEST_SKIP() << "No CUDA device available";
	}

	auto cfg = makeConfig();
	GpuProfiler profiler(cfg);
	profiler.probe();

	auto free = profiler.liveFreeMb();
	EXPECT_GT(free, 0U);
}

TEST_F(GpuProfilerTest, PressureLevelIsInRange)
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
		GTEST_SKIP() << "No CUDA device available";
	}

	auto cfg = makeConfig();
	GpuProfiler profiler(cfg);
	profiler.probe();

	auto level = profiler.pressureLevel();
	EXPECT_GE(level, 0);
	EXPECT_LE(level, 2);
}

// ── profile override ────────────────────────────────────────────────────

TEST_F(GpuProfilerTest, OverrideProfileForcesTier)
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
		GTEST_SKIP() << "No CUDA device available";
	}

	auto cfg = makeConfig();
	cfg.overrideProfile = "standard";  // Force standard tier
	GpuProfiler profiler(cfg);
	profiler.probe();

	EXPECT_EQ(profiler.tier(), GpuTier::kStandard);
}

} // namespace
