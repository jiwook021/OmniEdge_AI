#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "gpu/gpu_tier.hpp"
#include "common/runtime_defaults.hpp"

// ---------------------------------------------------------------------------
// GpuTierProbe tests — CPU-only path (tier selection logic).
// probeGpu() itself requires a CUDA device and is tagged LABELS=gpu.
// selectTier(), tierName(), and parseTier() are pure CPU logic and run
// in all CI environments.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// parseTier() — parse/name round-trip and invalid input handling
// ---------------------------------------------------------------------------

TEST(ParseTierTest, RoundTripAllTiers)
{
	EXPECT_EQ(parseTier("minimal"),  GpuTier::kMinimal);
	EXPECT_EQ(parseTier("balanced"), GpuTier::kBalanced);
	EXPECT_EQ(parseTier("standard"), GpuTier::kStandard);
	EXPECT_EQ(parseTier("ultra"),    GpuTier::kUltra);
}

TEST(ParseTierTest, UnknownNameThrowsInvalidArgument)
{
	EXPECT_THROW((void)parseTier("extreme"), std::invalid_argument);
	EXPECT_THROW((void)parseTier(""),        std::invalid_argument);
	EXPECT_THROW((void)parseTier("ULTRA"),   std::invalid_argument);  // case-sensitive
}

// ---------------------------------------------------------------------------
// selectTier() — threshold boundary conditions
// ---------------------------------------------------------------------------

// Helper: build a VramInfo with a given total VRAM (in MiB). freeBytes,
// computeCapabilityMajor, computeCapabilityMinor, and gpuName are irrelevant for selectTier().
static VramInfo makeVramInfo(std::size_t totalMb)
{
	return VramInfo{
		.totalBytes = totalMb * 1024ULL * 1024ULL,
		.freeBytes  = 0,
		.computeCapabilityMajor    = 8,
		.computeCapabilityMinor    = 6,
		.gpuName    = "Test GPU",
	};
}

TEST(SelectTierTest, ExactlyUltraThresholdSelectsUltra)
{
	// Usable = kVramThresholdUltraMb exactly → kUltra
	const std::size_t headroom = 500;
	const std::size_t total    = kUltraTierThresholdMiB + headroom;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kUltra);
}

TEST(SelectTierTest, OneBelowUltraThresholdSelectsStandard)
{
	const std::size_t headroom = 500;
	const std::size_t total    = kUltraTierThresholdMiB + headroom - 1;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kStandard);
}

TEST(SelectTierTest, ExactlyStandardThresholdSelectsStandard)
{
	const std::size_t headroom = 500;
	const std::size_t total    = kStandardTierThresholdMiB + headroom;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kStandard);
}

TEST(SelectTierTest, OneBelowStandardThresholdSelectsBalanced)
{
	const std::size_t headroom = 500;
	const std::size_t total    = kStandardTierThresholdMiB + headroom - 1;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kBalanced);
}

TEST(SelectTierTest, ExactlyBalancedThresholdSelectsBalanced)
{
	const std::size_t headroom = 500;
	const std::size_t total    = kBalancedTierThresholdMiB + headroom;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kBalanced);
}

TEST(SelectTierTest, BelowBalancedThresholdSelectsMinimal)
{
	const std::size_t headroom = 500;
	const std::size_t total    = kBalancedTierThresholdMiB + headroom - 1;
	EXPECT_EQ(selectTier(makeVramInfo(total), headroom), GpuTier::kMinimal);
}

TEST(SelectTierTest, ZeroVramSelectsMinimal)
{
	// Pathological case — guard against underflow in usable calculation
	EXPECT_EQ(selectTier(makeVramInfo(0), 500), GpuTier::kMinimal);
}

TEST(SelectTierTest, TwelveGbSelectsStandard)
{
	// 12 GB total, 500 MB headroom → 11 500 MB usable → kStandard
	EXPECT_EQ(selectTier(makeVramInfo(12 * 1024), 500), GpuTier::kStandard);
}

TEST(SelectTierTest, SixteenGbSelectsUltra)
{
	// 16 GB total, 500 MB headroom → 15 884 MB usable → kUltra
	EXPECT_EQ(selectTier(makeVramInfo(16 * 1024), 500), GpuTier::kUltra);
}

// ---------------------------------------------------------------------------
// VramInfo helpers
// ---------------------------------------------------------------------------

TEST(VramInfoTest, TotalMbAndFreeMbRoundDown)
{
	VramInfo info;
	info.totalBytes = 12ULL * 1024 * 1024 * 1024;  // exactly 12 GiB
	info.freeBytes  =  4ULL * 1024 * 1024 * 1024;  // exactly 4 GiB
	EXPECT_EQ(info.totalMb(), 12 * 1024u);
	EXPECT_EQ(info.freeMb(),   4 * 1024u);
}

// ---------------------------------------------------------------------------
// probeGpu() — requires a CUDA device; tagged LABELS=gpu
// ---------------------------------------------------------------------------

TEST(ProbeGpuTest, ProbeReturnsNonZeroVram)
{
	// This test only runs on machines with a CUDA GPU (LABELS=gpu in CTest).
	// Runtime guard: skip when the binary is invoked directly without ctest -L.
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
		GTEST_SKIP() << "No CUDA device available";
	}
	const VramInfo info = probeGpu(0);
	EXPECT_GT(info.totalBytes, 0u);
	EXPECT_GT(info.totalMb(),  0u);
	EXPECT_FALSE(info.gpuName.empty());
	EXPECT_GT(info.computeCapabilityMajor, 0);
}

