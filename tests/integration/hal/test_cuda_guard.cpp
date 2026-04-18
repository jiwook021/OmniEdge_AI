// ---------------------------------------------------------------------------
// test_cuda_guard.cpp — VRAM Pre-Flight Guard Tests
//
// Validates ensureVramAvailable() and ensureVramAvailableMiB() against real
// CUDA hardware.  GPU tests skip via GTEST_SKIP() when no device is present.
//
// Bugs these tests catch:
//   - Comparison direction reversed (free > required instead of free < required)
//   - MiB-to-bytes conversion overflow or off-by-factor error
//   - cudaMemGetInfo failure not propagated (error swallowed, returns success)
//   - Uninformative error message (no free/required numbers for debugging)
//   - Zero-byte edge case rejected when it should succeed
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "gpu/cuda_guard.hpp"
#include "gpu/gpu_diagnostics.hpp"
#include "common/constants/memory_constants.hpp"


// ---------------------------------------------------------------------------
// GPU fixture — skips if no CUDA device is available
// ---------------------------------------------------------------------------

class CudaGuardGpuTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		int deviceCount = 0;
		if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
			GTEST_SKIP() << "No CUDA device available";
		}
	}
};

// Bug caught: comparison direction reversed — always returns success
TEST_F(CudaGuardGpuTest, SmallRequestSucceeds)
{
	// 1 MiB should be available on any GPU with a running display driver.
	auto result = ensureVramAvailable(1 * kBytesPerMebibyte);
	EXPECT_TRUE(result.has_value())
		<< "1 MiB request should succeed on any GPU: " << result.error();
}

// Bug caught: function doesn't actually check free VRAM (no-op gate)
TEST_F(CudaGuardGpuTest, ImpossibleRequestFails)
{
	// 999 TB — no GPU has this much VRAM.
	constexpr std::size_t kAbsurdBytes = 999ULL * 1024 * 1024 * 1024 * 1024;
	auto result = ensureVramAvailable(kAbsurdBytes);
	EXPECT_FALSE(result.has_value())
		<< "999 TB request should always fail";
}

// Bug caught: error message missing diagnostic numbers
TEST_F(CudaGuardGpuTest, ErrorMessageContainsVramInfo)
{
	constexpr std::size_t kAbsurdBytes = 999ULL * 1024 * 1024 * 1024 * 1024;
	auto result = ensureVramAvailable(kAbsurdBytes);
	ASSERT_FALSE(result.has_value());

	const std::string& errorMsg = result.error();
	EXPECT_NE(errorMsg.find("insufficient VRAM"), std::string::npos)
		<< "Error should contain 'insufficient VRAM', got: " << errorMsg;
	EXPECT_NE(errorMsg.find("MiB"), std::string::npos)
		<< "Error should contain 'MiB' units, got: " << errorMsg;
	EXPECT_NE(errorMsg.find("need"), std::string::npos)
		<< "Error should contain the required amount, got: " << errorMsg;
	EXPECT_NE(errorMsg.find("free"), std::string::npos)
		<< "Error should contain the free amount, got: " << errorMsg;
}

// Bug caught: off-by-one where 0 bytes fails strict < comparison
TEST_F(CudaGuardGpuTest, ZeroBytesAlwaysSucceeds)
{
	auto result = ensureVramAvailable(0);
	EXPECT_TRUE(result.has_value())
		<< "Zero-byte request should always succeed: " << result.error();
}

// Bug caught: MiB overload multiplies wrong (e.g. * 1000 instead of * 1024^2)
TEST_F(CudaGuardGpuTest, MiBOverloadSmallRequestSucceeds)
{
	auto result = ensureVramAvailableMiB(1);
	EXPECT_TRUE(result.has_value())
		<< "1 MiB request via MiB overload should succeed: " << result.error();
}

// Bug caught: MiB overload conversion factor wrong — passes impossible request
TEST_F(CudaGuardGpuTest, MiBOverloadImpossibleRequestFails)
{
	constexpr std::size_t kAbsurdMiB = 999ULL * 1024 * 1024;  // ~999 TiB
	auto result = ensureVramAvailableMiB(kAbsurdMiB);
	EXPECT_FALSE(result.has_value())
		<< "~999 TiB MiB request should fail";
}

// Bug caught: ensureVramAvailable disagrees with queryVramMiB on free VRAM
TEST_F(CudaGuardGpuTest, ConsistentWithQueryVramMiB)
{
	VramSnapshot snap = queryVramMiB();
	ASSERT_GT(snap.totalMiB, 0u) << "Need a valid VRAM snapshot";

	// A request for half the free VRAM should succeed.
	if (snap.freeMiB > 2) {
		auto halfFree = ensureVramAvailableMiB(snap.freeMiB / 2);
		EXPECT_TRUE(halfFree.has_value())
			<< "Requesting half of free VRAM (" << snap.freeMiB / 2
			<< " MiB) should succeed: " << halfFree.error();
	}

	// A request for double the total VRAM should fail.
	auto doubleTotal = ensureVramAvailableMiB(snap.totalMiB * 2);
	EXPECT_FALSE(doubleTotal.has_value())
		<< "Requesting 2x total VRAM (" << snap.totalMiB * 2
		<< " MiB) should fail";
}

// Bug caught: pre-flight passes but real allocation fails (drift test)
TEST_F(CudaGuardGpuTest, PreflightMatchesRealAllocation)
{
	// Allocate a small CUDA buffer to verify the guard's free-VRAM reading
	// reflects actual allocatable memory.
	constexpr std::size_t kAllocBytes = 64 * kBytesPerMebibyte;

	auto preCheck = ensureVramAvailable(kAllocBytes);
	if (!preCheck.has_value()) {
		GTEST_SKIP() << "Less than 64 MiB free — cannot run allocation test";
	}

	void* devPtr = nullptr;
	cudaError_t allocErr = cudaMalloc(&devPtr, kAllocBytes);
	EXPECT_EQ(allocErr, cudaSuccess)
		<< "Guard said 64 MiB was free, but cudaMalloc failed: "
		<< cudaGetErrorString(allocErr);

	if (devPtr) {
		cudaFree(devPtr);
	}
}

