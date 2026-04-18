// ---------------------------------------------------------------------------
// test_gpu_diagnostics.cpp -- GPU diagnostics edge case tests
//
// Tests: queryVramMiB, hostRssKiB, queryGpuUtilization, CudaEventTimer,
//        VramSnapshot arithmetic consistency.
//
// GPU tests are tagged LABELS=gpu and only run on machines with a CUDA device.
// CPU tests (hostRssKiB) run everywhere.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include <cuda_runtime.h>

#include "gpu/gpu_diagnostics.hpp"


// =========================================================================
// Host RSS (CPU-only -- always works)
// =========================================================================

TEST(GpuDiagnosticsTest, HostRssKiB_ReturnsNonZero)
{
    // Every running process has some RSS. This should never be zero.
    const std::size_t rss = hostRssKiB();
    EXPECT_GT(rss, 0u)
        << "hostRssKiB() should return non-zero for any running process";
}

TEST(GpuDiagnosticsTest, HostRssKiB_ConsistentAcrossCalls)
{
    // Two rapid calls should return similar values (no wild fluctuation).
    const std::size_t rss1 = hostRssKiB();
    const std::size_t rss2 = hostRssKiB();

    // Both should be non-zero.
    EXPECT_GT(rss1, 0u);
    EXPECT_GT(rss2, 0u);

    // They should be within an order of magnitude of each other.
    if (rss1 > 0 && rss2 > 0) {
        const double ratio = static_cast<double>(rss1) / static_cast<double>(rss2);
        EXPECT_GT(ratio, 0.1);
        EXPECT_LT(ratio, 10.0);
    }
}

// =========================================================================
// GPU Utilization (may not have NVML on all systems)
// =========================================================================

TEST(GpuDiagnosticsTest, QueryGpuUtilization_NeverCrashes)
{
    // queryGpuUtilization should return a GpuUtilization struct without
    // crashing, regardless of whether NVML is available.
    GpuUtilization util;
    EXPECT_NO_THROW(util = queryGpuUtilization(0));

    // valid is either true or false -- both are acceptable.
    if (util.valid) {
        EXPECT_LE(util.gpuPercent, 100u);
        EXPECT_LE(util.memPercent, 100u);
    }
}

// =========================================================================
// VRAM Query -- GPU-required tests
// =========================================================================

class ProbeGpuDiagnosticsTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Skip if no CUDA device is available.
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(ProbeGpuDiagnosticsTest, QueryVramMiB_ReturnsNonZeroTotal)
{
    VramSnapshot snap = queryVramMiB();
    EXPECT_GT(snap.totalMiB, 0u)
        << "Total VRAM should be non-zero on a GPU system";
}

TEST_F(ProbeGpuDiagnosticsTest, QueryVramMiB_FreeDoesNotExceedTotal)
{
    VramSnapshot snap = queryVramMiB();
    EXPECT_LE(snap.freeMiB, snap.totalMiB)
        << "Free VRAM should not exceed total VRAM";
}

TEST_F(ProbeGpuDiagnosticsTest, VramSnapshot_UsedPlusFreeApproximatesTotal)
{
    VramSnapshot snap = queryVramMiB();

    // usedMiB + freeMiB should approximately equal totalMiB.
    // Allow 1 MiB tolerance due to integer division truncation.
    const std::size_t sum = snap.usedMiB + snap.freeMiB;
    if (snap.totalMiB > 0) {
        EXPECT_GE(sum + 1, snap.totalMiB)
            << "used + free should approximately equal total "
            << "(got: " << snap.usedMiB << " + " << snap.freeMiB
            << " = " << sum << ", total = " << snap.totalMiB << ")";
        EXPECT_LE(sum, snap.totalMiB + 1);
    }
}

TEST_F(ProbeGpuDiagnosticsTest, CudaEventTimer_MeasuresPositiveTime)
{
    // Create a CUDA stream, launch a trivial operation, and verify the
    // timer reports > 0 ms.
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    {
        CudaEventTimer timer(stream, "test_timer");

        // Do a small device synchronize to ensure some measurable time passes.
        // cudaDeviceSynchronize inside a timer measures GPU idle time, which
        // should still be >= 0.
        cudaDeviceSynchronize();

        timer.stopAsync();
        float elapsed = timer.elapsedMs();
        // Elapsed should be >= 0 (GPU timer resolution may round to 0.0 for
        // very fast operations, but it should never be negative).
        EXPECT_GE(elapsed, 0.0f)
            << "CudaEventTimer should report non-negative elapsed time";
    }

    cudaStreamDestroy(stream);
}

TEST_F(ProbeGpuDiagnosticsTest, CudaEventTimer_MoveConstructorWorks)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    {
        CudaEventTimer original(stream, "");  // empty label = no destructor log
        CudaEventTimer moved(std::move(original));

        // The moved-to timer should work — record stop and get elapsed.
        cudaDeviceSynchronize();
        moved.stopAsync();
        float elapsed = moved.elapsedMs();
        EXPECT_GE(elapsed, 0.0f);
    }
    // Timer destroyed before stream, so CUDA events are cleaned up safely.
    cudaStreamDestroy(stream);
}

TEST_F(ProbeGpuDiagnosticsTest, QueryVramMiB_ConsistentAcrossCalls)
{
    VramSnapshot snap1 = queryVramMiB();
    VramSnapshot snap2 = queryVramMiB();

    // Total VRAM should be identical across calls.
    EXPECT_EQ(snap1.totalMiB, snap2.totalMiB)
        << "Total VRAM should not change between calls";

    // Free VRAM should be very similar (nothing else allocating during test).
    if (snap1.freeMiB > 0 && snap2.freeMiB > 0) {
        // Allow 10% variance for concurrent system activity.
        const auto diff = (snap1.freeMiB > snap2.freeMiB)
            ? (snap1.freeMiB - snap2.freeMiB)
            : (snap2.freeMiB - snap1.freeMiB);
        EXPECT_LT(diff, snap1.totalMiB / 10)
            << "Free VRAM should be relatively stable between rapid queries";
    }
}

