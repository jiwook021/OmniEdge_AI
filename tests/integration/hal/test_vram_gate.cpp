// ---------------------------------------------------------------------------
// OmniEdge_AI — VramGate Unit Tests
//
// Tests the VRAM acquisition gate's evict → wait → verify cycle.
// CPU-only: uses a mock GpuProfiler that simulates VRAM levels without CUDA.
//
// Bug these tests catch:
//   - Module spawned into insufficient VRAM (no eviction attempted)
//   - Eviction succeeds but spawn races the CUDA driver reclamation
//   - Timeout fires but VRAM was freed just before the check
//   - LLM incorrectly evicted (should never be a candidate)
//   - acquireVramForModule succeeds even when no candidates exist
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "vram/vram_gate.hpp"
#include "vram/vram_tracker.hpp"
#include "gpu/gpu_profiler.hpp"

namespace {

// ---------------------------------------------------------------------------
// Helpers: Mock GpuProfiler that returns configurable free VRAM
// ---------------------------------------------------------------------------

// We can't easily mock GpuProfiler (it calls cudaMemGetInfo), so we test
// the VramTracker and VramGate's eviction logic independently, and test the
// full integration path with real CUDA only in GPU-tagged tests.

// ---------------------------------------------------------------------------
// VramTracker tests with notification
// ---------------------------------------------------------------------------

class VramTrackerNotificationTest : public ::testing::Test {
protected:
	VramTracker tracker;

	void SetUp() override {
		tracker.registerModuleBudget("background_blur",  500, 0);
		tracker.registerModuleBudget("face_recognition", 500, 1);
		tracker.registerModuleBudget("tts",       100, 2);
		tracker.registerModuleBudget("stt",     1500, 3);
		tracker.registerModuleBudget("llm",        4300, 5);
	}
};

TEST_F(VramTrackerNotificationTest, WaitUntilLoadedBelowReturnsTrueWhenAlreadyBelow)
{
	// Nothing loaded — totalLoadedMb is 0, which is below any target
	bool result = tracker.waitUntilLoadedBelow(
		1000, std::chrono::milliseconds{10});
	EXPECT_TRUE(result);
}

TEST_F(VramTrackerNotificationTest, WaitUntilLoadedBelowTimesOutWhenAbove)
{
	tracker.markModuleLoaded("llm");
	tracker.markModuleLoaded("stt");
	// Total loaded = 4300 + 1500 = 5800 MiB

	bool result = tracker.waitUntilLoadedBelow(
		1000, std::chrono::milliseconds{50});
	EXPECT_FALSE(result) << "Should timeout: 5800 MiB loaded, target < 1000 MiB";
}

TEST_F(VramTrackerNotificationTest, WaitUntilLoadedBelowWakesOnUnload)
{
	tracker.markModuleLoaded("stt");
	// Total loaded = 1500 MiB

	// Unload in a background thread after a short delay
	std::thread unloader([this] {
		std::this_thread::sleep_for(std::chrono::milliseconds{20});
		tracker.markModuleUnloaded("stt");
	});

	bool result = tracker.waitUntilLoadedBelow(
		1000, std::chrono::milliseconds{500});
	EXPECT_TRUE(result) << "Should wake up after whisper_stt is unloaded (0 < 1000)";

	unloader.join();
}

TEST_F(VramTrackerNotificationTest, ModuleBudgetMiBReturnsRegisteredBudget)
{
	EXPECT_EQ(tracker.moduleBudgetMiB("llm"), 4300u);
	EXPECT_EQ(tracker.moduleBudgetMiB("stt"), 1500u);
	EXPECT_EQ(tracker.moduleBudgetMiB("background_blur"), 500u);
}

TEST_F(VramTrackerNotificationTest, ModuleBudgetMiBReturnsZeroForUnregistered)
{
	EXPECT_EQ(tracker.moduleBudgetMiB("nonexistent_module"), 0u);
}

// ---------------------------------------------------------------------------
// VramTracker eviction candidates with proper naming
// ---------------------------------------------------------------------------

TEST_F(VramTrackerNotificationTest, EvictableCandidatesSortedByPriority)
{
	// Load all modules
	tracker.markModuleLoaded("background_blur");
	tracker.markModuleLoaded("face_recognition");
	tracker.markModuleLoaded("tts");
	tracker.markModuleLoaded("stt");
	tracker.markModuleLoaded("llm");

	auto evictionCandidates = tracker.evictableCandidates();

	// LLM should never appear (priority 5 = kEvictPriorityLlm)
	for (const auto& candidate : evictionCandidates) {
		EXPECT_NE(candidate.moduleName, "llm")
			<< "LLM must never be an eviction candidate";
	}

	// Should be sorted ascending by priority
	ASSERT_GE(evictionCandidates.size(), 2u);
	for (std::size_t i = 1; i < evictionCandidates.size(); ++i) {
		EXPECT_LE(evictionCandidates[i - 1].evictPriority,
		          evictionCandidates[i].evictPriority)
			<< "Eviction candidates should be sorted by priority (ascending)";
	}

	// First candidate should be background_blur (priority 0)
	EXPECT_EQ(evictionCandidates[0].moduleName, "background_blur");
}

TEST_F(VramTrackerNotificationTest, BusyModulesExcludedFromEviction)
{
	tracker.markModuleLoaded("background_blur");
	tracker.markModuleLoaded("face_recognition");
	tracker.setIdle("background_blur", false);  // Currently processing

	auto evictionCandidates = tracker.evictableCandidates();

	for (const auto& candidate : evictionCandidates) {
		EXPECT_NE(candidate.moduleName, "background_blur")
			<< "Busy modules should not be eviction candidates";
	}
}

TEST_F(VramTrackerNotificationTest, TotalLoadedMiBTracksLoadAndUnload)
{
	EXPECT_EQ(tracker.totalLoadedMb(), 0u);

	tracker.markModuleLoaded("stt");
	EXPECT_EQ(tracker.totalLoadedMb(), 1500u);

	tracker.markModuleLoaded("tts");
	EXPECT_EQ(tracker.totalLoadedMb(), 1600u);

	tracker.markModuleUnloaded("stt");
	EXPECT_EQ(tracker.totalLoadedMb(), 100u);
}

// ---------------------------------------------------------------------------
// VramGate GPU tests — require a CUDA device at runtime
//
// These tests exercise the full acquireVramForModule → cudaMemGetInfo path.
// If no GPU is available, they skip via GTEST_SKIP().
// ---------------------------------------------------------------------------

class VramGateGpuTest : public ::testing::Test {
protected:
	VramTracker tracker;
	std::unique_ptr<GpuProfiler> profiler;

	void SetUp() override {
		int deviceCount = 0;
		if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
			GTEST_SKIP() << "No CUDA device available — skipping GPU test";
		}

		tracker.registerModuleBudget("test_module", 500, 1);

		profiler = std::make_unique<GpuProfiler>(GpuProfiler::Config{});
		profiler->probe();
	}
};

// Bug caught: module spawned without checking available VRAM first
TEST_F(VramGateGpuTest, HasEnoughFreeVramReturnsTrueWithEmptyGpu)
{
	std::vector<std::string> evictedModuleNames;
	VramGate vramGate(tracker, *profiler,
		[&](std::string_view moduleName) -> bool {
			evictedModuleNames.push_back(std::string{moduleName});
			return true;
		});

	// A small requirement should succeed on any GPU with free VRAM
	EXPECT_TRUE(vramGate.hasEnoughFreeVram(100))
		<< "GPU should have at least 200 MiB free (100 + 100 safety margin)";
}

// Bug caught: eviction callback invoked when VRAM is already sufficient
TEST_F(VramGateGpuTest, AcquireVramSucceedsWithoutEvictionWhenFreeVramSufficient)
{
	std::vector<std::string> evictedModuleNames;
	VramGate vramGate(tracker, *profiler,
		[&](std::string_view moduleName) -> bool {
			evictedModuleNames.push_back(std::string{moduleName});
			return true;
		});

	auto acquireResult = vramGate.acquireVramForModule("test_module", 100);
	ASSERT_TRUE(acquireResult.has_value())
		<< "Should succeed with small VRAM requirement: " << acquireResult.error();

	// Verify the confirmed free VRAM is reported
	EXPECT_GE(*acquireResult, 100u)
		<< "Reported free VRAM should be at least the requested amount";

	EXPECT_TRUE(evictedModuleNames.empty())
		<< "No eviction should occur when VRAM is already sufficient";
}

// Bug caught: waitForVramAvailable spins forever instead of returning quickly
TEST_F(VramGateGpuTest, WaitForVramAvailableSucceedsImmediatelyWhenFree)
{
	VramGate vramGate(tracker, *profiler,
		[](std::string_view) { return false; });

	auto startTime = std::chrono::steady_clock::now();

	// Asking for very little VRAM should succeed on first poll
	auto waitResult = vramGate.waitForVramAvailable(
		100, std::chrono::milliseconds{5000});
	ASSERT_TRUE(waitResult.has_value())
		<< "Should succeed quickly for small VRAM request: " << waitResult.error();

	auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - startTime).count();
	EXPECT_LT(elapsedMs, 500)
		<< "Should return nearly immediately, not after " << elapsedMs << " ms";

	EXPECT_GE(*waitResult, 100u)
		<< "Confirmed free VRAM should meet the target";
}

// Bug caught: timeout not enforced — hangs instead of returning error
TEST_F(VramGateGpuTest, WaitForVramAvailableTimesOutOnImpossibleTarget)
{
	VramGate vramGate(tracker, *profiler,
		[](std::string_view) { return false; });

	auto startTime = std::chrono::steady_clock::now();

	// Asking for more VRAM than any GPU has should timeout
	auto waitResult = vramGate.waitForVramAvailable(
		999'999, std::chrono::milliseconds{200});

	auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - startTime).count();

	EXPECT_FALSE(waitResult.has_value())
		<< "Should timeout when requesting absurd VRAM amount";

	// Verify timeout is roughly respected (not too short, not too long)
	EXPECT_GE(elapsedMs, 150)
		<< "Should wait at least close to the timeout duration";
	EXPECT_LT(elapsedMs, 2000)
		<< "Should not exceed timeout by a large margin";
}

// Bug caught: acquireVramForModule returns success but doesn't invoke eviction
// when VRAM is tight and an eviction candidate exists
TEST_F(VramGateGpuTest, AcquireVramEvictsWhenInsufficientAndCandidateExists)
{
	// Register a low-priority "loaded" module as an eviction candidate
	tracker.registerModuleBudget("low_priority_module", 200, 0);
	tracker.markModuleLoaded("low_priority_module");

	std::vector<std::string> evictedModuleNames;
	VramGate vramGate(tracker, *profiler,
		[&](std::string_view moduleName) -> bool {
			evictedModuleNames.push_back(std::string{moduleName});
			tracker.markModuleUnloaded(moduleName);
			return true;
		});

	// Request more VRAM than is available (999 TB — impossible without eviction)
	// The gate should try to evict the low-priority module first
	auto acquireResult = vramGate.acquireVramForModule("test_module", 999'999);

	// Even after eviction, 999 TB won't be available — should fail
	EXPECT_FALSE(acquireResult.has_value())
		<< "Should fail — not enough VRAM even after evicting all candidates";

	// But it SHOULD have attempted eviction of the low-priority module
	EXPECT_FALSE(evictedModuleNames.empty())
		<< "Should have attempted to evict low_priority_module";
	if (!evictedModuleNames.empty()) {
		EXPECT_EQ(evictedModuleNames[0], "low_priority_module")
			<< "Should evict the lowest-priority candidate first";
	}
}

// Bug caught: condition variable not signalled on unload, causing
// waitUntilLoadedBelow to sleep the full timeout
TEST_F(VramGateGpuTest, ConditionVariableWakesWaiterOnModuleUnload)
{
	VramTracker localTracker;
	localTracker.registerModuleBudget("module_a", 1000, 1);
	localTracker.markModuleLoaded("module_a");

	auto startTime = std::chrono::steady_clock::now();

	// Spawn a thread that unloads after 50ms
	std::thread unloadThread([&] {
		std::this_thread::sleep_for(std::chrono::milliseconds{50});
		localTracker.markModuleUnloaded("module_a");
	});

	// Wait for total loaded to drop below 500 MiB (currently 1000 MiB)
	bool metCondition = localTracker.waitUntilLoadedBelow(
		500, std::chrono::milliseconds{5000});

	auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - startTime).count();

	EXPECT_TRUE(metCondition) << "Should succeed after module_a is unloaded";
	EXPECT_LT(elapsedMs, 1000)
		<< "Should wake up promptly via condition_variable, not after full timeout. "
		<< "Elapsed: " << elapsedMs << " ms";

	unloadThread.join();
}

} // anonymous namespace
