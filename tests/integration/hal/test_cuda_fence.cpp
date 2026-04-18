#include <gtest/gtest.h>

#include "gpu/cuda_fence.hpp"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CudaFence — RAII lifecycle and GPU synchronization tests
//
// These tests require a CUDA-capable GPU. They verify:
//   1. Default construction creates a valid event
//   2. Move semantics transfer ownership correctly
//   3. record() + synchronize() blocks until GPU work completes
//   4. isReady() returns correct status
//   5. Double-destroy safety (moved-from object is null)
// ---------------------------------------------------------------------------


// ===========================================================================
// 1. Construction / Destruction — "fence creates event, destructor cleans up"
// ===========================================================================

TEST(CudaFenceLifecycle, DefaultConstruction_CreatesValidEvent)
{
	CudaFence fence;
	EXPECT_TRUE(static_cast<bool>(fence));
	EXPECT_NE(fence.get(), nullptr);
}

TEST(CudaFenceLifecycle, MoveConstruction_TransfersOwnership)
{
	CudaFence original;
	cudaEvent_t originalEvent = original.get();
	ASSERT_NE(originalEvent, nullptr);

	CudaFence moved(std::move(original));
	EXPECT_EQ(moved.get(), originalEvent);
	EXPECT_FALSE(static_cast<bool>(original));  // NOLINT — testing moved-from state
	EXPECT_EQ(original.get(), nullptr);          // NOLINT
}

TEST(CudaFenceLifecycle, MoveAssignment_TransfersOwnership)
{
	CudaFence a;
	CudaFence b;
	cudaEvent_t bEvent = b.get();

	a = std::move(b);
	EXPECT_EQ(a.get(), bEvent);
	EXPECT_FALSE(static_cast<bool>(b));  // NOLINT
}

// ===========================================================================
// 2. Synchronization — "record + synchronize blocks until GPU is done"
// ===========================================================================

TEST(CudaFenceSync, RecordAndSynchronize_CompletesWithoutError)
{
	CudaFence fence;

	// Record on default stream (null stream) — there's no pending GPU work
	// so synchronize should return immediately.
	fence.record(nullptr);
	EXPECT_NO_FATAL_FAILURE(fence.synchronize());
}

TEST(CudaFenceSync, RecordOnStream_SynchronizesCorrectly)
{
	// Create a non-default stream, launch a trivial memset, fence it
	cudaStream_t stream = nullptr;
	ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

	// Allocate a small device buffer and launch work on the stream
	void* devPtr = nullptr;
	ASSERT_EQ(cudaMalloc(&devPtr, 1024), cudaSuccess);
	ASSERT_EQ(cudaMemsetAsync(devPtr, 0xAB, 1024, stream), cudaSuccess);

	// Fence after the GPU work
	CudaFence fence;
	fence.record(stream);
	fence.synchronize();

	// After synchronize, we can safely read the device memory
	uint8_t hostBuf[4] = {};
	ASSERT_EQ(cudaMemcpy(hostBuf, devPtr, 4, cudaMemcpyDeviceToHost), cudaSuccess);
	EXPECT_EQ(hostBuf[0], 0xAB);
	EXPECT_EQ(hostBuf[1], 0xAB);

	cudaFree(devPtr);
	cudaStreamDestroy(stream);
}

// ===========================================================================
// 3. isReady() — "non-blocking query of GPU completion status"
// ===========================================================================

TEST(CudaFenceQuery, IsReady_ReturnsTrueAfterSynchronize)
{
	CudaFence fence;
	fence.record(nullptr);
	fence.synchronize();
	EXPECT_TRUE(fence.isReady());
}

// ===========================================================================
// 4. Reuse — "fence can be recorded and synchronized multiple times"
// ===========================================================================

TEST(CudaFenceReuse, MultipleRecordSynchronizeCycles)
{
	CudaFence fence;

	for (int i = 0; i < 10; ++i) {
		fence.record(nullptr);
		fence.synchronize();
		EXPECT_TRUE(fence.isReady());
	}
}

