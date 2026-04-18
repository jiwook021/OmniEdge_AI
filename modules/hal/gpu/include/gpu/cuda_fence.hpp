#pragma once

#include <cuda_runtime.h>

#include "gpu/oe_cuda_check.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — CudaFence: RAII GPU completion fence
//
// Lightweight wrapper around a cudaEvent_t used as a GPU→host synchronization
// barrier. Created once at module init, reused every frame:
//
//   CudaFence fence;                         // allocates event
//   // ... enqueue GPU work on stream ...
//   fence.record(stream);                    // mark completion point
//   fence.synchronize();                     // block until GPU is done
//   shmBuf.flipSlot();                       // safe — GPU finished writing
//
// Uses cudaEventDisableTiming to avoid measurement overhead. The event is
// destroyed in the destructor.
//
// Move-only. Not thread-safe — caller serialises access (same as the
// producer node's processFrame loop).
// ---------------------------------------------------------------------------


class CudaFence {
public:
	CudaFence()
	{
		OE_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
	}

	~CudaFence()
	{
		if (event_) {
			cudaEventDestroy(event_);
		}
	}

	CudaFence(CudaFence&& other) noexcept
		: event_(other.event_)
	{
		other.event_ = nullptr;
	}

	CudaFence& operator=(CudaFence&& other) noexcept
	{
		if (this != &other) {
			if (event_) { cudaEventDestroy(event_); }
			event_ = other.event_;
			other.event_ = nullptr;
		}
		return *this;
	}

	CudaFence(const CudaFence&)            = delete;
	CudaFence& operator=(const CudaFence&) = delete;

	/// Record the fence on the given stream. The fence will be signalled
	/// when all preceding work on `stream` has completed.
	void record(cudaStream_t stream)
	{
		OE_CUDA_CHECK(cudaEventRecord(event_, stream));
	}

	/// Block the calling (host) thread until the recorded event completes.
	/// Must be called after record().
	void synchronize()
	{
		OE_CUDA_CHECK(cudaEventSynchronize(event_));
	}

	/// Non-blocking query: returns true if the GPU has completed the
	/// recorded work, false if still in progress.
	[[nodiscard]] bool isReady() const noexcept
	{
		return cudaEventQuery(event_) == cudaSuccess;
	}

	/// Make another CUDA stream wait for this fence before proceeding.
	/// Used for cross-stream GPU-side dependencies without blocking the CPU.
	///
	/// Example: after recording on streamA, call waitOn(streamB) to make
	/// streamB's subsequent kernels wait until streamA's work is done.
	void waitOn(cudaStream_t otherStream)
	{
		OE_CUDA_CHECK(cudaStreamWaitEvent(otherStream, event_, 0));
	}

	/// Access the underlying event (for advanced usage).
	[[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

	[[nodiscard]] explicit operator bool() const noexcept { return event_ != nullptr; }

private:
	cudaEvent_t event_{nullptr};
};

