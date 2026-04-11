#pragma once

#include <algorithm>
#include <string>

#include <cuda_runtime.h>

#include <tl/expected.hpp>


// ---------------------------------------------------------------------------
// clampCudaStreamPriority — query the device's priority range and clamp
// the requested value.  Useful when callers manage raw cudaStream_t.
//
// CUDA priorities are inverted: lower numerical value = higher priority.
// greatestPriority is typically a negative number (e.g. -5).
// ---------------------------------------------------------------------------
[[nodiscard]] inline tl::expected<int, std::string>
clampCudaStreamPriority(int requestedPriority) noexcept
{
    int leastPriority = 0, greatestPriority = 0;
    const cudaError_t err =
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    if (err != cudaSuccess) {
        return tl::unexpected(
            std::string("[CudaStream] cudaDeviceGetStreamPriorityRange failed: ") +
            cudaGetErrorString(err));
    }
    return std::clamp(requestedPriority, greatestPriority, leastPriority);
}


class CudaStream {
public:
    CudaStream() = default;

    [[nodiscard]] static tl::expected<CudaStream, std::string>
    create(int priority) noexcept
    {
        cudaStream_t rawStream{nullptr};
        const cudaError_t err =
            cudaStreamCreateWithPriority(&rawStream, cudaStreamNonBlocking, priority);
        if (err != cudaSuccess) {
            return tl::unexpected(
                std::string("[CudaStream] cudaStreamCreateWithPriority failed: ") +
                cudaGetErrorString(err));
        }
        CudaStream stream;
        stream.cudaStream_ = rawStream;
        return stream;
    }

    /// Create a CUDA stream with priority clamped to the device's supported
    /// range.  Combines cudaDeviceGetStreamPriorityRange + create() in one call.
    [[nodiscard]] static tl::expected<CudaStream, std::string>
    createClamped(int requestedPriority) noexcept
    {
        auto clamped = clampCudaStreamPriority(requestedPriority);
        if (!clamped) return tl::unexpected(clamped.error());
        return create(*clamped);
    }

    ~CudaStream()
    {
        if (cudaStream_) {
            cudaStreamDestroy(cudaStream_);
        }
    }

    CudaStream(CudaStream&& other) noexcept
        : cudaStream_(other.cudaStream_)
    {
        other.cudaStream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other) {
            if (cudaStream_) { cudaStreamDestroy(cudaStream_); }
            cudaStream_       = other.cudaStream_;
            other.cudaStream_ = nullptr;
        }
        return *this;
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return cudaStream_; }
    [[nodiscard]] explicit operator bool() const noexcept { return cudaStream_ != nullptr; }

private:
    cudaStream_t cudaStream_{nullptr};
};

