#pragma once

#include <string>

#include <cuda_runtime.h>

#include <tl/expected.hpp>


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

