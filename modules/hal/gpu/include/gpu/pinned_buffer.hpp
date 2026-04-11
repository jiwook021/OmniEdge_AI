#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "common/platform_detect.hpp"


struct PinnedDeleter {
    void operator()(uint8_t* ptr) const noexcept
    {
        if (ptr) {
            static_cast<void>(cudaFreeHost(ptr));
        }
    }
};

class PinnedStagingBuffer {
public:
    explicit PinnedStagingBuffer(std::size_t bytes)
        : bufferSize_(bytes)
    {
        if (bytes == 0) throw std::invalid_argument("[PinnedStagingBuffer] bytes must be > 0");

        void* rawMemory = nullptr;
        const cudaError_t err = cudaHostAlloc(&rawMemory, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::format("[PinnedStagingBuffer] cudaHostAlloc({} bytes) failed: {}",
                            bytes, cudaGetErrorString(err)));
        }
        pinnedMemory_ = std::unique_ptr<uint8_t[], PinnedDeleter>(static_cast<uint8_t*>(rawMemory));
    }

    PinnedStagingBuffer(const PinnedStagingBuffer&)            = delete;
    PinnedStagingBuffer& operator=(const PinnedStagingBuffer&) = delete;

    PinnedStagingBuffer(PinnedStagingBuffer&& other) noexcept
        : pinnedMemory_(std::move(other.pinnedMemory_))
        , bufferSize_(std::exchange(other.bufferSize_, 0))
    {}

    PinnedStagingBuffer& operator=(PinnedStagingBuffer&& other) noexcept
    {
        if (this != &other) {
            pinnedMemory_ = std::move(other.pinnedMemory_);
            bufferSize_   = std::exchange(other.bufferSize_, 0);
        }
        return *this;
    }

    [[nodiscard]] uint8_t*    data() const noexcept { return pinnedMemory_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return bufferSize_; }

private:
    std::unique_ptr<uint8_t[], PinnedDeleter> pinnedMemory_;
    std::size_t  bufferSize_ = 0;
};

