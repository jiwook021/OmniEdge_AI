#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "common/oe_logger.hpp"
#include "gpu/oe_cuda_check.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU & Host Diagnostics
//
// Provides:
//   CudaEventTimer  — RAII CUDA event pair for kernel/memcpy latency
//   queryVramMiB()  — free/total VRAM via cudaMemGetInfo
//   queryGpuUtil()  — GPU/memory utilization via NVML (if available)
//   hostRssKiB()    — host process RSS from /proc/self/status
//
// Usage in hot paths:
//   {
//       CudaEventTimer timer(stream, "blur_inference");
//       launchKernel<<<...>>>(stream);
//   }  // destructor logs elapsed time
// ---------------------------------------------------------------------------


// ── CUDA Event Timer (RAII) ─────────────────────────────────────────────

/// Measures elapsed GPU time between constructor and destructor.
/// Synchronizes the stream in the destructor — do NOT use inside
/// latency-critical inner loops where sync would stall the pipeline.
/// For fire-and-forget profiling, call `startAsync()` / `stopAsync()`
/// and read the result later with `elapsedMs()`.
class CudaEventTimer {
public:
    /// Construct and immediately record the start event on `stream`.
    /// If `label` is non-empty, the destructor logs the elapsed time.
    explicit CudaEventTimer(cudaStream_t stream, std::string label = {})
        : stream_(stream), label_(std::move(label))
    {
        OE_CUDA_CHECK(cudaEventCreate(&start_));
        OE_CUDA_CHECK(cudaEventCreate(&stop_));
        OE_CUDA_CHECK(cudaEventRecord(start_, stream_));
    }

    ~CudaEventTimer()
    {
        if (start_ && stop_) {
            // Record and sync are best-effort in the destructor — we must
            // never abort() during stack unwinding (OE_CUDA_CHECK can abort).
            cudaError_t err = cudaEventRecord(stop_, stream_);
            if (err == cudaSuccess) {
                err = cudaEventSynchronize(stop_);
                if (err == cudaSuccess && !label_.empty()) {
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, start_, stop_);
                    OE_LOG_DEBUG("cuda_timer: op={}, elapsed_ms={:.2f}", label_, ms);
                }
            }
        }
        // Destroy events unconditionally — ignore errors to stay noexcept-safe.
        if (start_) (void)cudaEventDestroy(start_);
        if (stop_)  (void)cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    CudaEventTimer(CudaEventTimer&& other) noexcept
        : stream_(other.stream_), start_(other.start_), stop_(other.stop_),
          label_(std::move(other.label_))
    {
        other.start_ = nullptr;
        other.stop_  = nullptr;
    }

    CudaEventTimer& operator=(CudaEventTimer&&) = delete;

    /// Record the stop event without synchronizing.
    void stopAsync()
    {
        if (stop_) {
            OE_CUDA_CHECK(cudaEventRecord(stop_, stream_));
        }
    }

    /// Synchronize and return elapsed time in milliseconds.
    /// Call after `stopAsync()` or let the destructor handle it.
    [[nodiscard]] float elapsedMs()
    {
        if (!start_ || !stop_) return -1.0f;
        OE_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        OE_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaStream_t stream_  = nullptr;
    cudaEvent_t  start_   = nullptr;
    cudaEvent_t  stop_    = nullptr;
    std::string  label_;
};

// ── VRAM Query ──────────────────────────────────────────────────────────

struct VramSnapshot {
    std::size_t freeMiB  = 0;
    std::size_t totalMiB = 0;
    std::size_t usedMiB  = 0;
};

/// Query current VRAM via cudaMemGetInfo. Returns zeros on failure.
[[nodiscard]] inline VramSnapshot queryVramMiB()
{
    constexpr std::size_t kMiB = 1048576;
    std::size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) {
        OE_LOG_WARN("cudaMemGetInfo failed: {} ({})",
                    cudaGetErrorName(err), cudaGetErrorString(err));
        return {};
    }
    return {
        .freeMiB  = freeBytes  / kMiB,
        .totalMiB = totalBytes / kMiB,
        .usedMiB  = (totalBytes - freeBytes) / kMiB,
    };
}

// ── GPU Utilization (NVML) ──────────────────────────────────────────────

struct GpuUtilization {
    uint32_t gpuPercent = 0;   ///< SM utilization [0..100]
    uint32_t memPercent = 0;   ///< Memory controller utilization [0..100]
    bool     valid      = false;
};

/// Query GPU utilization via NVML (libnvidia-ml, always linked).
/// Returns {valid=false} if the NVML runtime call fails.
[[nodiscard]] GpuUtilization queryGpuUtilization(unsigned int deviceIndex = 0);

// ── Host RAM (VmRSS) ───────────────────────────────────────────────────

/// Read VmRSS from /proc/self/status. Returns 0 on parse failure.
[[nodiscard]] std::size_t hostRssKiB();

/// Log both GPU and host memory state.
inline void logSystemResources(std::string_view label)
{
    auto vram = queryVramMiB();
    auto rss  = hostRssKiB();
    auto util = queryGpuUtilization();

    OE_LOG_INFO("resources: label={}, vram_used={}MiB, vram_free={}MiB, "
                "host_rss={}KiB, gpu_util={}%, mem_util={}%",
                label, vram.usedMiB, vram.freeMiB,
                rss,
                util.valid ? util.gpuPercent : 0u,
                util.valid ? util.memPercent : 0u);
}

