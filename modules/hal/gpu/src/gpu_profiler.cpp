#include "gpu/gpu_profiler.hpp"

#include <format>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

#include "common/constants/gpu_constants.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


GpuProfiler::GpuProfiler(Config config)
    : config_(std::move(config))
{
}

void GpuProfiler::probe()
{
    for (int attempt = 1; attempt <= kProbeMaxAttempts; ++attempt) {
        try {
            vramInfo_ = probeGpu(config_.deviceId);
            break;
        } catch (const std::exception& ex) {
            if (attempt == kProbeMaxAttempts) throw;
            OE_LOG_WARN("gpu_probe_retry: {}/{}, {}", attempt, kProbeMaxAttempts, ex.what());
            std::this_thread::sleep_for(kProbeRetryDelay);
        }
    }

    if (!config_.overrideProfile.empty()) {
        selectedTier_ = parseTier(config_.overrideProfile);
    } else {
        selectedTier_ = selectTier(vramInfo_, config_.headroomMb,
                                   config_.ultraTierThresholdMiB,
                                   config_.standardTierThresholdMiB,
                                   config_.balancedTierThresholdMiB);
    }

    OE_LOG_INFO("gpu: tier={}, name={}, total={}MB, free={}MB",
              tierName(selectedTier_), vramInfo_.gpuName, vramInfo_.totalMb(), vramInfo_.freeMb());
    hasProbed_ = true;
}

std::size_t GpuProfiler::liveFreeMb() const
{
    std::size_t freeVramBytes = 0;
    std::size_t totalVramBytes = 0;
    if (cudaMemGetInfo(&freeVramBytes, &totalVramBytes) != cudaSuccess) return 0;
    return freeVramBytes / kBytesPerMebibyte;
}

std::size_t GpuProfiler::liveUsedMb() const
{
    std::size_t freeVramBytes = 0;
    std::size_t totalVramBytes = 0;
    if (cudaMemGetInfo(&freeVramBytes, &totalVramBytes) != cudaSuccess) return 0;
    return (totalVramBytes - freeVramBytes) / kBytesPerMebibyte;
}

int GpuProfiler::pressureLevel() const
{
    const std::size_t freeVramMiB = liveFreeMb();
    if (freeVramMiB > config_.pressureWarningFreeMiB)   return 0;
    if (freeVramMiB >= config_.pressureCriticalFreeMiB)  return 1;
    return 2;
}

