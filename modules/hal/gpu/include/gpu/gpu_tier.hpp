#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "vram/vram_thresholds.hpp"
#include "common/constants/memory_constants.hpp"


struct VramInfo {
    std::size_t totalBytes = 0;
    std::size_t freeBytes  = 0;
    int         computeCapabilityMajor    = 0;
    int         computeCapabilityMinor    = 0;
    std::string gpuName;

    [[nodiscard]] std::size_t totalMb() const noexcept
    {
        return totalBytes / kBytesPerMebibyte;
    }

    [[nodiscard]] std::size_t freeMb() const noexcept
    {
        return freeBytes / kBytesPerMebibyte;
    }

};

enum class GpuTier : uint8_t {
    kMinimal  = 0,
    kBalanced = 1,
    kStandard = 2,
    kUltra    = 3,
};

[[nodiscard]] VramInfo probeGpu(int deviceId = 0);

[[nodiscard]] GpuTier selectTier(const VramInfo& info,
                                  std::size_t     headroomMb = kHeadroomMiB,
                                  std::size_t     ultraThresholdMiB = kUltraTierThresholdMiB,
                                  std::size_t     standardThresholdMiB = kStandardTierThresholdMiB,
                                  std::size_t     balancedThresholdMiB = kBalancedTierThresholdMiB);

[[nodiscard]] std::string_view tierName(GpuTier tier) noexcept;

[[nodiscard]] GpuTier parseTier(std::string_view name);

