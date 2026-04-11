#include "gpu/gpu_tier.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <format>
#include <stdexcept>

#include <cuda_runtime.h>


VramInfo probeGpu(int deviceId)
{
    OE_LOG_INFO("gpu_probe_start: device_id={}", deviceId);

    cudaDeviceProp deviceProperties{};
    cudaError_t err = cudaGetDeviceProperties(&deviceProperties, deviceId);
    if (err != cudaSuccess) {
        OE_LOG_ERROR("gpu_probe_failed: device_id={}, error={}", deviceId, cudaGetErrorString(err));
        throw std::runtime_error(
            std::format("[probeGpu] cudaGetDeviceProperties(device={}) failed: {}",
                        deviceId, cudaGetErrorString(err)));
    }
    constexpr int kKhzPerMhz = 1000;
    int clockKhz = 0;
    (void)cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, deviceId);
    [[maybe_unused]] const int clockMhz = clockKhz / kKhzPerMhz;
    OE_LOG_DEBUG("gpu_device_props: name={}, sm={}.{}, multiprocessors={}, clock_mhz={}",
               deviceProperties.name, deviceProperties.major, deviceProperties.minor,
               deviceProperties.multiProcessorCount, clockMhz);

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::format("[probeGpu] cudaSetDevice({}) failed: {}",
                        deviceId, cudaGetErrorString(err)));
    }

    std::size_t freeBytes  = 0;
    std::size_t totalBytes = 0;
    err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) {
        OE_LOG_ERROR("gpu_meminfo_failed: error={}", cudaGetErrorString(err));
        throw std::runtime_error(
            std::format("[probeGpu] cudaMemGetInfo failed: {}",
                        cudaGetErrorString(err)));
    }
    OE_LOG_INFO("gpu_memory_info: total_mb={}, free_mb={}, used_mb={}",
              totalBytes / kBytesPerMebibyte,
              freeBytes / kBytesPerMebibyte,
              (totalBytes - freeBytes) / kBytesPerMebibyte);

    return VramInfo{
        .totalBytes = totalBytes,
        .freeBytes  = freeBytes,
        .computeCapabilityMajor    = deviceProperties.major,
        .computeCapabilityMinor    = deviceProperties.minor,
        .gpuName    = deviceProperties.name,
    };
}

GpuTier selectTier(const VramInfo& info, std::size_t headroomMb,
                    std::size_t ultraThresholdMiB,
                    std::size_t standardThresholdMiB,
                    std::size_t balancedThresholdMiB)
{
    const std::size_t totalMiB  = info.totalMb();
    const std::size_t usableMiB = (totalMiB > headroomMb) ? (totalMiB - headroomMb) : 0;

    GpuTier tier;
    if (usableMiB >= ultraThresholdMiB)         tier = GpuTier::kUltra;
    else if (usableMiB >= standardThresholdMiB)  tier = GpuTier::kStandard;
    else if (usableMiB >= balancedThresholdMiB)  tier = GpuTier::kBalanced;
    else tier = GpuTier::kMinimal;

    OE_LOG_INFO("gpu_tier_selected: gpu={}, total_mb={}, usable_mb={}, headroom_mb={}, tier={}",
              info.gpuName, totalMiB, usableMiB, headroomMb, tierName(tier));

    return tier;
}

std::string_view tierName(GpuTier tier) noexcept
{
    switch (tier) {
        case GpuTier::kMinimal:  return "minimal";
        case GpuTier::kBalanced: return "balanced";
        case GpuTier::kStandard: return "standard";
        case GpuTier::kUltra:    return "ultra";
    }
    return "minimal";
}

GpuTier parseTier(std::string_view name)
{
    OE_LOG_DEBUG("parse_tier: input={}", name);
    if (name == "minimal")  return GpuTier::kMinimal;
    if (name == "balanced") return GpuTier::kBalanced;
    if (name == "standard") return GpuTier::kStandard;
    if (name == "ultra")    return GpuTier::kUltra;
    OE_LOG_ERROR("parse_tier_failed: unknown_tier={}", name);
    throw std::invalid_argument(
        std::format("[parseTier] Unknown GPU tier name: '{}'. "
                    "Valid values: minimal, balanced, standard, ultra", name));
}

