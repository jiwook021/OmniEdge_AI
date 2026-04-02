#include "gpu/gpu_diagnostics.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "common/oe_tracy.hpp"

#include <nvml.h>


// ── GPU Utilization ─────────────────────────────────────────────────────

GpuUtilization queryGpuUtilization(unsigned int deviceIndex)
{
    static bool nvmlInitialized = false;
    if (!nvmlInitialized) {
        nvmlReturn_t ret = nvmlInit_v2();
        if (ret != NVML_SUCCESS) {
            SPDLOG_ERROR("NVML init failed (ret={})", static_cast<int>(ret));
            return {};
        }
        nvmlInitialized = true;
    }

    nvmlDevice_t device{};
    nvmlReturn_t ret = nvmlDeviceGetHandleByIndex_v2(deviceIndex, &device);
    if (ret != NVML_SUCCESS) {
        SPDLOG_WARN("NVML: cannot get device handle for index {} (ret={})",
                     deviceIndex, static_cast<int>(ret));
        return {};
    }

    nvmlUtilization_t util{};
    ret = nvmlDeviceGetUtilizationRates(device, &util);
    if (ret != NVML_SUCCESS) {
        SPDLOG_WARN("NVML: utilization query failed for device {} (ret={})",
                     deviceIndex, static_cast<int>(ret));
        return {};
    }

    return {
        .gpuPercent = util.gpu,
        .memPercent = util.memory,
        .valid      = true,
    };
}

// ── Host RSS ────────────────────────────────────────────────────────────

std::size_t hostRssKiB()
{
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        SPDLOG_WARN("Cannot open /proc/self/status for RSS query");
        return 0;
    }

    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // Format: "VmRSS:    123456 kB"
            std::size_t rss = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %zu", &rss) == 1) {
                return rss;
            }
        }
    }
    return 0;
}

