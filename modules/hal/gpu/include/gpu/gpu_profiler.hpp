#pragma once

#include <string>

#include "gpu/gpu_tier.hpp"
#include "vram/vram_thresholds.hpp"


class GpuProfiler {
public:
    struct Config {
        int         deviceId{0};
        std::string overrideProfile;
        std::size_t headroomMb{kHeadroomMiB};
        std::size_t pressureWarningFreeMiB{kPressureWarningMiB};
        std::size_t pressureCriticalFreeMiB{kPressureCriticalMiB};
        // Tier thresholds (from INI [vram_tiers] or compile-time defaults)
        std::size_t ultraTierThresholdMiB{kUltraTierThresholdMiB};
        std::size_t standardTierThresholdMiB{kStandardTierThresholdMiB};
        std::size_t balancedTierThresholdMiB{kBalancedTierThresholdMiB};
    };

    explicit GpuProfiler(Config config);
    GpuProfiler(const GpuProfiler&)            = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    void probe();

    [[nodiscard]] GpuTier tier() const noexcept { return selectedTier_; }
    [[nodiscard]] const VramInfo& vramInfo() const noexcept { return vramInfo_; }
    [[nodiscard]] std::size_t liveFreeMb() const;
    [[nodiscard]] std::size_t liveUsedMb() const;
    [[nodiscard]] int pressureLevel() const;

private:
    Config   config_;
    VramInfo vramInfo_;
    GpuTier  selectedTier_{GpuTier::kMinimal};
    bool     hasProbed_{false};
};

