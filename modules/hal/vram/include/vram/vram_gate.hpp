#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>
#include "gpu/gpu_profiler.hpp"
#include "vram/vram_tracker.hpp"
#include "common/runtime_defaults.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — VRAM Acquisition Gate
//
// Provides wait-until-VRAM-is-available semantics for the daemon.
// Orchestrates evict -> wait -> verify cycle so that callers never
// attempt to spawn a module into insufficient VRAM.
// ---------------------------------------------------------------------------


class VramGate {
public:
    struct Config {
        std::chrono::milliseconds maxWaitAfterEviction{kVramGateMaxWaitMs};
        std::chrono::milliseconds initialPollInterval{kVramGatePollInitialMs};
        std::chrono::milliseconds maxPollInterval{kVramGatePollMaxMs};
        std::size_t safetyMarginMiB{kVramGateSafetyMarginMiB};
        std::size_t maxTotalVramMiB{0};  ///< 0 = no cap; >0 = hard VRAM budget cap
    };

    explicit VramGate(VramTracker&        vramTracker,
                      GpuProfiler&        gpuProfiler,
                      std::function<bool(std::string_view)> evictCallback,
                      Config              config);

    explicit VramGate(VramTracker&        vramTracker,
                      GpuProfiler&        gpuProfiler,
                      std::function<bool(std::string_view)> evictCallback)
        : VramGate(vramTracker, gpuProfiler, std::move(evictCallback), Config{}) {}

    VramGate(const VramGate&)            = delete;
    VramGate& operator=(const VramGate&) = delete;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    acquireVramForModule(std::string_view moduleName,
                         std::size_t      requiredVramMiB);

    /// Notify that a module was evicted outside of VramGate's own eviction
    /// path (e.g., by PipelineOrchestrator's switch-plan eviction phase).
    /// Forwards to VramTracker::markModuleUnloaded() so VRAM accounting stays
    /// consistent with the actual set of running processes.
    void notifyModuleEvicted(std::string_view moduleName);

    [[nodiscard]] bool hasEnoughFreeVram(std::size_t requiredVramMiB) const;

    [[nodiscard]] tl::expected<std::size_t, std::string>
    waitForVramAvailable(std::size_t                   targetFreeMiB,
                         std::chrono::milliseconds      timeout);

private:
    VramTracker&        vramTracker_;
    GpuProfiler&        gpuProfiler_;
    std::function<bool(std::string_view)> evictCallback_;
    Config              config_;
};

