#include "vram/vram_gate.hpp"

#include <algorithm>
#include <format>
#include <thread>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


VramGate::VramGate(VramTracker&        vramTracker,
                   GpuProfiler&        gpuProfiler,
                   std::function<bool(std::string_view)> evictCallback,
                   Config              config)
    : vramTracker_(vramTracker)
    , gpuProfiler_(gpuProfiler)
    , evictCallback_(std::move(evictCallback))
    , config_(config)
{
}

void VramGate::notifyModuleEvicted(std::string_view moduleName)
{
    OE_LOG_INFO("vram_gate_module_evicted: module={}", moduleName);
    vramTracker_.markModuleUnloaded(moduleName);
}

bool VramGate::hasEnoughFreeVram(std::size_t requiredVramMiB) const
{
    const std::size_t currentFreeMiB = gpuProfiler_.liveFreeMb();
    const std::size_t totalNeededMiB = requiredVramMiB + config_.safetyMarginMiB;
    return currentFreeMiB >= totalNeededMiB;
}

tl::expected<std::size_t, std::string>
VramGate::acquireVramForModule(std::string_view moduleName,
                               std::size_t      requiredVramMiB)
{
    const std::size_t totalNeededMiB = requiredVramMiB + config_.safetyMarginMiB;

    // Budget cap check: total loaded + requested must not exceed hard cap
    if (config_.maxTotalVramMiB > 0) {
        const std::size_t currentLoadedMiB = vramTracker_.totalLoadedMb();
        if ((currentLoadedMiB + requiredVramMiB) > config_.maxTotalVramMiB) {
            return tl::make_unexpected(std::format(
                "VRAM budget cap exceeded for {}: loaded={} MiB + requested={} MiB "
                "= {} MiB > cap={} MiB",
                moduleName, currentLoadedMiB, requiredVramMiB,
                currentLoadedMiB + requiredVramMiB, config_.maxTotalVramMiB));
        }
    }

    OE_LOG_INFO("vram_acquire_start: module={}, required_mib={}, safety_margin_mib={}, total_needed_mib={}",
              moduleName, requiredVramMiB, config_.safetyMarginMiB, totalNeededMiB);

    // -- Fast path: enough VRAM already free --
    const std::size_t currentFreeMiB = gpuProfiler_.liveFreeMb();
    if (currentFreeMiB >= totalNeededMiB) {
        OE_LOG_INFO("vram_acquire_fast_path: module={}, free_mib={}, needed_mib={} — no eviction needed",
                  moduleName, currentFreeMiB, totalNeededMiB);
        vramTracker_.markModuleLoaded(moduleName);
        return currentFreeMiB;
    }

    OE_LOG_WARN("vram_insufficient: module={}, free_mib={}, needed_mib={} — starting eviction",
              moduleName, currentFreeMiB, totalNeededMiB);

    // -- Slow path: evict modules until enough VRAM is freed --
    std::vector<VramTracker::ModuleRecord> evictionCandidates =
        vramTracker_.evictableCandidates();

    if (evictionCandidates.empty()) {
        return tl::make_unexpected(std::format(
            "Cannot acquire {} MiB for {}: no evictable modules available "
            "(free={} MiB, needed={} MiB)",
            requiredVramMiB, moduleName, currentFreeMiB, totalNeededMiB));
    }

    std::size_t totalEvictedBudgetMiB = 0;

    for (const auto& candidate : evictionCandidates) {
        OE_LOG_WARN("vram_evicting: module={}, budget_mib={}, priority={} — to make room for {}",
                  candidate.moduleName, candidate.budgetMb,
                  candidate.evictPriority, moduleName);

        const bool evictionSucceeded = evictCallback_(candidate.moduleName);

        if (!evictionSucceeded) {
            OE_LOG_ERROR("vram_eviction_failed: module={} — could not stop process",
                       candidate.moduleName);
            continue;
        }

        vramTracker_.markModuleUnloaded(candidate.moduleName);
        totalEvictedBudgetMiB += candidate.budgetMb;

        OE_LOG_INFO("vram_eviction_complete: module={}, freed_budget_mib={}, total_evicted_mib={}",
                  candidate.moduleName, candidate.budgetMb, totalEvictedBudgetMiB);

        auto waitResult = waitForVramAvailable(totalNeededMiB, config_.maxWaitAfterEviction);

        if (waitResult) {
            OE_LOG_INFO("vram_acquire_success: module={}, confirmed_free_mib={}, "
                      "evicted_count={}, evicted_budget_mib={}",
                      moduleName, *waitResult,
                      static_cast<int>(&candidate - evictionCandidates.data()) + 1,
                      totalEvictedBudgetMiB);
            vramTracker_.markModuleLoaded(moduleName);
            return *waitResult;
        }

        OE_LOG_WARN("vram_still_insufficient_after_eviction: module={}, "
                  "free_mib={}, needed_mib={} — trying next candidate",
                  candidate.moduleName, gpuProfiler_.liveFreeMb(), totalNeededMiB);
    }

    // All candidates exhausted — final check
    auto finalWait = waitForVramAvailable(totalNeededMiB, config_.maxWaitAfterEviction);
    if (finalWait) {
        OE_LOG_INFO("vram_acquire_success_after_full_eviction: module={}, free_mib={}",
                  moduleName, *finalWait);
        vramTracker_.markModuleLoaded(moduleName);
        return *finalWait;
    }

    const std::size_t finalFreeMiB = gpuProfiler_.liveFreeMb();
    return tl::make_unexpected(std::format(
        "VRAM acquisition timed out for {}: evicted {} MiB across all candidates "
        "but only {} MiB free (need {} MiB).",
        moduleName, totalEvictedBudgetMiB, finalFreeMiB, totalNeededMiB));
}

tl::expected<std::size_t, std::string>
VramGate::waitForVramAvailable(std::size_t               targetFreeMiB,
                               std::chrono::milliseconds  timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto currentPollInterval = config_.initialPollInterval;

    OE_LOG_DEBUG("vram_wait_start: target_free_mib={}, timeout_ms={}",
               targetFreeMiB, timeout.count());

    int pollIteration = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const std::size_t currentFreeMiB = gpuProfiler_.liveFreeMb();

        if (currentFreeMiB >= targetFreeMiB) {
            OE_LOG_DEBUG("vram_wait_success: free_mib={}, target_mib={}, polls={}",
                       currentFreeMiB, targetFreeMiB, pollIteration);
            return currentFreeMiB;
        }

        pollIteration++;
        OE_LOG_DEBUG("vram_wait_polling: free_mib={}, target_mib={}, poll={}, interval_ms={}",
                   currentFreeMiB, targetFreeMiB, pollIteration, currentPollInterval.count());

        std::this_thread::sleep_for(currentPollInterval);
        currentPollInterval = std::min(
            currentPollInterval * 2, config_.maxPollInterval);
    }

    const std::size_t finalFreeMiB = gpuProfiler_.liveFreeMb();
    return tl::make_unexpected(std::format(
        "Timed out waiting for {} MiB free VRAM (current: {} MiB, timeout: {} ms, polls: {})",
        targetFreeMiB, finalFreeMiB, timeout.count(), pollIteration));
}

