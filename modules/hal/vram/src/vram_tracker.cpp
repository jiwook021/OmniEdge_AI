#include "vram/vram_tracker.hpp"

#include "vram/vram_thresholds.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>


// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Requires: mutex_ must be held by the caller.
VramTracker::ModuleRecord& VramTracker::findOrThrow(std::string_view moduleName)
{
    auto recordIterator = records_.find(moduleName);
    if (recordIterator == records_.end()) {
        throw std::runtime_error(
            std::format("[VramTracker] Module '{}' was not registered. "
                        "Call registerModuleBudget() during daemon initialization.",
                        moduleName));
    }
    return recordIterator->second;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void VramTracker::registerModuleBudget(std::string_view moduleName,
                                  std::size_t      budgetMb,
                                  int              evictPriority)
{
    if (moduleName.empty()) {
        throw std::invalid_argument("[VramTracker] moduleName must not be empty");
    }

    std::lock_guard lock(mutex_);
    auto [it, inserted] = records_.emplace(
        std::string{moduleName},
        ModuleRecord{
            .moduleName    = std::string{moduleName},
            .budgetMb      = budgetMb,
            .evictPriority = evictPriority,
            .isLoaded      = false,
            .isIdle        = true,
        });

    if (!inserted) {
        OE_LOG_ERROR("vram_register_duplicate: module={}", moduleName);
        throw std::runtime_error(
            std::format("[VramTracker] Module '{}' is already registered.",
                        moduleName));
    }

    OE_LOG_INFO("vram_module_registered: module={}, budget_mb={}, evict_priority={}",
              moduleName, budgetMb, evictPriority);
}

void VramTracker::markModuleLoaded(std::string_view moduleName)
{
    std::lock_guard lock(mutex_);
    ModuleRecord& moduleRecord = findOrThrow(moduleName);
    moduleRecord.isLoaded = true;

    OE_LOG_INFO("vram_module_loaded: module={}, budget_mib={}", moduleName, moduleRecord.budgetMb);
}

void VramTracker::markModuleUnloaded(std::string_view moduleName)
{
    {
        std::lock_guard lock(mutex_);
        ModuleRecord& moduleRecord = findOrThrow(moduleName);

        const std::size_t freedBudgetMiB = moduleRecord.budgetMb;
        moduleRecord.isLoaded = false;
        moduleRecord.isIdle   = true;

        OE_LOG_INFO("vram_module_unloaded: module={}, freed_mib={}", moduleName, freedBudgetMiB);
    }

    vramFreedCondition_.notify_all();
}

void VramTracker::setIdle(std::string_view moduleName, bool isNowIdle)
{
    std::lock_guard lock(mutex_);
    ModuleRecord& moduleRecord = findOrThrow(moduleName);
    const bool stateChanged = (moduleRecord.isIdle != isNowIdle);
    moduleRecord.isIdle = isNowIdle;
    if (stateChanged) {
        OE_LOG_DEBUG("vram_module_idle_changed: module={}, idle={}", moduleName, isNowIdle);
    }
}

void VramTracker::updateEvictPriority(std::string_view moduleName, int newPriority)
{
    std::lock_guard lock(mutex_);
    ModuleRecord& moduleRecord = findOrThrow(moduleName);
    const int oldPriority = moduleRecord.evictPriority;
    moduleRecord.evictPriority = newPriority;

    if (oldPriority != newPriority) {
        OE_LOG_INFO("vram_priority_updated: module={}, old={}, new={}",
                   moduleName, oldPriority, newPriority);
    }
}

std::size_t VramTracker::totalLoadedMb() const
{
    std::lock_guard lock(mutex_);
    std::size_t totalLoadedMiB = 0;
    for (const auto& [_, moduleRecord] : records_) {
        if (moduleRecord.isLoaded) {
            totalLoadedMiB += moduleRecord.budgetMb;
        }
    }
    OE_LOG_DEBUG("vram_total_loaded_mib: total={}", totalLoadedMiB);
    return totalLoadedMiB;
}

std::vector<VramTracker::ModuleRecord> VramTracker::evictableCandidates() const
{
    std::lock_guard lock(mutex_);

    std::vector<ModuleRecord> evictionCandidates;
    evictionCandidates.reserve(records_.size());

    for (const auto& [_, moduleRecord] : records_) {
        if (!moduleRecord.isLoaded || !moduleRecord.isIdle) continue;
        // LLM is never evicted — it is the core pipeline component
        if (moduleRecord.evictPriority >= kNeverEvictPriority) continue;

        evictionCandidates.push_back(moduleRecord);
    }

    // Sort: lowest priority first; within same priority, largest VRAM budget
    // first (frees the most memory per eviction).  Consistent with
    // PriorityScheduler::evictionCandidates() ordering.
    std::ranges::sort(evictionCandidates,
        [](const ModuleRecord& lhs, const ModuleRecord& rhs) {
            if (lhs.evictPriority != rhs.evictPriority)
                return lhs.evictPriority < rhs.evictPriority;
            return lhs.budgetMb > rhs.budgetMb;
        });

    OE_LOG_DEBUG("vram_evictable_candidates: count={}", evictionCandidates.size());
    for ([[maybe_unused]] const auto& candidate : evictionCandidates) {
        OE_LOG_DEBUG("  candidate: module={}, budget_mib={}, priority={}",
                   candidate.moduleName, candidate.budgetMb, candidate.evictPriority);
    }

    return evictionCandidates;
}

std::vector<VramTracker::ModuleRecord> VramTracker::snapshot() const
{
    std::lock_guard lock(mutex_);

    std::vector<ModuleRecord> result;
    result.reserve(records_.size());
    for (const auto& [_, moduleRecord] : records_) {
        result.push_back(moduleRecord);
    }
    return result;
}

bool VramTracker::waitUntilLoadedBelow(
    std::size_t               targetLoadedMiB,
    std::chrono::milliseconds timeout) const
{
    std::unique_lock lock(mutex_);

    OE_LOG_DEBUG("vram_wait_until_loaded_below: target_mib={}, timeout_ms={}",
               targetLoadedMiB, timeout.count());

    return vramFreedCondition_.wait_for(lock, timeout, [&] {
        std::size_t currentTotalLoadedMiB = 0;
        for (const auto& [_, moduleRecord] : records_) {
            if (moduleRecord.isLoaded) {
                currentTotalLoadedMiB += moduleRecord.budgetMb;
            }
        }
        return currentTotalLoadedMiB < targetLoadedMiB;
    });
}

std::size_t VramTracker::moduleBudgetMiB(std::string_view moduleName) const
{
    std::lock_guard lock(mutex_);
    auto recordIterator = records_.find(moduleName);
    if (recordIterator == records_.end()) return 0;
    return recordIterator->second.budgetMb;
}

