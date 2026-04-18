#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/string_hash.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — Per-Module VRAM Accounting for Daemon Eviction
//
// The daemon uses VramTracker to maintain the expected logical VRAM
// allocation across all running modules. Module name keys must match the
// INI config keys from conventions.md:
//   "conversation_model", "stt", "tts",
//   "face_recognition", "background_blur"
//
// Eviction invariants:
//   - Only loaded AND idle modules are eviction candidates.
//   - conversation_model (kNeverEvictPriority = 5) is NEVER evicted.
//   - evictableCandidates() returns modules sorted ascending by priority.
// ---------------------------------------------------------------------------


class VramTracker {
public:
    struct ModuleRecord {
        std::string moduleName;
        std::size_t budgetMb      = 0;     ///< Expected VRAM in MiB on the active tier
        int         evictPriority = 0;     ///< Lower = evicted first
        bool        isLoaded      = false;
        bool        isIdle        = true;
    };

    VramTracker() = default;

    VramTracker(const VramTracker&)            = delete;
    VramTracker& operator=(const VramTracker&) = delete;

    void registerModuleBudget(std::string_view moduleName,
                        std::size_t      budgetMb,
                        int              evictPriority);

    void markModuleLoaded(std::string_view moduleName);
    void markModuleUnloaded(std::string_view moduleName);
    void setIdle(std::string_view moduleName, bool isNowIdle);

    /// Update a module's eviction priority at runtime.
    /// Called during interaction profile switches to keep VramTracker
    /// eviction ordering consistent with PriorityScheduler.
    void updateEvictPriority(std::string_view moduleName, int newPriority);

    [[nodiscard]] std::size_t totalLoadedMb() const;
    [[nodiscard]] std::vector<ModuleRecord> evictableCandidates() const;
    [[nodiscard]] std::vector<ModuleRecord> snapshot() const;

    [[nodiscard]] bool waitUntilLoadedBelow(
        std::size_t              targetLoadedMiB,
        std::chrono::milliseconds timeout) const;

    [[nodiscard]] std::size_t moduleBudgetMiB(std::string_view moduleName) const;

private:
    mutable std::mutex                             mutex_;
    mutable std::condition_variable                vramFreedCondition_;
    std::unordered_map<std::string, ModuleRecord, StringHash, std::equal_to<>>  records_;

    ModuleRecord& findOrThrow(std::string_view moduleName);
};

