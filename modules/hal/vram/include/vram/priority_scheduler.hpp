#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/string_hash.hpp"
#include "vram/vram_thresholds.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — PriorityScheduler
//
// Priority-based GPU allocation scheduler for module lifecycle and VRAM
// eviction.  
//
// Eviction walks from lowest priority upward. Within the same priority
// level, the module with the largest VRAM budget is evicted first (frees
// the most memory per eviction). Higher-priority tasks preempt
// lower-priority ones.
//
// Priorities are dynamic: the daemon shifts them via setPriority() or
// applyProfile() when the user's interaction pattern changes (e.g.,
// enabling STT/TTS → kConversation profile bumps voice modules to
// priority 4 and lowers visual modules to priority 1).
// ---------------------------------------------------------------------------


struct SchedulerEntry {
	std::string moduleName;
	std::size_t vramBudgetMiB{0};
	int         priority{0};
	bool        isIdle{true};
	bool        isLoaded{false};
};


class PriorityScheduler {
public:
	PriorityScheduler() = default;

	/// Register a module with an initial priority level.
	void registerModuleBudget(std::string_view moduleName,
	                    int priority,
	                    std::size_t vramBudgetMiB);

	/// Mark a module as loaded.
	void markModuleLoaded(std::string_view moduleName);

	/// Mark a module as unloaded.
	void markModuleUnloaded(std::string_view moduleName);

	/// Mark module as idle or busy.
	void setIdle(std::string_view moduleName, bool idle);

	/// Change a single module's priority dynamically.
	void setPriority(std::string_view moduleName, int newPriority);

	/// Apply an interaction profile — sets priorities for all registered
	/// modules that appear in the given priority map (from IniConfig).
	/// Returns the list of (moduleName, newPriority) pairs that changed.
	[[nodiscard]] std::vector<std::pair<std::string, int>>
	applyProfile(InteractionProfile profile,
	             const std::unordered_map<std::string, int>& priorities);

	/// Get the current interaction profile.
	[[nodiscard]] InteractionProfile currentProfile() const;

	/// Get eviction candidates: lowest priority first, then largest VRAM
	/// budget within the same priority. Only loaded + idle modules are
	/// candidates. Never returns modules at priority >= kNeverEvictPriority.
	[[nodiscard]] std::vector<SchedulerEntry> evictionCandidates() const;

	/// Find the single best eviction target (lowest priority, largest budget).
	[[nodiscard]] std::optional<SchedulerEntry> bestEvictionCandidate() const;

	/// Check if a higher-priority module can preempt a lower-priority one.
	/// Returns the name of the module to preempt, or nullopt.
	[[nodiscard]] std::optional<std::string> findPreemptionTarget(
		int requesterPriority, std::size_t requiredVramMiB) const;

	/// Get a module's current priority. Returns -1 if not registered.
	[[nodiscard]] int modulePriority(std::string_view moduleName) const;

	/// Snapshot for debugging/logging.
	[[nodiscard]] std::vector<SchedulerEntry> snapshot() const;

private:
	std::unordered_map<std::string, SchedulerEntry, StringHash, std::equal_to<>> modules_;
	InteractionProfile currentProfile_{InteractionProfile::kConversation};
	mutable std::mutex mutex_;
};

