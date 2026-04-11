#include "vram/priority_scheduler.hpp"

#include <algorithm>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


void PriorityScheduler::registerModuleBudget(std::string_view moduleName,
                                        int priority,
                                        std::size_t vramBudgetMiB)
{
	std::lock_guard lock(mutex_);

	// Overwrite any previous registration
	modules_[std::string{moduleName}] = SchedulerEntry{
		std::string{moduleName}, vramBudgetMiB, priority,
		/*isIdle=*/true, /*isLoaded=*/false};
	SPDLOG_INFO("Scheduler: registered module '{}' priority={} vram={}MiB",
	            moduleName, priority, vramBudgetMiB);
}

void PriorityScheduler::markModuleLoaded(std::string_view moduleName)
{
	std::lock_guard lock(mutex_);
	auto it = modules_.find(moduleName);
	if (it == modules_.end()) return;
	it->second.isLoaded = true;
}

void PriorityScheduler::markModuleUnloaded(std::string_view moduleName)
{
	std::lock_guard lock(mutex_);
	auto it = modules_.find(moduleName);
	if (it == modules_.end()) return;
	it->second.isLoaded = false;
}

void PriorityScheduler::setIdle(std::string_view moduleName, bool idle)
{
	std::lock_guard lock(mutex_);
	auto it = modules_.find(moduleName);
	if (it == modules_.end()) return;
	it->second.isIdle = idle;
}

void PriorityScheduler::setPriority(std::string_view moduleName, int newPriority)
{
	std::lock_guard lock(mutex_);
	auto it = modules_.find(moduleName);
	if (it == modules_.end()) return;
	it->second.priority = newPriority;
}

std::vector<std::pair<std::string, int>>
PriorityScheduler::applyProfile(InteractionProfile profile,
                                const std::unordered_map<std::string, int>& priorities)
{
	std::lock_guard lock(mutex_);

	std::vector<std::pair<std::string, int>> changed;

	for (auto& [name, entry] : modules_) {
		auto profileIt = priorities.find(name);
		if (profileIt == priorities.end()) continue;

		const int newPriority = profileIt->second;
		if (entry.priority != newPriority) {
			SPDLOG_INFO("Scheduler: profile change '{}' priority {} -> {}",
			            name, entry.priority, newPriority);
			changed.emplace_back(name, newPriority);
			entry.priority = newPriority;
		}
	}

	currentProfile_ = profile;
	return changed;
}

InteractionProfile PriorityScheduler::currentProfile() const
{
	std::lock_guard lock(mutex_);
	return currentProfile_;
}

std::vector<SchedulerEntry> PriorityScheduler::evictionCandidates() const
{
	std::lock_guard lock(mutex_);
	std::vector<SchedulerEntry> result;

	for (const auto& [name, entry] : modules_) {
		if (entry.priority >= kNeverEvictPriority) continue;
		if (!entry.isLoaded || !entry.isIdle) continue;
		result.push_back(entry);
	}

	// Sort: lowest priority first; within same priority, largest VRAM first
	std::ranges::sort(result, [](const SchedulerEntry& a, const SchedulerEntry& b) {
		if (a.priority != b.priority) return a.priority < b.priority;
		return a.vramBudgetMiB > b.vramBudgetMiB;
	});

	return result;
}

std::optional<SchedulerEntry> PriorityScheduler::bestEvictionCandidate() const
{
	// Linear scan O(n) instead of sort O(n log n) — we only need the minimum.
	std::lock_guard lock(mutex_);
	std::optional<SchedulerEntry> best;

	for (const auto& [name, entry] : modules_) {
		if (entry.priority >= kNeverEvictPriority) continue;
		if (!entry.isLoaded || !entry.isIdle) continue;

		if (!best.has_value() ||
		    entry.priority < best->priority ||
		    (entry.priority == best->priority &&
		     entry.vramBudgetMiB > best->vramBudgetMiB)) {
			best = entry;
		}
	}
	if (best.has_value()) {
		SPDLOG_DEBUG("Scheduler: best eviction candidate '{}' priority={} vram={}MiB",
		             best->moduleName, best->priority, best->vramBudgetMiB);
	}
	return best;
}

std::optional<std::string> PriorityScheduler::findPreemptionTarget(
	int requesterPriority, std::size_t requiredVramMiB) const
{
	std::lock_guard lock(mutex_);

	// Collect candidates at strictly lower priority with enough VRAM
	std::optional<std::string> best;
	int bestPriority = requesterPriority;  // must be strictly lower
	std::size_t bestBudget = 0;

	for (const auto& [name, entry] : modules_) {
		if (entry.priority >= requesterPriority) continue;
		if (entry.priority >= kNeverEvictPriority) continue;
		if (!entry.isLoaded || !entry.isIdle) continue;
		if (entry.vramBudgetMiB < requiredVramMiB) continue;

		// Prefer lowest priority; within same priority, largest budget
		if (!best.has_value() ||
		    entry.priority < bestPriority ||
		    (entry.priority == bestPriority && entry.vramBudgetMiB > bestBudget)) {
			best = entry.moduleName;
			bestPriority = entry.priority;
			bestBudget = entry.vramBudgetMiB;
		}
	}

	if (best.has_value()) {
		SPDLOG_INFO("Scheduler: preemption target '{}' (pri={}, {}MiB) for requester pri={}",
		            *best, bestPriority, bestBudget, requesterPriority);
	} else {
		SPDLOG_WARN("Scheduler: no preemption target found for pri={} needing {}MiB",
		            requesterPriority, requiredVramMiB);
	}
	return best;
}

int PriorityScheduler::modulePriority(std::string_view moduleName) const
{
	std::lock_guard lock(mutex_);
	auto it = modules_.find(moduleName);
	if (it == modules_.end()) return -1;
	return it->second.priority;
}

std::vector<SchedulerEntry> PriorityScheduler::snapshot() const
{
	std::lock_guard lock(mutex_);
	std::vector<SchedulerEntry> result;
	result.reserve(modules_.size());

	for (const auto& [name, entry] : modules_) {
		result.push_back(entry);
	}
	return result;
}

