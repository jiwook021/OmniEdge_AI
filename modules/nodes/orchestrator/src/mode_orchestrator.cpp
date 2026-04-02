#include "orchestrator/mode_orchestrator.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <format>
#include <thread>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


ModeOrchestrator::ModeOrchestrator(Config config)
	: config_(std::move(config))
{
}

const ModeDefinition* ModeOrchestrator::findModeDefinition(const std::string& modeName) const
{
	for (const auto& modeDefinition : config_.modes) {
		if (modeDefinition.name == modeName) return &modeDefinition;
	}
	return nullptr;
}

tl::expected<std::unordered_set<std::string>, std::string>
ModeOrchestrator::handleModeSwitch(const std::string&             targetModeName,
                                   std::vector<ModuleDescriptor>& modules,
                                   ModuleLauncher&                launcher,
                                   VramGate&                      vramGate,
                                   zmq::context_t&                zmqContext)
{
	const auto* targetModeDefinition = findModeDefinition(targetModeName);
	if (!targetModeDefinition) {
		return tl::make_unexpected(std::format("unknown mode: {}", targetModeName));
	}
	if (targetModeName == currentModeName_) {
		return std::unordered_set<std::string>{};
	}

	OE_LOG_INFO("mode_switch_start: current={}, target={}", currentModeName_, targetModeName);

	// ── Step 1: Evict modules not needed in the target mode ─────────────
	// Sort by eviction priority (lowest = evicted first) to free VRAM
	// from the least important modules first.
	std::vector<ModuleDescriptor*> modulesToEvict;
	for (auto& moduleDescriptor : modules) {
		if (moduleDescriptor.isRunning() &&
		    targetModeDefinition->required.count(moduleDescriptor.name) == 0) {
			modulesToEvict.push_back(&moduleDescriptor);
		}
	}
	std::ranges::sort(modulesToEvict,
		[](auto* lhs, auto* rhs) { return lhs->evictPriority < rhs->evictPriority; });

	for (auto* moduleToEvict : modulesToEvict) {
		OE_LOG_WARN("mode_switch_evicting: module={}, priority={}",
		          moduleToEvict->name, moduleToEvict->evictPriority);

		// Save PID before stopModule() resets it to -1
		const pid_t savedPid = moduleToEvict->pid;
		launcher.stopModule(*moduleToEvict);

		// ── Eviction Verification Gate ──────────────────────────────
		// On WSL2, SIGKILL can rarely fail to terminate a process stuck
		// in uninterruptible (D) state. If the old process survives,
		// two modules share the same VRAM budget → CUDA OOM.
		// Verify death before reclaiming VRAM.
		if (savedPid > 0) {
			bool processVerifiedDead = false;
			for (int attempt = 0;
			     attempt < kEvictionVerifyMaxAttempts;
			     ++attempt) {
				if (::kill(savedPid, 0) == -1 && errno == ESRCH) {
					processVerifiedDead = true;
					break;
				}
				std::this_thread::sleep_for(
					kEvictionVerifyPollInterval);
			}

			if (!processVerifiedDead) {
				OE_LOG_ERROR(
					"eviction_zombie: {} (pid={}) refused to die after "
					"SIGKILL — VRAM budget may be over-committed",
					moduleToEvict->name, savedPid);
				// Restore pid so the system knows this process is still
				// running. Do NOT reclaim its VRAM budget.
				moduleToEvict->pid = savedPid;
				continue;
			}
		}

		// Process verified dead — safe to update VRAM accounting
		vramGate.notifyModuleEvicted(moduleToEvict->name);
	}

	// ── Step 2: Identify modules that need to be spawned ────────────────
	std::vector<ModuleDescriptor*> modulesToSpawn;
	for (auto& moduleDescriptor : modules) {
		if (!moduleDescriptor.isRunning() &&
		    targetModeDefinition->required.count(moduleDescriptor.name) > 0) {
			modulesToSpawn.push_back(&moduleDescriptor);
		}
	}

	// ── Step 3: Acquire VRAM and spawn each module ──────────────────────
	// Use VramGate to ensure VRAM is available before each spawn.
	// This handles the case where the CUDA driver needs time to reclaim
	// memory from evicted processes.
	std::unordered_set<std::string> failedModuleNames;

	for (auto* moduleToSpawn : modulesToSpawn) {
		// Look up the module's VRAM budget
		std::size_t moduleVramBudgetMiB = 0;
		if (auto budgetIterator = config_.vramBudgetMiB.find(moduleToSpawn->name);
		    budgetIterator != config_.vramBudgetMiB.end()) {
			moduleVramBudgetMiB = budgetIterator->second;
		}

		// Acquire VRAM (evicts further if needed, waits for driver to reclaim)
		if (moduleVramBudgetMiB > 0) {
			auto vramAcquireResult = vramGate.acquireVramForModule(
				moduleToSpawn->name, moduleVramBudgetMiB);

			if (!vramAcquireResult) {
				OE_LOG_ERROR("mode_switch_vram_acquire_failed: module={}, error={}",
				           moduleToSpawn->name, vramAcquireResult.error());
				failedModuleNames.insert(moduleToSpawn->name);
				continue;
			}
		}

		// Spawn the module process
		auto spawnResult = launcher.spawnModule(*moduleToSpawn);
		if (!spawnResult) {
			OE_LOG_ERROR("mode_switch_spawn_failed: module={}, error={}",
			           moduleToSpawn->name, spawnResult.error());
			failedModuleNames.insert(moduleToSpawn->name);
		}
	}

	// ── Step 4: Wait for spawned modules to report readiness ────────────
	// Collect already-spawned module names and wait for their module_ready
	// via the launcher's readiness-wait logic. We do NOT call launchAll()
	// because Step 3 already spawned the processes — calling launchAll()
	// would double-spawn them (P0 bug fix).
	if (!modulesToSpawn.empty()) {
		auto modulesNotReady = launcher.waitForReady(modulesToSpawn, zmqContext);
		for (const auto& notReadyName : modulesNotReady) {
			failedModuleNames.insert(notReadyName);
		}
	}

	currentModeName_ = targetModeName;
	OE_LOG_INFO("mode_switch_complete: mode={}, evicted={}, spawned={}, failed={}",
	          targetModeName, modulesToEvict.size(), modulesToSpawn.size(),
	          failedModuleNames.size());
	return failedModuleNames;
}

