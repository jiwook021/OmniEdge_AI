#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tl/expected.hpp>
#include "vram/vram_gate.hpp"
#include "orchestrator/module_launcher.hpp"


struct ModeDefinition {
	std::string                     name;
	std::unordered_set<std::string> required;  ///< Module YAML keys that must run
};

/// Manages UI mode switching with VRAM-aware module transitions.
///
/// Uses VramGate to ensure VRAM is available before spawning new modules.
/// The gate handles the evict → wait → verify cycle so mode switches
/// never race against CUDA driver memory reclamation.
///
/// NOT thread-safe — all calls from orchestrator poll loop.
class ModeOrchestrator {
public:
	struct Config {
		std::vector<ModeDefinition>                  modes;
		std::unordered_map<std::string, std::size_t> vramBudgetMiB;
	};

	explicit ModeOrchestrator(Config config);
	ModeOrchestrator(const ModeOrchestrator&)            = delete;
	ModeOrchestrator& operator=(const ModeOrchestrator&) = delete;

	/// Switch mode: evict unneeded → wait for VRAM → spawn required.
	/// Uses VramGate to verify VRAM availability before each spawn.
	/// Returns names of modules that failed to start.
	[[nodiscard]] tl::expected<std::unordered_set<std::string>, std::string>
	handleModeSwitch(const std::string&            targetModeName,
	                 std::vector<ModuleDescriptor>& modules,
	                 ModuleLauncher&                launcher,
	                 VramGate&                      vramGate,
	                 zmq::context_t&                zmqContext);

	[[nodiscard]] const std::string& currentMode() const noexcept;
	void setInitialMode(const std::string& modeName);

private:
	Config      config_;
	std::string currentModeName_{"conversation"};

	[[nodiscard]] const ModeDefinition* findModeDefinition(const std::string& modeName) const;
};

