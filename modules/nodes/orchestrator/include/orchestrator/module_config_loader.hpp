#pragma once

// ---------------------------------------------------------------------------
// Pure YAML → data transformation for the daemon's module registry.
//
// Extracts the YAML parsing half of OmniEdgeDaemon::loadModuleConfigFromYaml
// so the daemon's loader becomes a thin orchestration layer that composes
// INI loading, SHM registration, and config overrides with pure-data YAML
// parse results.
//
// No daemon state, no I/O beyond YAML::Node traversal — unit-testable in
// isolation.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "orchestrator/mode_definition.hpp"
#include "orchestrator/module_launcher.hpp"


/// Per-model flags parsed from the YAML `conversation_models` section.
/// Consumed by handleSelectConversationModel() to drive companion-module
/// decisions (TTS spawn, vision support, auto-fallback chain).
struct ConversationModelFlags {
	bool        needsTts{false};       ///< !native_tts — spawn Kokoro sidecar
	bool        supportsVision{false}; ///< native_vision — accepts video frames
	std::string resolvedModelDir;      ///< absolute path (models_root + model_dir)
	std::string fallback;              ///< next-smaller model, empty = terminal
	std::size_t vramBudgetMiB{0};      ///< from YAML vram_budget_mib
};


/// Aggregated result of parsing a module-registry YAML file.
struct ModuleConfig {
	std::vector<ModuleDescriptor>                            modules;
	std::unordered_map<std::string, ConversationModelFlags>  conversationModelFlags;
};


/// Parse the YAML `modules:` and `conversation_models:` sections into value
/// types. Pure function — no I/O, no daemon state.
///
/// @param rootNode         Parsed YAML root (from YAML::LoadFile).
/// @param launchOrder      Module names to parse in order — drives modules.push_back.
///                         Names absent from YAML are skipped silently.
/// @param evictPriorities  Name → eviction priority (lower = evicted first),
///                         sourced from the active interaction profile at parse time.
/// @param modelsRoot       Root directory for resolving conversation_models.model_dir.
[[nodiscard]] ModuleConfig parseYamlModuleConfig(
    const YAML::Node&                            rootNode,
    const std::vector<std::string>&              launchOrder,
    const std::unordered_map<std::string, int>&  evictPriorities,
    const std::string&                           modelsRoot);


/// Five-Mode Architecture catalog — the five mutually exclusive GPU-inference
/// modes (Conversation, SAM2 Segmentation, Vision Model, Security, Beauty),
/// each listing its required modules. Always-on modules (video_ingest,
/// audio_ingest, background_blur, websocket_bridge) appear in every mode so
/// the orchestrator never evicts them during a switch.
[[nodiscard]] std::vector<ModeDefinition> buildDefaultModeCatalog();
