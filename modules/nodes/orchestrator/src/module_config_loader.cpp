#include "orchestrator/module_config_loader.hpp"

#include <format>
#include <stdexcept>
#include <unordered_set>

#include "common/model_path.hpp"
#include "common/oe_logger.hpp"


namespace {

/// Parse one `modules:` entry into a ModuleDescriptor.
/// Throws std::runtime_error if the zmq_pub port is malformed.
[[nodiscard]] ModuleDescriptor parseModuleNode(
    const std::string&                           name,
    const YAML::Node&                            modNode,
    const std::unordered_map<std::string, int>&  evictPriorities)
{
	ModuleDescriptor desc;
	desc.name           = name;
	desc.binaryPath     = modNode["binary"].as<std::string>("");
	desc.maxRestarts    = modNode["max_restarts"].as<int>(5);
	desc.zmqPubEndpoint = modNode["zmq_pub"].as<std::string>("");

	if (!desc.zmqPubEndpoint.empty()) {
		auto underscorePos = desc.zmqPubEndpoint.rfind('_');
		if (underscorePos != std::string::npos) {
			try {
				desc.zmqPubPort = std::stoi(desc.zmqPubEndpoint.substr(underscorePos + 1));
			} catch (const std::exception&) {
				throw std::runtime_error(std::format(
				    "bad zmq_pub port for '{}': {}", name, desc.zmqPubEndpoint));
			}
		}
	}

	if (auto argsNode = modNode["args"]; argsNode && argsNode.IsSequence()) {
		for (const auto& a : argsNode) {
			desc.args.push_back(a.as<std::string>());
		}
	}

	if (auto it = evictPriorities.find(name); it != evictPriorities.end()) {
		desc.evictPriority = it->second;
	}

	return desc;
}

}  // namespace


ModuleConfig parseYamlModuleConfig(
    const YAML::Node&                            rootNode,
    const std::vector<std::string>&              launchOrder,
    const std::unordered_map<std::string, int>&  evictPriorities,
    const std::string&                           modelsRoot)
{
	ModuleConfig cfg;

	auto modulesNode = rootNode["modules"];
	if (modulesNode) {
		// ── Launch-order modules ────────────────────────────────────────
		for (const auto& name : launchOrder) {
			auto modNode = modulesNode[name];
			if (!modNode) continue;
			cfg.modules.push_back(parseModuleNode(name, modNode, evictPriorities));
		}

		// ── On-demand modules: present in YAML but absent from launch order ──
		std::unordered_set<std::string> launchSet(launchOrder.begin(), launchOrder.end());
		for (auto it = modulesNode.begin(); it != modulesNode.end(); ++it) {
			const std::string name = it->first.as<std::string>();
			if (launchSet.contains(name)) continue;

			ModuleDescriptor desc = parseModuleNode(name, it->second, evictPriorities);
			desc.onDemand = true;
			cfg.modules.push_back(std::move(desc));
			OE_LOG_DEBUG("on_demand_module_parsed: name={}", name);
		}
	}

	// ── Conversation models ─────────────────────────────────────────────
	if (auto convModels = rootNode["conversation_models"]; convModels && convModels.IsMap()) {
		const auto resolveModelPath = makeModelPathResolver(modelsRoot);

		for (auto it = convModels.begin(); it != convModels.end(); ++it) {
			const std::string modelName = it->first.as<std::string>();
			cfg.conversationModelFlags[modelName] = ConversationModelFlags{
			    .needsTts         = !it->second["native_tts"].as<bool>(true),
			    .supportsVision   = it->second["native_vision"].as<bool>(false),
			    .resolvedModelDir = resolveModelPath(it->second["model_dir"].as<std::string>("")),
			    .fallback         = it->second["fallback"].as<std::string>(""),
			    .vramBudgetMiB    = it->second["vram_budget_mib"].as<std::size_t>(0),
			};
			OE_LOG_DEBUG("conversation_model_flags: model={}, needs_tts={}, vision={}, dir={}",
			    modelName,
			    cfg.conversationModelFlags[modelName].needsTts,
			    cfg.conversationModelFlags[modelName].supportsVision,
			    cfg.conversationModelFlags[modelName].resolvedModelDir);
		}
	}

	return cfg;
}


std::vector<ModeDefinition> buildDefaultModeCatalog()
{
	// Mode 1 — Conversation (default): Gemma-4 handles STT+LLM natively,
	//          pairs with Kokoro TTS sidecar. Optional audio_denoise toggle.
	// Mode 2 — SAM2 Segmentation: interactive segmentation (no audio).
	// Mode 3 — Vision Model: VLM (vision+audio+text) in conversation_model slot.
	// Mode 4 — Security: object detection + event logging.
	// Mode 5 — Beauty: face-mesh + landmark-driven effects.
	// NOTE: `websocket_bridge` is NOT part of the daemon-managed mode graph.
	// It runs as its own binary (omniedge_ws_bridge), launched alongside the
	// daemon by profile entrypoints (`run_conversation.sh`,
	// `run_security_mode.sh`, `run_beautymode.sh`) and Docker entrypoint.sh.
	// The orchestrator doesn't spawn or track ws_bridge, so listing it here
	// produces "unknown module 'websocket_bridge'" errors on every mode switch.
	// Keep the list to modules the daemon actually owns.
	return {
	    {"conversation",      {"video_ingest", "audio_ingest",
	                           "conversation_model"}},
	    {"sam2_segmentation", {"video_ingest", "audio_ingest", "sam2"}},
	    {"vision_model",      {"video_ingest", "audio_ingest",
	                           "conversation_model"}},
	    {"security",          {"video_ingest", "audio_ingest",
	                           "security_camera", "security_vlm"}},
	    {"beauty",            {"video_ingest", "audio_ingest", "beauty"}},
	};
}
