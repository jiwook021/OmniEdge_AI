#include "orchestrator/omniedge_daemon.hpp"

#include <array>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

#include "common/file.hpp"
#include "common/model_path.hpp"
#include "common/oe_tracy.hpp"
#include "gpu/gpu_diagnostics.hpp"
#include "zmq/zmq_endpoint.hpp"


// == Construction / Destruction ================================================

OmniEdgeDaemon::OmniEdgeDaemon(Config config)
	: config_(std::move(config))
{
}

OmniEdgeDaemon::~OmniEdgeDaemon()
{
	stop();
	for (auto& mod : modules_) {
		if (mod.isRunning()) {
			OE_LOG_INFO("stopping_module: name={}, pid={}", mod.name, mod.pid);
			launcher_.stopModule(mod);
		}
	}
}

// == initialize ===============================================================

void OmniEdgeDaemon::initialize()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule("omniedge_orchestrator");

	loadModuleConfigFromYaml();

	launcher_.setBackoffConfig(
		static_cast<unsigned>(config_.restartBaseBackoffMs),
		static_cast<unsigned>(config_.restartMaxBackoffMs));

	// GPU probe + tier selection — thresholds from INI [vram_tiers] / [vram_limits]
	{
		const auto& vt = iniConfig_.vramTiers();
		gpuProfiler_ = std::make_unique<GpuProfiler>(GpuProfiler::Config{
			.deviceId                 = config_.gpuDeviceId,
			.overrideProfile          = config_.gpuOverrideProfile,
			.headroomMb               = vt.headroomMiB,
			.pressureWarningFreeMiB   = vt.pressureWarningFreeMiB,
			.pressureCriticalFreeMiB  = vt.pressureCriticalFreeMiB,
			.ultraTierThresholdMiB    = vt.ultraThresholdMiB,
			.standardTierThresholdMiB = vt.standardThresholdMiB,
			.balancedTierThresholdMiB = vt.balancedThresholdMiB,
		});
	}
	gpuProfiler_->probe();

	// Register all GPU-consuming modules with the VRAM tracker and
	// PriorityScheduler using the default kConversation interaction profile.
	// Priorities are dynamic — shifted by applyInteractionProfile() when
	// the active GPU mode changes.
	//
	// Four-mode architecture: conversation_model is registered with the
	// default Qwen2.5-Omni-7B budget.  On model switch, the daemon
	// updates the budget via vramTracker_.updateBudget().
	// Register GPU modules — budgets from INI [vram_budgets], priorities from
	// INI [profile_conversation] (the default boot profile).
	{
		const auto& defaultPriorities =
			iniConfig_.profilePriorities(InteractionProfile::kConversation);

		struct ModuleVramDef {
			std::string name;
			std::size_t budgetMiB;
		};
		const std::array kGpuModules = {
			// Always-on modules (resident across all modes)
			ModuleVramDef{"background_blur",    iniConfig_.vramBudgetMiB("background_blur",    kBgBlurMiB)},
			ModuleVramDef{"face_recognition",   iniConfig_.vramBudgetMiB("face_recognition",   kFaceRecogMiB)},
			// Conversation mode — unified model (default: Qwen2.5-Omni-7B)
			ModuleVramDef{"conversation_model",  iniConfig_.vramBudgetMiB("qwen_omni_7b",       kQwenOmni7bMiB)},
			ModuleVramDef{"tts",                 iniConfig_.vramBudgetMiB("tts",                kTtsMiB)},
			ModuleVramDef{"audio_denoise",       iniConfig_.vramBudgetMiB("audio_denoise",      kDtlnMiB)},
			// Super Resolution mode
			ModuleVramDef{"super_resolution",    iniConfig_.vramBudgetMiB("super_resolution",   kBasicVsrppMiB)},
			// Image Transform mode
			ModuleVramDef{"image_transform",     iniConfig_.vramBudgetMiB("image_transform",    kStableDiffusionMiB)},
			// SAM2 Segmentation mode
			ModuleVramDef{"sam2",                iniConfig_.vramBudgetMiB("sam2",               kSam2MiB)},
			// Legacy separate modules (backward compatibility)
			ModuleVramDef{"llm",                 iniConfig_.vramBudgetMiB("llm",                kLlmMiB)},
			ModuleVramDef{"stt",                 iniConfig_.vramBudgetMiB("stt",                kSttMiB)},
			ModuleVramDef{"video_denoise",       iniConfig_.vramBudgetMiB("video_denoise",      kBasicVsrppMiB)},
		};

		for (const auto& [name, budget] : kGpuModules) {
			const int priority = defaultPriorities.count(name)
				? defaultPriorities.at(name) : 0;
			vramTracker_.registerModuleBudget(name, budget, priority);
			priorityScheduler_.registerModuleBudget(name, priority, budget);
		}
	}

	// VramGate — provides wait-for-VRAM-available semantics.
	// The eviction callback stops a module process (blocking until waitpid)
	// so that the CUDA driver can reclaim its VRAM.
	vramGate_ = std::make_unique<VramGate>(
		vramTracker_, *gpuProfiler_,
		/*evictCallback=*/ [this](std::string_view moduleNameToEvict) -> bool {
			for (auto& moduleDescriptor : modules_) {
				if (moduleDescriptor.name == moduleNameToEvict && moduleDescriptor.isRunning()) {
					OE_LOG_WARN("vram_gate_evict_callback: stopping module={}, pid={}",
					          moduleDescriptor.name, moduleDescriptor.pid);
					launcher_.stopModule(moduleDescriptor);
					const bool stopped = !moduleDescriptor.isRunning();
					if (stopped) {
						// Sync PriorityScheduler — VramGate updates VramTracker
						// directly, but PriorityScheduler only learns about
						// load/unload via the EventBus.
						const std::size_t budgetMiB =
							vramTracker_.moduleBudgetMiB(moduleDescriptor.name);
						eventBus_.publish(VramChanged{
							.moduleName = moduleDescriptor.name,
							.budgetMiB  = budgetMiB,
							.loaded     = false,
						});
					}
					return stopped;
				}
			}
			OE_LOG_WARN("vram_gate_evict_callback: module={} not found or not running",
			          moduleNameToEvict);
			return false;
		},
		VramGate::Config{
			.maxTotalVramMiB = iniConfig_.vramLimits().maxTotalVramMb,
		});

	// ── EventBus subscriptions ──────────────────────────────────────────
	// PriorityScheduler reacts to VRAM load/unload events without the
	// caller needing to know about it — decoupled via the event bus.
	eventBus_.subscribe<VramChanged>([this](const VramChanged& e) {
		if (e.loaded) {
			priorityScheduler_.markModuleLoaded(e.moduleName);
		} else {
			priorityScheduler_.markModuleUnloaded(e.moduleName);
		}
		OE_LOG_DEBUG("event_bus_vram_changed: module={}, loaded={}, budget_mib={}",
		           e.moduleName, e.loaded, e.budgetMiB);
	});

	eventBus_.subscribe<InteractionProfileChanged>(
		[](const InteractionProfileChanged& e) {
		OE_LOG_INFO("event_bus_profile_changed: from={}, to={}, modules_changed={}",
		          e.fromProfile, e.toProfile, e.modulesChanged);
	});

	eventBus_.subscribe<ModuleCrashed>([this](const ModuleCrashed& e) {
		OE_LOG_INFO("event_bus_module_crashed: module={}, pid={}, signal={}",
		          e.moduleName, e.pid, e.signal);
	});

	eventBus_.subscribe<ModeChanged>([this](const ModeChanged& e) {
		OE_LOG_INFO("event_bus_mode_changed: mode={}, success={}", e.modeName, e.success);
	});

	// Prompt assembler — token budgets from INI [prompt] section or compile-time defaults
	{
		const auto& pc = iniConfig_.prompt();
		promptAssembler_ = std::make_unique<PromptAssembler>(PromptAssembler::Config{
			.maxContextTokens    = pc.maxContextTokens,
			.systemPromptTokens  = pc.systemPromptTokens,
			.dynamicContextTokens = pc.dynamicContextTokens,
			.maxUserTurnTokens   = pc.maxUserTurnTokens,
			.generationHeadroom  = pc.generationHeadroom,
			.systemPrompt        = "You are OmniEdge AI, a helpful voice assistant. "
			                       "Respond concisely and naturally.",
		});
	}

	// ── Mode orchestrator — Four mutually exclusive GPU inference modes ─────
	//
	// Always-on modules (video_ingest, audio_ingest, background_blur,
	// websocket_bridge) are listed in EVERY mode so the ModeOrchestrator
	// never evicts them during a mode switch.
	//
	// Mode 1 — Conversation (default): unified conversation model handles
	//   STT+LLM+TTS in a single process (Qwen2.5-Omni-7B/3B) or with a
	//   companion TTS (Gemma 4 E4B).  Optional audio_denoise toggle.
	//   Kills: super_resolution, image_transform.
	//
	// Mode 2 — Super Resolution: BasicVSR++ temporal video enhancement.
	//   Kills: conversation_model, tts, image_transform.
	//
	// Mode 3 — Image Transform: Stable Diffusion + ControlNet/IP-Adapter.
	//   Kills: conversation_model, tts, super_resolution.
	//
	// Mode 4 — SAM2 Segmentation: interactive segmentation (no audio).
	//   Kills: conversation_model, tts, super_resolution, image_transform.
	//
	modeOrchestrator_ = std::make_unique<ModeOrchestrator>(ModeOrchestrator::Config{
		.modes = {
			{"conversation",        {"video_ingest", "audio_ingest", "background_blur",
			                         "websocket_bridge", "conversation_model"}},
			{"super_resolution",    {"video_ingest", "audio_ingest", "background_blur",
			                         "websocket_bridge", "super_resolution"}},
			{"image_transform",     {"video_ingest", "audio_ingest", "background_blur",
			                         "websocket_bridge", "image_transform"}},
			{"sam2_segmentation",   {"video_ingest", "background_blur",
			                         "websocket_bridge", "sam2"}},
		},
		.vramBudgetMiB = {
			// Budgets from INI [vram_budgets] with compile-time fallbacks
			{"background_blur",    iniConfig_.vramBudgetMiB("background_blur",    kBgBlurMiB)},
			{"face_recognition",   iniConfig_.vramBudgetMiB("face_recognition",   kFaceRecogMiB)},
			{"conversation_model", iniConfig_.vramBudgetMiB("qwen_omni_7b",       kQwenOmni7bMiB)},
			{"tts",                iniConfig_.vramBudgetMiB("tts",                kTtsMiB)},
			{"audio_denoise",      iniConfig_.vramBudgetMiB("audio_denoise",      kDtlnMiB)},
			{"super_resolution",   iniConfig_.vramBudgetMiB("super_resolution",   kBasicVsrppMiB)},
			{"image_transform",    iniConfig_.vramBudgetMiB("image_transform",    kStableDiffusionMiB)},
			{"sam2",               iniConfig_.vramBudgetMiB("sam2",               kSam2MiB)},
			{"llm",                iniConfig_.vramBudgetMiB("llm",                kLlmMiB)},
			{"stt",                iniConfig_.vramBudgetMiB("stt",                kSttMiB)},
			{"video_denoise",      iniConfig_.vramBudgetMiB("video_denoise",      kBasicVsrppMiB)},
		},
	});

	// Session persistence — timing from INI [session] section or compile-time defaults
	{
		const auto& sc = iniConfig_.session();
		sessionPersistence_ = std::make_unique<SessionPersistence>(SessionPersistence::Config{
			.filePath             = config_.sessionFilePath,
			.periodicSaveInterval = std::chrono::seconds{sc.periodicSaveIntervalS},
			.maxStaleness         = std::chrono::seconds{sc.maxStalenessS},
		});
	}

	if (auto restored = sessionPersistence_->load()) {
		restoreSessionState(*restored);
		OE_LOG_INFO("session_restored");
	}

	// MessageRouter — owns ZMQ context, PUB socket, all SUB sockets.
	messageRouter_ = std::make_unique<MessageRouter>(MessageRouter::Config{
		.moduleName  = "omniedge_orchestrator",
		.pubPort     = config_.pubPort,
	});

	// Generic message handler used for most topic subscriptions.
	// Also runs periodic tasks (deferred events, timeouts, watchdog,
	// session save) after dispatching each message — the router's poll
	// timeout ensures this fires frequently even during quiet periods.
	auto handler = [this](const nlohmann::json& msg) {
		auto typeIt = msg.find("type");
		if (typeIt != msg.end()) {
			handleMessage(typeIt->get<std::string>(), msg);
		}
		tickPeriodicTasks();
	};

	// Subscribe to upstream data topics.
	messageRouter_->subscribe(kWsBridge,  "ui_command",      false, handler);
	messageRouter_->subscribe(kFaceRecog, "identity",        false, handler);
	messageRouter_->subscribe(kAudioIngest, "vad_status",    false, handler);
	messageRouter_->subscribe(kLlm,       "llm_response",   false, handler);
	// Four-mode architecture subscriptions
	messageRouter_->subscribe(kConversationModel, "transcription",  false, handler);
	messageRouter_->subscribe(kConversationModel, "llm_response",   false, handler);
	messageRouter_->subscribe(kConversationModel, "audio_output",   false, handler);
	messageRouter_->subscribe(kImageTransform,    "transform_result", false, handler);
	messageRouter_->subscribe(kSuperResolution,   "sr_frame",        false, handler);

	// Subscribe to module_ready and heartbeat from every module's PUB port.
	static constexpr int kModulePorts[] = {
		kVideoIngest,       // 5555
		kAudioIngest,       // 5556
		kLlm,              // 5561 (legacy standalone)
		kStt,              // 5563 (legacy standalone)
		kTts,              // 5565
		kFaceRecog,        // 5566
		kBgBlur,           // 5567
		kVideoDenoise,     // 5568 (legacy alias)
		kAudioDenoise,     // 5569
		kWsBridge,         // 5570
		kConversationModel,// 5572 (unified Qwen-Omni / Gemma)
		kSuperResolution,  // 5573 (BasicVSR++)
		kImageTransform,   // 5574 (Stable Diffusion)
	};
	for (int port : kModulePorts) {
		messageRouter_->subscribe(port, "module_ready", false, handler);
		messageRouter_->subscribe(port, "heartbeat",    false, handler);
	}

	// Transitions are defined at compile time in the Transitions struct —
	// no runtime registration needed.
	// Validate FSM coverage: flag dead states and missing recovery paths at boot.
	validateFsmCoverage();

	// Launch modules and wait for readiness.
	// GPU modules only launch at boot if their priority in the active
	// profile >= kAutoLaunchPriorityThreshold (3).  Low-priority GPU
	// modules (e.g., STT at priority 1 in kConversation) stay dormant until
	// the user explicitly enables them via toggle.
	// Non-GPU modules (video_ingest, audio_ingest, ws_bridge) always launch.
	if (!modules_.empty()) {
		for (auto& moduleDescriptor : modules_) {
			if (moduleDescriptor.onDemand) continue;  // skip on-demand modules

			const std::size_t moduleBudgetMiB =
				vramTracker_.moduleBudgetMiB(moduleDescriptor.name);

			// Skip low-priority GPU modules at boot — they launch on user toggle
			if (moduleBudgetMiB > 0) {
				const int modulePriority =
					priorityScheduler_.modulePriority(moduleDescriptor.name);
				if (modulePriority < kAutoLaunchPriorityThreshold) {
					OE_LOG_INFO("boot_skip_low_priority: module={}, priority={} "
					          "(threshold={}), will launch on user toggle",
					          moduleDescriptor.name, modulePriority,
					          kAutoLaunchPriorityThreshold);
					continue;
				}

				if (vramGate_) {
					auto vramResult = vramGate_->acquireVramForModule(
						moduleDescriptor.name, moduleBudgetMiB);
					if (!vramResult) {
						OE_LOG_ERROR("initial_launch_vram_insufficient: module={}, error={}",
						           moduleDescriptor.name, vramResult.error());
						publishModuleStatus(moduleDescriptor.name, "vram_insufficient");
						continue;
					}
				}
			}
		}

		auto modulesNotReady = launcher_.launchAll(modules_, messageRouter_->context(),
			std::chrono::seconds{config_.moduleReadyTimeoutS});
		for (const auto& moduleName : modulesNotReady) {
			OE_LOG_WARN("module_not_ready_at_startup: name={}", moduleName);
			publishModuleStatus(moduleName, "timeout");
		}

		// Seed heartbeat tracking for modules that became ready during launch.
		{
			auto now = std::chrono::steady_clock::now();
			for (const auto& mod : modules_) {
				if (mod.ready) {
					moduleHeartbeats_[mod.name] = now;
				}
			}
		}

		// Mark successfully launched modules as loaded in the VRAM tracker
		for (const auto& moduleDescriptor : modules_) {
			if (moduleDescriptor.isRunning() && moduleDescriptor.ready) {
				const std::size_t budgetMiB =
					vramTracker_.moduleBudgetMiB(moduleDescriptor.name);
				if (budgetMiB > 0) {
					eventBus_.publish(VramChanged{
						.moduleName = moduleDescriptor.name,
						.budgetMiB  = budgetMiB,
						.loaded     = true,
					});
					OE_LOG_INFO("initial_vram_accounting: module={}, budget_mib={}",
					          moduleDescriptor.name, budgetMiB);
				}
			}
		}

		// ── Auto-launch high-priority on-demand modules ────────────────
		// In the four-mode architecture, on-demand modules with priority >= 3
		// in the active profile are auto-launched at boot.  Under kConversation
		// the only priority-5 modules are conversation_model (in launch_order)
		// and always-on modules, so typically nothing auto-launches here.
		{
			for (auto& moduleDescriptor : modules_) {
				if (!moduleDescriptor.onDemand) continue;
				if (moduleDescriptor.isRunning()) continue;

				const int modulePriority =
					priorityScheduler_.modulePriority(moduleDescriptor.name);
				if (modulePriority < kAutoLaunchPriorityThreshold) continue;

				const std::size_t moduleBudgetMiB =
					vramTracker_.moduleBudgetMiB(moduleDescriptor.name);

				if (moduleBudgetMiB > 0 && vramGate_) {
					auto vramResult = vramGate_->acquireVramForModule(
						moduleDescriptor.name, moduleBudgetMiB);
					if (!vramResult) {
						OE_LOG_WARN("auto_launch_vram_insufficient: module={}, priority={}, error={}",
						           moduleDescriptor.name, modulePriority, vramResult.error());
						continue;
					}
				}

				auto spawnResult = launcher_.spawnModule(moduleDescriptor);
				if (spawnResult) {
					if (moduleBudgetMiB > 0) {
						eventBus_.publish(VramChanged{
							.moduleName = moduleDescriptor.name,
							.budgetMiB  = moduleBudgetMiB,
							.loaded     = true,
						});
					}
					moduleHeartbeats_[moduleDescriptor.name] = std::chrono::steady_clock::now();
					OE_LOG_INFO("auto_launch_on_demand: module={}, priority={}, budget_mib={}",
					           moduleDescriptor.name, modulePriority, moduleBudgetMiB);
				} else {
					// Rollback VRAM accounting — acquireVramForModule already
					// marked this module as loaded in VramTracker.
					if (moduleBudgetMiB > 0) {
						vramTracker_.markModuleUnloaded(moduleDescriptor.name);
						eventBus_.publish(VramChanged{
							.moduleName = moduleDescriptor.name,
							.budgetMiB  = moduleBudgetMiB,
							.loaded     = false,
						});
					}
					OE_LOG_WARN("auto_launch_failed: module={}, error={}",
					           moduleDescriptor.name, spawnResult.error());
				}
			}
		}
	}

	// ── Auto-spawn companion modules for the default conversation model ──
	// If the default model needs companion STT or TTS, spawn them now.
	{
		auto flagsIt = conversationModelFlags_.find(activeConversationModel_);
		if (flagsIt != conversationModelFlags_.end()) {
			if (flagsIt->second.needsTts) {
				auto ttsIt = moduleIndex_.find("tts");
				if (ttsIt != moduleIndex_.end()) {
					ModuleDescriptor& ttsMod = modules_[ttsIt->second];
					if (!ttsMod.isRunning()) {
						if (vramGate_) {
							(void)vramGate_->acquireVramForModule("tts", iniConfig_.vramBudgetMiB("tts", kTtsMiB));
						}
						auto ttsSpawn = launcher_.spawnModule(ttsMod);
						if (ttsSpawn) {
							vramTracker_.markModuleLoaded("tts");
							moduleHeartbeats_["tts"] = std::chrono::steady_clock::now();
							OE_LOG_INFO("companion_tts_auto_spawned: default model {} needs companion TTS",
							          activeConversationModel_);
						}
					}
				}
			}
			// STT is always native — handled by ConversationNode directly.
		}
	}

	const auto& vramLimits = iniConfig_.vramLimits();
	OE_LOG_INFO("vram_cap_configured: max_total_mb={}, warning_mb={}, critical_mb={}, "
	          "check_interval_ms={}, initial_loaded_mb={}",
	          vramLimits.maxTotalVramMb, vramLimits.warningThresholdMb,
	          vramLimits.criticalThresholdMb, vramLimits.vramCheckIntervalMs,
	          vramTracker_.totalLoadedMb());
	OE_LOG_INFO("daemon_initialized: tier={}, module_count={}",
	          tierName(gpuProfiler_->tier()), modules_.size());
	logSystemResources("daemon_init");
}

// == loadModuleConfigFromYaml =============================================

void OmniEdgeDaemon::loadModuleConfigFromYaml()
{
	if (config_.configFile.empty()) return;

	YAML::Node root = YAML::LoadFile(config_.configFile);

	if (auto daemon = root["daemon"]) {
		config_.watchdogPollMs = daemon["watchdog_poll_ms"].as<int>(config_.watchdogPollMs);

		if (auto sm = daemon["state_machine"]) {
			config_.vadSilenceThresholdMs = sm["vad_silence_threshold_ms"].as<int>(config_.vadSilenceThresholdMs);
			config_.bargeInEnabled        = sm["barge_in_enabled"].as<bool>(config_.bargeInEnabled);
			config_.llmGenerationTimeoutS = sm["llm_generation_timeout_s"].as<int>(config_.llmGenerationTimeoutS);
		}

		if (auto hb = daemon["heartbeat"]) {
			config_.heartbeatIvlMs     = hb["ivl_ms"].as<int>(config_.heartbeatIvlMs);
			config_.heartbeatTimeoutMs = hb["timeout_ms"].as<int>(config_.heartbeatTimeoutMs);
			config_.heartbeatTtlMs     = hb["ttl_ms"].as<int>(config_.heartbeatTtlMs);
		}

		config_.moduleReadyTimeoutS     = daemon["module_ready_timeout_s"].as<int>(config_.moduleReadyTimeoutS);
		config_.restartBaseBackoffMs    = daemon["restart_base_backoff_ms"].as<int>(config_.restartBaseBackoffMs);
		config_.restartMaxBackoffMs     = daemon["restart_max_backoff_ms"].as<int>(config_.restartMaxBackoffMs);

		if (auto subNodes = daemon["zmq_sub"]; subNodes && subNodes.IsSequence()) {
			config_.subEndpoints.clear();
			for (const auto& ep : subNodes) {
				config_.subEndpoints.push_back(ep.as<std::string>());
			}
		}
	}

	// Parse modules in launch order
	auto modulesNode = root["modules"];
	if (!modulesNode) return;

	auto launchOrder = root["launch_order"];
	std::vector<std::string> order;
	if (launchOrder && launchOrder.IsSequence()) {
		for (const auto& m : launchOrder) {
			order.push_back(m.as<std::string>());
		}
	}

	// Eviction priorities come from the active interaction profile (dynamic).
	// Use the default kConversation profile at parse time; applyInteractionProfile()
	// updates them at runtime when the user's interaction pattern changes.
	const auto& kEvictPriorities =
		iniConfig_.profilePriorities(InteractionProfile::kConversation);

	for (const auto& name : order) {
		auto modNode = modulesNode[name];
		if (!modNode) continue;

		ModuleDescriptor desc;
		desc.name          = name;
		desc.binaryPath    = modNode["binary"].as<std::string>("");
		desc.maxRestarts   = modNode["max_restarts"].as<int>(5);
		desc.zmqPubEndpoint = modNode["zmq_pub"].as<std::string>("");

		// Extract port from zmq_pub endpoint (ipc:///tmp/omniedge_<port>)
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

		if (auto it = kEvictPriorities.find(name); it != kEvictPriorities.end()) {
			desc.evictPriority = it->second;
		}

		modules_.push_back(std::move(desc));
	}

	// ── On-demand modules: parse descriptors not in launch_order ────────
	// These modules (e.g. video_denoise, audio_denoise) are defined in the
	// YAML modules section but NOT in launch_order.  They are spawned via
	// UI toggle (handleToggleDenoise) and need a descriptor for the
	// launcher to know the binary path, args, and ZMQ endpoint.
	{
		std::unordered_set<std::string> launchSet(order.begin(), order.end());
		for (auto it = modulesNode.begin(); it != modulesNode.end(); ++it) {
			const std::string name = it->first.as<std::string>();
			if (launchSet.contains(name)) continue;  // already parsed above

			auto modNode = it->second;
			ModuleDescriptor desc;
			desc.name          = name;
			desc.binaryPath    = modNode["binary"].as<std::string>("");
			desc.maxRestarts   = modNode["max_restarts"].as<int>(5);
			desc.zmqPubEndpoint = modNode["zmq_pub"].as<std::string>("");

			// Extract port from zmq_pub endpoint (ipc:///tmp/omniedge_<port>)
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

			if (auto evIt = kEvictPriorities.find(name); evIt != kEvictPriorities.end()) {
				desc.evictPriority = evIt->second;
			}

			desc.onDemand = true;  // flag so launchAll skips these
			modules_.push_back(std::move(desc));
			OE_LOG_DEBUG("on_demand_module_parsed: name={}", name);
		}
	}

	// ── SHM Registry: register producer segments per module ─────────────
	// Parse each module's top-level YAML section for SHM output keys so
	// the watchdog can shm_unlink them on crash (prevents stale /dev/shm/oe.*).
	{
		// Well-known ingest SHM names (hardcoded in source, not in YAML)
		shmRegistry_.registerSegment("video_ingest", "/oe.vid.ingest");
		shmRegistry_.registerSegment("audio_ingest", "/oe.aud.ingest");

		static const std::vector<std::pair<std::string, std::string>> kShmYamlKeys{
			{"tts",             "shm_output"},
			{"background_blur", "shm_output"},
			{"video_denoise",   "shm_output"},
			{"audio_denoise",   "shm_output"},
		};
		for (const auto& [modName, yamlKey] : kShmYamlKeys) {
			if (auto section = root[modName]; section) {
				if (auto shmNode = section[yamlKey]; shmNode) {
					shmRegistry_.registerSegment(modName, shmNode.as<std::string>());
				}
			}
		}
		OE_LOG_INFO("shm_registry: {} modules with registered SHM segments",
		            shmRegistry_.moduleCount());
	}

	// ── INI Config: load omniedge.ini and filter disabled modules ────────
	{
		std::string iniPath = config_.iniFilePath;
		if (iniPath.empty() && !config_.configFile.empty()) {
			// Default: omniedge.ini in the same directory as omniedge_config.yaml
			std::filesystem::path configDir =
				std::filesystem::path(config_.configFile).parent_path();
			iniPath = (configDir / "omniedge.ini").string();
		}

		if (iniConfig_.loadFromFile(iniPath)) {
			// Remove modules that are disabled in the INI file
			std::vector<std::string> disabledNames;
			auto it = modules_.begin();
			while (it != modules_.end()) {
				if (!iniConfig_.isModuleEnabled(it->name)) {
					OE_LOG_WARN("module_disabled_by_ini: name={} — skipping launch", it->name);
					disabledNames.push_back(it->name);
					it = modules_.erase(it);
				} else {
					++it;
				}
			}

			if (!disabledNames.empty()) {
				std::string disabled;
				for (const auto& n : disabledNames) {
					if (!disabled.empty()) disabled += ", ";
					disabled += n;
				}
				OE_LOG_INFO("ini_config_disabled_modules: [{}]", disabled);
			}

			// ── Apply INI overrides to daemon Config ─────────────────────
			const auto& dt = iniConfig_.daemonTiming();
			config_.watchdogPollMs        = dt.watchdogPollMs;
			config_.moduleReadyTimeoutS   = dt.moduleReadyTimeoutS;
			config_.llmGenerationTimeoutS = dt.llmGenerationTimeoutS;
			config_.vadSilenceThresholdMs = dt.vadSilenceThresholdMs;

			const auto& hb = iniConfig_.heartbeat();
			config_.heartbeatIvlMs     = hb.intervalMs;
			config_.heartbeatTimeoutMs = hb.timeoutMs;
			config_.heartbeatTtlMs     = hb.ttlMs;

			// Apply max restarts from INI to all module descriptors
			for (auto& mod : modules_) {
				mod.maxRestarts = dt.maxModuleRestarts;
			}

			OE_LOG_INFO("ini_overrides_applied: watchdog={}ms, ready_timeout={}s, "
			          "heartbeat_ivl={}ms, heartbeat_timeout={}ms, "
			          "llm_timeout={}s, vad_silence={}ms",
			          config_.watchdogPollMs, config_.moduleReadyTimeoutS,
			          config_.heartbeatIvlMs, config_.heartbeatTimeoutMs,
			          config_.llmGenerationTimeoutS, config_.vadSilenceThresholdMs);
		}
	}

	// ── Parse conversation model flags from YAML ───────────────────────
	// Store native_tts flag per model so handleSelectConversationModel
	// can spawn companion TTS module data-driven (not hardcoded).
	// STT is always native (handled by ConversationNode directly).
	// Also resolve engine_dir/model_dir to absolute paths for pre-spawn validation.
	if (auto convModels = root["conversation_models"]; convModels && convModels.IsMap()) {
		const std::string rawModelsRoot = root["models_root"].as<std::string>("");
		const auto resolveModelPath = makeModelPathResolver(rawModelsRoot);

		for (auto it = convModels.begin(); it != convModels.end(); ++it) {
			const std::string modelName = it->first.as<std::string>();
			const bool nativeTts = it->second["native_tts"].as<bool>(true);
			const std::string rawDir = it->second["engine_dir"].as<std::string>(
				it->second["model_dir"].as<std::string>(""));
			const std::string resolvedDir = resolveModelPath(rawDir);
			const bool nativeVision = it->second["native_vision"].as<bool>(false);
			conversationModelFlags_[modelName] = {
				.needsTts = !nativeTts,
				.supportsVision = nativeVision,
				.resolvedEngineDir = resolvedDir,
			};
			OE_LOG_DEBUG("conversation_model_flags: model={}, needs_tts={}, vision={}, dir={}",
			           modelName, !nativeTts, nativeVision, resolvedDir);
		}
	}

	// Build name→index map for O(1) module lookups in dispatchers
	moduleIndex_.clear();
	moduleIndex_.reserve(modules_.size());
	for (std::size_t i = 0; i < modules_.size(); ++i) {
		moduleIndex_[modules_[i].name] = i;
	}

	{
		std::string moduleList;
		for (const auto& m : modules_) {
			if (!moduleList.empty()) moduleList += ", ";
			moduleList += m.name;
			if (m.onDemand) moduleList += "(od)";
		}
		OE_LOG_INFO("config_parsed: modules=[{}]", moduleList);
	}
	OE_LOG_INFO("config_parsed: modules={}, watchdog_ms={}, barge_in={}",
	          modules_.size(), config_.watchdogPollMs, config_.bargeInEnabled);
}

// == Type-safe transition table (T1–T23) ======================================
//
// Each overload is one transition. Guards are inline conditionals.
// Missing (state, event) pairs fall through to the catch-all → std::nullopt
// and are logged at WARN level by StateMachine::dispatch().
// Adding a new state or event without handling it is a compile error when the
// catch-all is temporarily removed during development.
// CoverageProbe (below) mirrors this table for boot-time dead-state detection.

struct OmniEdgeDaemon::Transitions {
	OmniEdgeDaemon* d;

	// T1: IDLE → LISTENING (PTT press)
	std::optional<State> operator()(fsm::Idle, fsm::PttPress) const {
		d->fsmOnStartListening();
		return fsm::Listening{};
	}

	// T2: IDLE → PROCESSING (describe scene — LLMs handle vision natively)
	std::optional<State> operator()(fsm::Idle, fsm::DescribeScene) const {
		d->buildAndPublishLlmPrompt("Describe what you see.");
		d->llmTimeoutDeadline_ = std::chrono::steady_clock::now() +
		                         std::chrono::seconds{d->config_.llmGenerationTimeoutS};
		return fsm::Processing{};
	}

	// T3: IDLE → PROCESSING (text input)
	std::optional<State> operator()(fsm::Idle, fsm::TextInput) const {
		d->buildAndPublishLlmPrompt(d->lastTranscription_);
		d->llmTimeoutDeadline_ = std::chrono::steady_clock::now() +
		                         std::chrono::seconds{d->config_.llmGenerationTimeoutS};
		return fsm::Processing{};
	}

	// T4: IDLE → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::Idle, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("");
		return fsm::ErrorRecovery{};
	}

	// T5/T6: LISTENING → PROCESSING or IDLE on PTT_RELEASE (guard: transcription)
	std::optional<State> operator()(fsm::Listening, fsm::PttRelease) const {
		if (!d->lastTranscription_.empty()) {
			d->fsmOnStopListeningAndSubmitPrompt();
			return fsm::Processing{};
		}
		d->fsmOnReturnToIdle();
		return fsm::Idle{};
	}

	// T7: LISTENING → PROCESSING on VAD_SILENCE (guard: non-empty transcription)
	std::optional<State> operator()(fsm::Listening, fsm::VadSilence) const {
		if (d->lastTranscription_.empty()) return std::nullopt;
		d->fsmOnStopListeningAndSubmitPrompt();
		return fsm::Processing{};
	}

	// T8: LISTENING → IDLE (PTT cancel)
	std::optional<State> operator()(fsm::Listening, fsm::PttCancel) const {
		d->fsmOnReturnToIdle();
		return fsm::Idle{};
	}

	// T9: LISTENING → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::Listening, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("");
		return fsm::ErrorRecovery{};
	}

	// T10: PROCESSING → SPEAKING (first sentence ready)
	std::optional<State> operator()(fsm::Processing, fsm::LlmFirstSentence) const {
		d->publishFsmState(StateIndex::kSpeaking);
		return fsm::Speaking{};
	}

	// T11: PROCESSING → IDLE (LLM done, no TTS)
	std::optional<State> operator()(fsm::Processing, fsm::LlmComplete) const {
		d->fsmOnFinishTurn();
		return fsm::Idle{};
	}

	// T12: PROCESSING → IDLE (LLM timeout)
	std::optional<State> operator()(fsm::Processing, fsm::LlmTimeout) const {
		OE_LOG_ERROR("llm_timeout");
		d->publishError("LLM generation timed out — no response received");
		d->publishFsmState(StateIndex::kIdle);
		return fsm::Idle{};
	}

	// T13: PROCESSING → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::Processing, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("llm");
		return fsm::ErrorRecovery{};
	}

	// T17/T18: SPEAKING + PTT_PRESS → INTERRUPTED (barge-in) or stay SPEAKING
	std::optional<State> operator()(fsm::Speaking, fsm::PttPress) const {
		if (d->config_.bargeInEnabled) {
			d->fsmOnBargeIn();
			return fsm::Interrupted{};
		}
		return fsm::Speaking{};
	}

	// T19: SPEAKING → IDLE (TTS done)
	std::optional<State> operator()(fsm::Speaking, fsm::TtsComplete) const {
		d->fsmOnFinishTurn();
		return fsm::Idle{};
	}

	// T20: SPEAKING → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::Speaking, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("");
		return fsm::ErrorRecovery{};
	}

	// T21: INTERRUPTED → LISTENING (interrupt acknowledged)
	std::optional<State> operator()(fsm::Interrupted, fsm::InterruptDone) const {
		d->fsmOnStartListening();
		return fsm::Listening{};
	}

	// T22: ERROR_RECOVERY → IDLE (module recovered)
	std::optional<State> operator()(fsm::ErrorRecovery, fsm::ModuleRecovered) const {
		d->publishFsmState(StateIndex::kIdle);
		OE_LOG_INFO("module_recovered");
		return fsm::Idle{};
	}

	// T23: ERROR_RECOVERY → IDLE (recovery timeout)
	std::optional<State> operator()(fsm::ErrorRecovery, fsm::RecoveryTimeout) const {
		d->publishFsmState(StateIndex::kIdle);
		OE_LOG_WARN("recovery_timeout");
		return fsm::Idle{};
	}

	// T25: ERROR_RECOVERY + TEXT_INPUT → PROCESSING (don't block user input during recovery)
	std::optional<State> operator()(fsm::ErrorRecovery, fsm::TextInput) const {
		OE_LOG_INFO("text_input_during_recovery: forwarding to LLM");
		d->buildAndPublishLlmPrompt(d->lastTranscription_);
		d->llmTimeoutDeadline_ = std::chrono::steady_clock::now() +
		                         std::chrono::seconds{d->config_.llmGenerationTimeoutS};
		return fsm::Processing{};
	}

	// ── Four-mode architecture transitions ──────────────────────────────

	// T26: IDLE → IMAGE_TRANSFORM (user triggered style transfer)
	std::optional<State> operator()(fsm::Idle, fsm::ImageTransformRequest) const {
		d->publishFsmState(StateIndex::kImageTransform);
		return fsm::ImageTransform{};
	}

	// T27: IMAGE_TRANSFORM → IDLE (inference complete)
	std::optional<State> operator()(fsm::ImageTransform, fsm::ImageTransformComplete) const {
		d->fsmOnImageTransformComplete();
		return fsm::Idle{};
	}

	// T28: IMAGE_TRANSFORM → IDLE (timeout)
	std::optional<State> operator()(fsm::ImageTransform, fsm::ImageTransformTimeout) const {
		d->fsmOnImageTransformTimeout();
		return fsm::Idle{};
	}

	// T29: IMAGE_TRANSFORM → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::ImageTransform, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("image_transform");
		return fsm::ErrorRecovery{};
	}

	// T30: SUPER_RESOLUTION → IDLE (user left SR mode via mode switch)
	std::optional<State> operator()(fsm::SuperResolution, fsm::SuperResolutionStop) const {
		d->publishFsmState(StateIndex::kIdle);
		return fsm::Idle{};
	}

	// T31: SUPER_RESOLUTION → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::SuperResolution, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("super_resolution");
		return fsm::ErrorRecovery{};
	}

	// Catch-all — unhandled (state, event) pair → logged by dispatch(), returns nullopt
	template <typename S, typename E>
	std::optional<State> operator()(S, E) const { return std::nullopt; }
};

std::optional<StateIndex> OmniEdgeDaemon::applyFsmEvent(const AnyEvent& event)
{
	return stateMachine_.dispatch(Transitions{this}, event);
}

// == FSM coverage validation (boot-time) =====================================
//
// Side-effect-free probe that mirrors the Transitions overload set.
// Returns destination StateIndex without executing daemon actions.
// Keep in sync with Transitions above — a missing entry here produces a
// false-positive dead-state warning at boot, prompting the developer to update.

namespace {
struct CoverageProbe {
	// T1
	std::optional<StateIndex> operator()(fsm::Idle, fsm::PttPress) const           { return StateIndex::kListening; }
	// T2
	std::optional<StateIndex> operator()(fsm::Idle, fsm::DescribeScene) const      { return StateIndex::kProcessing; }
	// T3
	std::optional<StateIndex> operator()(fsm::Idle, fsm::TextInput) const          { return StateIndex::kProcessing; }
	// T4
	std::optional<StateIndex> operator()(fsm::Idle, fsm::ModuleCrash) const        { return StateIndex::kErrorRecovery; }
	// T5/T6
	std::optional<StateIndex> operator()(fsm::Listening, fsm::PttRelease) const    { return StateIndex::kProcessing; }
	// T7
	std::optional<StateIndex> operator()(fsm::Listening, fsm::VadSilence) const    { return StateIndex::kProcessing; }
	// T8
	std::optional<StateIndex> operator()(fsm::Listening, fsm::PttCancel) const     { return StateIndex::kIdle; }
	// T9
	std::optional<StateIndex> operator()(fsm::Listening, fsm::ModuleCrash) const   { return StateIndex::kErrorRecovery; }
	// T10
	std::optional<StateIndex> operator()(fsm::Processing, fsm::LlmFirstSentence) const { return StateIndex::kSpeaking; }
	// T11
	std::optional<StateIndex> operator()(fsm::Processing, fsm::LlmComplete) const      { return StateIndex::kIdle; }
	// T12
	std::optional<StateIndex> operator()(fsm::Processing, fsm::LlmTimeout) const       { return StateIndex::kIdle; }
	// T13
	std::optional<StateIndex> operator()(fsm::Processing, fsm::ModuleCrash) const      { return StateIndex::kErrorRecovery; }
	// T17/T18
	std::optional<StateIndex> operator()(fsm::Speaking, fsm::PttPress) const       { return StateIndex::kInterrupted; }
	// T19
	std::optional<StateIndex> operator()(fsm::Speaking, fsm::TtsComplete) const    { return StateIndex::kIdle; }
	// T20
	std::optional<StateIndex> operator()(fsm::Speaking, fsm::ModuleCrash) const    { return StateIndex::kErrorRecovery; }
	// T21
	std::optional<StateIndex> operator()(fsm::Interrupted, fsm::InterruptDone) const { return StateIndex::kListening; }
	// T22
	std::optional<StateIndex> operator()(fsm::ErrorRecovery, fsm::ModuleRecovered) const { return StateIndex::kIdle; }
	// T23
	std::optional<StateIndex> operator()(fsm::ErrorRecovery, fsm::RecoveryTimeout) const { return StateIndex::kIdle; }
	// T25
	std::optional<StateIndex> operator()(fsm::ErrorRecovery, fsm::TextInput) const       { return StateIndex::kProcessing; }
	// T26
	std::optional<StateIndex> operator()(fsm::Idle, fsm::ImageTransformRequest) const    { return StateIndex::kImageTransform; }
	// T27
	std::optional<StateIndex> operator()(fsm::ImageTransform, fsm::ImageTransformComplete) const { return StateIndex::kIdle; }
	// T28
	std::optional<StateIndex> operator()(fsm::ImageTransform, fsm::ImageTransformTimeout) const  { return StateIndex::kIdle; }
	// T29
	std::optional<StateIndex> operator()(fsm::ImageTransform, fsm::ModuleCrash) const    { return StateIndex::kErrorRecovery; }
	// T30
	std::optional<StateIndex> operator()(fsm::SuperResolution, fsm::SuperResolutionStop) const  { return StateIndex::kIdle; }
	// T31
	std::optional<StateIndex> operator()(fsm::SuperResolution, fsm::ModuleCrash) const   { return StateIndex::kErrorRecovery; }

	// Catch-all — no transition defined
	template <typename S, typename E>
	std::optional<StateIndex> operator()(S, E) const { return std::nullopt; }
};
} // anonymous namespace

void OmniEdgeDaemon::validateFsmCoverage() const
{
	OE_ZONE_SCOPED;

	CoverageProbe probe;
	auto states = allStates();
	auto events = allEvents();

	for (std::size_t si = 0; si < kStateCount; ++si) {
		bool hasAnyTransition = false;
		std::visit([&](const auto& s) {
			for (const auto& event : events) {
				std::visit([&](const auto& e) {
					if (probe(s, e).has_value()) hasAnyTransition = true;
				}, event);
			}
		}, states[si]);

		if (!hasAnyTransition) {
			OE_LOG_ERROR("fsm_dead_state: {} has NO outgoing transitions",
			             stateName(static_cast<StateIndex>(si)));
		}
	}

	// Verify kErrorRecovery can reach kIdle
	bool canReachIdle = false;
	auto recoveryIdx = static_cast<std::size_t>(StateIndex::kErrorRecovery);
	std::visit([&](const auto& s) {
		for (const auto& event : events) {
			std::visit([&](const auto& e) {
				auto dest = probe(s, e);
				if (dest && *dest == StateIndex::kIdle) canReachIdle = true;
			}, event);
		}
	}, states[recoveryIdx]);

	if (!canReachIdle) {
		OE_LOG_ERROR("fsm_no_recovery_path: kErrorRecovery cannot reach kIdle");
	}

	OE_LOG_INFO("fsm_coverage_validated: {} states, {} events",
	            kStateCount, kEventCount);
}

// == run (main poll loop) =====================================================

void OmniEdgeDaemon::run()
{
	lastWatchdogPoll_ = std::chrono::steady_clock::now();
	lastMemoryCheck_  = std::chrono::steady_clock::now();
	lastVramCheck_    = std::chrono::steady_clock::now();
	publishFsmState(StateIndex::kIdle);
	messageRouter_->publishModuleReady();
	messageRouter_->run();   // blocks until messageRouter_.stop()
}

void OmniEdgeDaemon::stop() noexcept
{
	if (messageRouter_) messageRouter_->stop();
}

// == Message routing ==========================================================

void OmniEdgeDaemon::handleMessage(const std::string& type, const nlohmann::json& msg)
{
	if (type == "ui_command")       { executeUiCommand(msg); return; }
	if (type == "transcription")    { lastTranscription_ = msg.value("text", ""); return; }
	if (type == "vad_status")       { if (!msg.value("speaking", true)) (void)applyFsmEvent(fsm::VadSilence{}); return; }
	if (type == "identity")         { if (promptAssembler_) promptAssembler_->setFaceIdentity(msg.value("name", ""), msg.value("confidence", 0.0f)); return; }
	if (type == "llm_response")     { processLlmTokenStream(msg); return; }

	// ── Four-mode architecture messages ─────────────────────────────────
	// Unified conversation model publishes audio_output when TTS is native
	// (Qwen2.5-Omni).  The bridge forwards it to the /audio WebSocket channel.
	if (type == "audio_output") {
		if (messageRouter_) messageRouter_->publish("audio_output", msg);
		return;
	}
	// Image Transform result — Stable Diffusion pipeline completed.
	if (type == "transform_result") {
		bool success = msg.value("success", false);
		std::string style = msg.value("style", "");
		if (success) {
			(void)applyFsmEvent(fsm::ImageTransformComplete{});
		}
		eventBus_.publish(ImageTransformResult{
			.style        = style,
			.success      = success,
			.errorMessage = msg.value("error", ""),
		});
		return;
	}

	if (type == "heartbeat" && msg.contains("module")) {
		const auto modName = msg["module"].get<std::string>();
		// Only accept heartbeats from known modules to prevent unbounded map growth.
		if (moduleIndex_.contains(modName)) {
			moduleHeartbeats_[modName] = std::chrono::steady_clock::now();
		}
		return;
	}

	if (type == "module_ready" && msg.contains("module")) {
		std::string modName = msg["module"].get<std::string>();
		for (auto& mod : modules_) {
			if (mod.name == modName && !mod.ready) {
				mod.ready = true;
				moduleHeartbeats_[modName] = std::chrono::steady_clock::now();
				OE_LOG_INFO("module_ready_late: {}", modName);
				break;
			}
		}
	}
}

// == LLM availability check ===================================================

bool OmniEdgeDaemon::isLlmAvailable() const
{
	// Check both legacy "llm" module and four-mode "conversation_model"
	auto checkRunning = [&](const std::string& name) -> bool {
		auto it = moduleIndex_.find(name);
		return it != moduleIndex_.end() && modules_[it->second].isRunning();
	};
	return checkRunning("llm") || checkRunning("conversation_model");
}

// == executeUiCommand =========================================================

void OmniEdgeDaemon::executeUiCommand(const nlohmann::json& cmd)
{
	const auto action = parseUiAction(cmd.value("action", std::string{}));
	if (action == UiAction::kUnknown) return;

	switch (action) {
	case UiAction::kPushToTalk: {
		// STT is on-demand — auto-start if not running
		auto sttIt = moduleIndex_.find("stt");
		bool sttRunning = (sttIt != moduleIndex_.end() && modules_[sttIt->second].isRunning());
		if (!sttRunning) {
			// Auto-start STT module on first PTT press
			auto toggleCmd = nlohmann::json{{"enabled", true}};
			handleToggleModule(UiAction::kToggleStt, toggleCmd);
			OE_LOG_INFO("stt_auto_started: triggered by push_to_talk");
			publishError("Starting speech recognition... Please press again in a moment.");
			break;
		}
		bool pressed = cmd.value("state", false);
		if (pressed) {
			(void)applyFsmEvent(fsm::PttPress{});
		} else if (stateMachine_.currentState() == StateIndex::kListening && lastTranscription_.empty()) {
			(void)applyFsmEvent(fsm::PttCancel{});
		} else {
			(void)applyFsmEvent(fsm::PttRelease{});
		}
		break;
	}
	case UiAction::kDescribeScene: {
		// LLMs handle vision natively — verify a conversation model is running
		if (!isLlmAvailable()) {
			publishError("Assistant is unavailable — LLM module is not running");
			break;
		}
		(void)applyFsmEvent(fsm::DescribeScene{});
		break;
	}
	case UiAction::kTextInput: {
		// Verify LLM module is alive (checks both legacy "llm" and "conversation_model")
		if (!isLlmAvailable()) {
			publishError("Assistant is unavailable — LLM module is not running");
			return;
		}
		lastTranscription_ = cmd.value("text", "");
		(void)applyFsmEvent(fsm::TextInput{});
		break;
	}
	case UiAction::kTtsComplete:
		(void)applyFsmEvent(fsm::TtsComplete{});
		break;

	case UiAction::kToggleVideoDenoise:
	case UiAction::kToggleAudioDenoise:
	case UiAction::kToggleStt:
	case UiAction::kToggleTts:
	case UiAction::kToggleFaceRecognition:
	case UiAction::kToggleBackgroundBlur:
		handleToggleModule(action, cmd);
		break;

	case UiAction::kSwitchMode: {
		// ── Four-mode architecture: switch between conversation/super_resolution/image_transform ──
		// Delegates to switchToMode() which handles profile switching, module
		// lifecycle, FSM state, and session persistence in one place.
		std::string requestedMode = cmd.value("mode", "");
		(void)switchToMode(requestedMode);
		break;
	}

	// ── Four-mode architecture commands ──────────────────────────────────
	case UiAction::kSelectConversationModel:
		handleSelectConversationModel(cmd);
		break;
	case UiAction::kSelectTtsModel:
		handleSelectTtsModel(cmd);
		break;
	case UiAction::kToggleVideoConversation:
		handleToggleVideoConversation(cmd);
		break;
	case UiAction::kUploadImage:
		handleUploadImage(cmd);
		break;
	case UiAction::kSelectStyle:
		handleSelectImageStyle(cmd);
		break;
	case UiAction::kStartImageTransform:
		handleStartImageTransform(cmd);
		break;

	default:
		break;
	}
}

void OmniEdgeDaemon::handleToggleModule(UiAction action, const nlohmann::json& cmd)
{
	// ── Lookup table: UiAction → module name + upstream dependency ────────
	struct ToggleInfo {
		std::string_view moduleName;
		std::string_view upstreamDependency; // empty = no dep check
	};
	static const std::unordered_map<UiAction, ToggleInfo> kToggleMap{
		{UiAction::kToggleVideoDenoise,    {"video_denoise",    "video_ingest"}},
		{UiAction::kToggleAudioDenoise,    {"audio_denoise",    "audio_ingest"}},
		{UiAction::kToggleStt,             {"stt",              "audio_ingest"}},
		{UiAction::kToggleTts,             {"tts",              ""}},
		{UiAction::kToggleFaceRecognition, {"face_recognition", "video_ingest"}},
		{UiAction::kToggleBackgroundBlur,  {"background_blur",  "video_ingest"}},
	};

	auto it = kToggleMap.find(action);
	if (it == kToggleMap.end()) return;

	const std::string moduleName{it->second.moduleName};
	const std::string upstreamDep{it->second.upstreamDependency};
	const bool enabled = cmd.value("enabled", false);

	// Find the module descriptor via index map (O(1) instead of linear scan)
	auto targetIt = moduleIndex_.find(moduleName);
	if (targetIt == moduleIndex_.end()) {
		OE_LOG_WARN("toggle_module_no_descriptor: module={}", moduleName);
		publishError(std::format("{} module not configured", moduleName));
		return;
	}
	ModuleDescriptor* targetMod = &modules_[targetIt->second];

	if (enabled && !targetMod->isRunning()) {
		// Check upstream dependency if one is defined
		if (!upstreamDep.empty()) {
			auto depIt = moduleIndex_.find(upstreamDep);
			const ModuleDescriptor* producer =
				(depIt != moduleIndex_.end()) ? &modules_[depIt->second] : nullptr;
			if (!producer || !producer->isRunning() || !producer->ready) {
				OE_LOG_WARN("toggle_dependency_not_ready: module={}, requires={}, running={}, ready={}",
					moduleName, upstreamDep,
					(producer ? producer->isRunning() : false),
					(producer ? producer->ready : false));
				publishError(std::format("Cannot start {} — {} must be running first",
					moduleName, upstreamDep));
				publishModuleStatus(moduleName, "unavailable");
				return;
			}
		}

		// ── Dynamic priority shift ─────────────────────────────────────
		// In the four-mode architecture, profile switches happen at mode
		// level (conversation/super_resolution/image_transform), not per
		// individual module toggle.  Module toggles within the current
		// mode do not change the interaction profile.
		// Legacy toggles (stt, tts, video_denoise) still work for
		// backward compatibility but use the current mode's profile.

		// Spawn the on-demand module — check VRAM budget cap first
		OE_LOG_INFO("spawning_on_demand_module: name={}", moduleName);

		const std::size_t moduleBudgetMiB = vramTracker_.moduleBudgetMiB(moduleName);
		if (moduleBudgetMiB > 0 && vramGate_) {
			auto vramResult = vramGate_->acquireVramForModule(moduleName, moduleBudgetMiB);
			if (!vramResult) {
				OE_LOG_ERROR("toggle_vram_acquire_failed: module={}, error={}",
				           moduleName, vramResult.error());
				publishError(std::format("Cannot start {} — insufficient GPU memory",
				           moduleName));
				publishModuleStatus(moduleName, "vram_insufficient");
				return;
			}
		}

		// Reset restart counter — explicit UI toggle resets the crash budget
		targetMod->restartCount = 0;

		auto spawnResult = launcher_.spawnModule(*targetMod);
		if (spawnResult) {
			targetMod->pid = spawnResult.value();
			if (moduleBudgetMiB > 0) {
				vramTracker_.markModuleLoaded(moduleName);
				eventBus_.publish(VramChanged{
					.moduleName = moduleName,
					.budgetMiB  = moduleBudgetMiB,
					.loaded     = true,
				});
			}
			// Seed heartbeat so the module has a full timeout window
			moduleHeartbeats_[moduleName] = std::chrono::steady_clock::now();
			publishModuleStatus(moduleName, "running");
		} else {
			// Rollback VRAM accounting — acquireVramForModule already
			// marked this module as loaded in VramTracker.
			if (moduleBudgetMiB > 0) {
				vramTracker_.markModuleUnloaded(moduleName);
				eventBus_.publish(VramChanged{
					.moduleName = moduleName,
					.budgetMiB  = moduleBudgetMiB,
					.loaded     = false,
				});
			}
			OE_LOG_ERROR("on_demand_spawn_failed: name={}, error={}",
				moduleName, spawnResult.error());
			publishModuleStatus(moduleName, "down");
			publishError(std::format("Failed to start {}", moduleName));
		}
	} else if (!enabled && targetMod->isRunning()) {
		// Stop the on-demand module
		OE_LOG_INFO("stopping_on_demand_module: name={}, pid={}",
			moduleName, targetMod->pid);
		launcher_.stopModule(*targetMod);
		vramTracker_.markModuleUnloaded(moduleName);
		eventBus_.publish(VramChanged{
			.moduleName = moduleName,
			.budgetMiB  = vramTracker_.moduleBudgetMiB(moduleName),
			.loaded     = false,
		});
		publishModuleStatus(moduleName, "unavailable");

		// ── Four-mode architecture: profile stays at mode level ──────
		// In the four-mode system, interaction profiles are tied to the
		// active GPU mode (conversation / super_resolution / image_transform),
		// not individual module toggles.  Disabling a legacy module does
		// not revert the profile — the mode switch handles that.
		// Auto-relaunch high-priority modules that fit within the current
		// mode's VRAM budget.
		{
			for (auto& mod : modules_) {
				if (mod.isRunning() || mod.degraded) continue;

				const int modPriority =
					priorityScheduler_.modulePriority(mod.name);
				if (modPriority < kAutoLaunchPriorityThreshold) continue;

				const std::size_t modBudgetMiB =
					vramTracker_.moduleBudgetMiB(mod.name);
				if (modBudgetMiB == 0) continue;

				if (vramGate_) {
					auto vramResult = vramGate_->acquireVramForModule(
						mod.name, modBudgetMiB);
					if (!vramResult) {
						OE_LOG_DEBUG("toggle_relaunch_skip: module={}, "
						           "priority={}, error={}",
						           mod.name, modPriority, vramResult.error());
						continue;
					}
				}

				auto spawnResult = launcher_.spawnModule(mod);
				if (spawnResult) {
					eventBus_.publish(VramChanged{
						.moduleName = mod.name,
						.budgetMiB  = modBudgetMiB,
						.loaded     = true,
					});
					moduleHeartbeats_[mod.name] =
						std::chrono::steady_clock::now();
					publishModuleStatus(mod.name, "running");
					OE_LOG_INFO("toggle_relaunched: module={}, "
					          "priority={}", mod.name, modPriority);
				} else {
					vramTracker_.markModuleUnloaded(mod.name);
					eventBus_.publish(VramChanged{
						.moduleName = mod.name,
						.budgetMiB  = modBudgetMiB,
						.loaded     = false,
					});
					OE_LOG_WARN("toggle_relaunch_failed: module={}, "
					          "error={}", mod.name, spawnResult.error());
				}
			}
		}
	}
}

void OmniEdgeDaemon::processLlmTokenStream(const nlohmann::json& msg)
{
	std::string token = msg.value("token", "");
	bool finished     = msg.value("finished", false);
	bool sentBoundary = msg.value("sentence_boundary", false);

	// Reserve on first token to avoid O(n²) reallocation across the generation
	constexpr std::size_t kExpectedResponseLen = 2048;
	if (lastLlmResponse_.empty()) {
		lastLlmResponse_.reserve(kExpectedResponseLen);
		pendingSentence_.reserve(kExpectedResponseLen / 4);
	}
	lastLlmResponse_ += token;
	pendingSentence_ += token;

	// First sentence triggers PROCESSING → SPEAKING
	if (sentBoundary && stateMachine_.currentState() == StateIndex::kProcessing) {
		(void)applyFsmEvent(fsm::LlmFirstSentence{});
	}

	// Forward completed sentence to TTS
	if (sentBoundary && !pendingSentence_.empty()) {
		publishSentenceToTts(pendingSentence_);
		pendingSentence_.clear();
	}

	if (finished) {
		if (!pendingSentence_.empty()) {
			publishSentenceToTts(pendingSentence_);
			pendingSentence_.clear();
		}
		(void)applyFsmEvent(fsm::LlmComplete{});
	}
}

// == Periodic tasks ===========================================================

void OmniEdgeDaemon::tickPeriodicTasks()
{
	// Drain deferred state-machine event (avoids reentrant dispatch from handlers)
	if (pendingEvent_.has_value()) {
		auto evt = std::move(*pendingEvent_);
		pendingEvent_.reset();
		(void)applyFsmEvent(evt);
	}

	// Check timeouts
	auto now = std::chrono::steady_clock::now();
	if (stateMachine_.currentState() == StateIndex::kProcessing && now >= llmTimeoutDeadline_) {
		(void)applyFsmEvent(fsm::LlmTimeout{});
	}
	if (stateMachine_.currentState() == StateIndex::kImageTransform && now >= imageTransformTimeoutDeadline_) {
		(void)applyFsmEvent(fsm::ImageTransformTimeout{});
	}

	// Watchdog
	if (now - lastWatchdogPoll_ >= std::chrono::milliseconds{config_.watchdogPollMs}) {
		watchdogPoll();
		lastWatchdogPoll_ = now;
	}

	// Memory watchdog — check RSS limits (less frequent than main watchdog)
	const int memCheckMs = iniConfig_.crashProtection().memoryCheckIntervalMs;
	if (memCheckMs > 0 &&
	    now - lastMemoryCheck_ >= std::chrono::milliseconds{memCheckMs}) {
		memoryWatchdog();
		lastMemoryCheck_ = now;
	}

	// VRAM watchdog — check GPU VRAM usage and evict if over cap
	const int vramCheckMs = iniConfig_.vramLimits().vramCheckIntervalMs;
	if (vramCheckMs > 0 &&
	    now - lastVramCheck_ >= std::chrono::milliseconds{vramCheckMs}) {
		vramWatchdog();
		lastVramCheck_ = now;
	}

	// Periodic session save
	if (sessionPersistence_ && sessionPersistence_->shouldPeriodicSave()) {
		(void)sessionPersistence_->save(buildSessionState());
	}
}

// == Watchdog =================================================================

void OmniEdgeDaemon::watchdogPoll()
{
	// ── Phase 1: Detect all crashed modules ──────────────────────────────
	// Collect crashes before firing any state-machine events so we don't
	// re-enter the FSM multiple times in one poll cycle.
	struct CrashRecord {
		std::string name;
		pid_t       pidWas;
		std::size_t index;
		bool        wasSignalKill{false};
		int         signal{0};
		int         exitCode{0};
	};
	std::vector<CrashRecord> crashes;

	// Exit code 78 (EX_CONFIG) means "inferencer not available" — the module
	// detected at startup that a required dependency is missing (e.g., LLM
	// built with stub inferencer).  Do not restart; mark as degraded instead.
	static constexpr int kExitCodeConfigUnavailable = 78;

	for (std::size_t i = 0; i < modules_.size(); ++i) {
		auto& mod = modules_[i];
		if (!mod.isRunning()) continue;

		// Capture PID before checkExitedWithSignal() resets it to -1.
		const pid_t pidBefore = mod.pid;
		int exitSignal = 0;
		int exitCode   = 0;
		if (!launcher_.checkExitedWithSignal(mod, exitSignal, exitCode)) continue;

		CrashRecord record{mod.name, pidBefore, i, false, 0, exitCode};

		if (exitCode == kExitCodeConfigUnavailable) {
			OE_LOG_WARN("module_inferencer_unavailable: module={}, pid_was={} — "
			            "dependency missing, marking degraded (no restart)",
			            mod.name, pidBefore);
			mod.degraded = true;
			publishModuleStatus(mod.name, "degraded");
			if (vramTracker_.moduleBudgetMiB(mod.name) > 0) {
				vramTracker_.markModuleUnloaded(mod.name);
			}
			continue;  // skip adding to crashes — no restart
		}

		if (exitSignal == SIGSEGV || exitSignal == SIGBUS || exitSignal == SIGABRT) {
			record.wasSignalKill = true;
			record.signal = exitSignal;
			OE_LOG_ERROR("crash_detected: module={}, pid_was={}, signal={}({})",
			           mod.name, pidBefore, exitSignal, ::strsignal(exitSignal));

			// Read the diagnostic ring dump written by the module's
			// signal handler — shows what it was doing when it died.
			std::filesystem::path dumpPath =
				std::filesystem::path{"logs"} / ("crash_" + mod.name + ".dump");
			if (auto dumpResult = readText(dumpPath); dumpResult && !dumpResult->empty()) {
				OE_LOG_ERROR("crash_diagnostic_dump: module={}\n{}",
				           mod.name, *dumpResult);
			}
		} else {
			OE_LOG_ERROR("crash_detected: module={}, pid_was={}", mod.name, pidBefore);
		}
		crashes.push_back(record);

		// Notify subscribers that a module crashed
		eventBus_.publish(ModuleCrashed{
			.moduleName = record.name,
			.pid        = record.pidWas,
			.signal     = record.signal,
		});

		// Update VRAM accounting — the crashed process freed its GPU memory
		// Only for GPU-consuming modules registered in the tracker.
		if (vramTracker_.moduleBudgetMiB(mod.name) > 0) {
			vramTracker_.markModuleUnloaded(mod.name);
			eventBus_.publish(VramChanged{
				.moduleName = mod.name,
				.budgetMiB  = vramTracker_.moduleBudgetMiB(mod.name),
				.loaded     = false,
			});
		}
	}

	// ── Phase 1.5: Detect hung modules (stale heartbeats) ───────────────
	// A module whose process is still alive but has stopped publishing
	// heartbeats is likely deadlocked or stuck in an infinite loop.
	// SIGKILL it now; the next watchdog poll will reap it as a crash.
	{
		const auto now = std::chrono::steady_clock::now();
		const auto heartbeatTimeout =
			std::chrono::milliseconds{config_.heartbeatTimeoutMs};

		for (auto& mod : modules_) {
			if (!mod.isRunning()) continue;
			if (!mod.ready) continue;  // not yet initialised — no heartbeats expected

			auto it = moduleHeartbeats_.find(mod.name);
			if (it == moduleHeartbeats_.end()) continue;

			const auto silence = now - it->second;
			if (silence > heartbeatTimeout) {
				const auto silenceMs =
					std::chrono::duration_cast<std::chrono::milliseconds>(silence).count();
				OE_LOG_ERROR("heartbeat_timeout: module={}, pid={}, silent_for_ms={} "
				             "— likely hung, sending SIGKILL",
				             mod.name, mod.pid, silenceMs);

				eventBus_.publish(ModuleHung{
					.moduleName = mod.name,
					.pid        = mod.pid,
					.silenceMs  = static_cast<int>(silenceMs),
				});
				publishModuleStatus(mod.name, "hung");

				::kill(mod.pid, SIGKILL);

				// Remove entry so we don't re-trigger on the next poll
				// while the process is being reaped.
				moduleHeartbeats_.erase(it);
			}
		}
	}

	if (crashes.empty()) return;

	// ── Phase 2: Single state-machine transition ──────────────────────
	lastCrashedModule_ = crashes.back().name;
	(void)applyFsmEvent(fsm::ModuleCrash{});

	// ── Phase 2.5: Clean up stale SHM segments ─────────────────────────
	// Unlink SHM segments owned by crashed modules before restarting.
	// Prevents stale-data reads by consumers that might map the old segment.
	for (const auto& crash : crashes) {
		shmRegistry_.cleanupForModule(crash.name);
	}

	// ── Phase 3: Restart crashed modules (with segfault protection) ──────
	bool anyRestarted = false;
	const int maxSegfaults = iniConfig_.crashProtection().maxSegfaultRestarts;

	for (const auto& crash : crashes) {
		auto& mod = modules_[crash.index];

		// Check if the module died from a signal (SIGSEGV, SIGBUS, SIGABRT)
		// that indicates a programming error — restarting will just crash again.
		if (crash.wasSignalKill) {
			segfaultCounts_[crash.name]++;
			int count = segfaultCounts_[crash.name];
			OE_LOG_ERROR("signal_crash_detected: module={}, signal={}, "
			           "segfault_count={}/{}", crash.name, crash.signal,
			           count, maxSegfaults);

			if (count > maxSegfaults) {
				OE_LOG_ERROR("module_permanently_disabled: module={} — "
				           "exceeded max segfault restarts ({}). "
				           "Disable in omniedge.ini and investigate.",
				           crash.name, maxSegfaults);
				eventBus_.publish(ModuleDisabled{
					.moduleName = crash.name,
					.reason     = std::format("exceeded max segfault restarts ({})", maxSegfaults),
				});
				publishModuleStatus(crash.name, "disabled_segfault");
				publishError(std::format(
					"{} crashed with signal {} — disabled to protect system stability. "
					"Check logs and disable in omniedge.ini.",
					crash.name, crash.signal));
				continue;
			}
		}

		// Before restarting, acquire VRAM (may evict conflicting modules).
		// waitForVramAvailable only polls — acquireVramForModule actively
		// evicts lowest-priority modules to free space.
		const std::size_t moduleBudgetMiB = vramTracker_.moduleBudgetMiB(crash.name);
		if (vramGate_ && moduleBudgetMiB > 0) {
			auto vramResult = vramGate_->acquireVramForModule(crash.name, moduleBudgetMiB);
			if (!vramResult) {
				OE_LOG_ERROR("restart_vram_acquire_failed: module={}, error={}",
				           crash.name, vramResult.error());
				publishModuleStatus(crash.name, "vram_insufficient");
				continue;
			}
		}

		auto restartResult = launcher_.restartModule(mod);
		if (!restartResult) {
			OE_LOG_ERROR("restart_failed: module={}, error={}",
			           crash.name, restartResult.error());
			publishModuleStatus(crash.name, "down");
		} else {
			if (vramTracker_.moduleBudgetMiB(crash.name) > 0) {
				vramTracker_.markModuleLoaded(crash.name);
				eventBus_.publish(VramChanged{
					.moduleName = crash.name,
					.budgetMiB  = vramTracker_.moduleBudgetMiB(crash.name),
					.loaded     = true,
				});
			}
			eventBus_.publish(ModuleRestarted{
				.moduleName = crash.name,
				.newPid     = mod.pid,
			});
			// Seed heartbeat clock so the new process has a full timeout
			// window to initialise and begin publishing heartbeats.
			moduleHeartbeats_[crash.name] = std::chrono::steady_clock::now();
			publishModuleStatus(crash.name, "restarting");
			anyRestarted = true;
		}
	}

	// Fire a single recovery event if any modules were successfully restarted.
	if (anyRestarted) {
		pendingEvent_ = fsm::ModuleRecovered{};
	}

	// ── Phase 4: Sweep for zombie children ──────────────────────────────
	// Reap any child processes that exited but weren't collected by the
	// per-module check above (e.g., process group children, race conditions).
	while (::waitpid(-1, nullptr, WNOHANG) > 0) { /* reap */ }
}

// == State machine actions ====================================================

void OmniEdgeDaemon::fsmOnStartListening()
{
	lastTranscription_.clear();
	publishFsmState(StateIndex::kListening);
}

void OmniEdgeDaemon::fsmOnStopListeningAndSubmitPrompt()
{
	publishFsmState(StateIndex::kProcessing);
	buildAndPublishLlmPrompt(lastTranscription_);
	llmTimeoutDeadline_ = std::chrono::steady_clock::now() +
	                      std::chrono::seconds{config_.llmGenerationTimeoutS};
}

void OmniEdgeDaemon::fsmOnReturnToIdle()
{
	lastTranscription_.clear();
	publishFsmState(StateIndex::kIdle);
}

void OmniEdgeDaemon::buildAndPublishLlmPrompt(const std::string& userText)
{
	if (!promptAssembler_ || !messageRouter_) return;

	// Gate on LLM readiness — do not publish if no LLM-capable module has
	// sent its module_ready message.  Publishing before the inferencer is fully
	// initialised triggers a crash (munmap_chunk / MPI_ABORT) because TRT-LLM
	// cannot handle inference requests while the KV cache is still being allocated.
	// Check both legacy "llm" and four-mode "conversation_model" (matches isLlmAvailable).
	auto isModuleReady = [&](const std::string& name) -> bool {
		auto it = moduleIndex_.find(name);
		return it != moduleIndex_.end()
		    && modules_[it->second].isRunning()
		    && modules_[it->second].ready;
	};
	if (!isModuleReady("llm") && !isModuleReady("conversation_model")) {
		OE_LOG_WARN("conversation_prompt_deferred: LLM not ready, dropping prompt (len={})", userText.size());
		return;
	}

	auto payload = promptAssembler_->assemble(userText);
	OE_LOG_INFO("conversation_prompt_published: text_len={}, payload_keys={}",
	           userText.size(), payload.size());
	messageRouter_->publish("conversation_prompt", payload);
}

void OmniEdgeDaemon::publishSentenceToTts(const std::string& sentence)
{
	if (!messageRouter_) return;

	// TTS is on-demand — skip if not running (text-only response mode).
	// Check both standalone TTS module and native-TTS conversation models
	// (e.g. Qwen2.5-Omni which handles TTS internally).
	auto ttsIt = moduleIndex_.find("tts");
	bool standaloneTtsRunning = (ttsIt != moduleIndex_.end() && modules_[ttsIt->second].isRunning());

	// Check if the active conversation model handles TTS natively
	auto convIt = moduleIndex_.find("conversation_model");
	bool nativeTtsAvailable = false;
	if (convIt != moduleIndex_.end() && modules_[convIt->second].isRunning()) {
		auto flagsIt = conversationModelFlags_.find(activeConversationModel_);
		if (flagsIt != conversationModelFlags_.end() && !flagsIt->second.needsTts) {
			nativeTtsAvailable = true;
		}
	}

	if (!standaloneTtsRunning && !nativeTtsAvailable) return;

	messageRouter_->publish("tts_sentence", nlohmann::json{
		{"v",        kSchemaVersion},
		{"type",     "tts_sentence"},
		{"sentence", sentence},
	});
}

void OmniEdgeDaemon::fsmOnFinishTurn()
{
	if (promptAssembler_ && !lastTranscription_.empty()) {
		promptAssembler_->addToHistory(lastTranscription_, lastLlmResponse_);
	}
	lastTranscription_.clear();
	lastLlmResponse_.clear();
	publishFsmState(StateIndex::kIdle);
}

void OmniEdgeDaemon::fsmOnBargeIn()
{
	if (messageRouter_) {
		messageRouter_->publish("conversation_prompt", nlohmann::json{
			{"v",    kSchemaVersion},
			{"type", "cancel_generation"},
		});
	}
	publishFsmState(StateIndex::kInterrupted);
	pendingEvent_ = fsm::InterruptDone{};
}

void OmniEdgeDaemon::fsmOnModuleCrash(const std::string& moduleName)
{
	publishFsmState(StateIndex::kErrorRecovery);
	if (!moduleName.empty()) {
		publishModuleStatus(moduleName, "error");
	}
	OE_LOG_ERROR("error_recovery_started: {}", moduleName);
}

// == Session state ============================================================

nlohmann::json OmniEdgeDaemon::buildSessionState() const
{
	nlohmann::json modulesJson = nlohmann::json::array();
	for (const auto& mod : modules_) {
		modulesJson.push_back({
			{"name",     mod.name},
			{"running",  mod.isRunning()},
			{"pid",      mod.pid},
			{"restarts", mod.restartCount},
			{"degraded", mod.degraded},
		});
	}

	return {
		{"state",               std::string(stateName(stateMachine_.currentState()))},
		{"mode",                modeOrchestrator_ ? modeOrchestrator_->currentMode() : "conversation"},
		{"conversation_model",  activeConversationModel_},
		{"tier",                gpuProfiler_ ? std::string(tierName(gpuProfiler_->tier())) : "unknown"},
		{"history_size",        promptAssembler_ ? promptAssembler_->historySize() : 0},
		{"modules",             modulesJson},
	};
}

void OmniEdgeDaemon::restoreSessionState(const nlohmann::json& state)
{
	if (state.contains("mode") && modeOrchestrator_) {
		modeOrchestrator_->setInitialMode(state["mode"].get<std::string>());
	}
}

// == Publish helpers ==========================================================

void OmniEdgeDaemon::publishFsmState(StateIndex s)
{
	if (!messageRouter_) return;
	messageRouter_->publish("module_status", nlohmann::json{
		{"v",      kSchemaVersion},
		{"type",   "module_status"},
		{"module", "omniedge_orchestrator"},
		{"state",  std::string(stateName(s))},
	});
}

void OmniEdgeDaemon::publishModuleStatus(const std::string& module, const std::string& status)
{
	if (!messageRouter_) return;
	messageRouter_->publish("module_status", nlohmann::json{
		{"v",      kSchemaVersion},
		{"type",   "module_status"},
		{"module", module},
		{"status", status},
	});
}

void OmniEdgeDaemon::publishError(const std::string& message)
{
	if (!messageRouter_) return;
	messageRouter_->publish("module_status", nlohmann::json{
		{"v",       kSchemaVersion},
		{"type",    "error"},
		{"message", message},
	});
}

// == switchToMode (non-recursive mode switch) ====================================

bool OmniEdgeDaemon::switchToMode(const std::string& requestedMode)
{
	if (!modeOrchestrator_ || !vramGate_ || requestedMode.empty()) return false;

	// Apply interaction profile for the target mode
	if (requestedMode == "conversation") {
		applyInteractionProfile(InteractionProfile::kConversation);
	} else if (requestedMode == "super_resolution") {
		applyInteractionProfile(InteractionProfile::kSuperResolution);
	} else if (requestedMode == "image_transform") {
		applyInteractionProfile(InteractionProfile::kImageTransform);
	} else if (requestedMode == "sam2_segmentation") {
		applyInteractionProfile(InteractionProfile::kSam2Segmentation);
	}

	auto switchResult = modeOrchestrator_->handleModeSwitch(
		requestedMode, modules_, launcher_, *vramGate_,
		messageRouter_->context());
	if (!switchResult) {
		OE_LOG_ERROR("mode_switch_failed: {}", switchResult.error());
		eventBus_.publish(ModeChanged{requestedMode, false});
		return false;
	}

	eventBus_.publish(ModeChanged{requestedMode, true});
	if (requestedMode == "super_resolution") {
		stateMachine_.forceState(StateIndex::kSuperResolution);
		publishFsmState(StateIndex::kSuperResolution);
	} else if (requestedMode == "sam2_segmentation") {
		stateMachine_.forceState(StateIndex::kSam2Segmentation);
		publishFsmState(StateIndex::kSam2Segmentation);
	} else {
		stateMachine_.forceState(StateIndex::kIdle);
		publishFsmState(StateIndex::kIdle);
	}
	if (sessionPersistence_) {
		(void)sessionPersistence_->save(buildSessionState());
	}
	return true;
}

// == Dynamic Interaction Profile ===============================================
// Shifts module priorities when the user's interaction pattern changes.
// Profile switches re-order eviction candidates so that the lowest-priority
// modules under the new profile are evicted first when VRAM is scarce.

void OmniEdgeDaemon::applyInteractionProfile(InteractionProfile profile)
{
	const auto currentProfile = priorityScheduler_.currentProfile();
	if (profile == currentProfile) return;

	OE_LOG_INFO("interaction_profile_switch: from={}, to={}",
	          profileName(currentProfile), profileName(profile));

	// Apply the new profile — get back the list of modules whose priority changed
	auto changed = priorityScheduler_.applyProfile(profile, iniConfig_.profilePriorities(profile));

	// Sync VramTracker eviction priorities with PriorityScheduler
	for (const auto& [moduleName, newPriority] : changed) {
		vramTracker_.updateEvictPriority(moduleName, newPriority);

		// Update the ModuleDescriptor evictPriority for mode_orchestrator sorting
		for (auto& mod : modules_) {
			if (mod.name == moduleName) {
				mod.evictPriority = newPriority;
				break;
			}
		}
	}

	eventBus_.publish(InteractionProfileChanged{
		.fromProfile    = std::string{profileName(currentProfile)},
		.toProfile      = std::string{profileName(profile)},
		.modulesChanged = changed.size(),
	});

	OE_LOG_INFO("interaction_profile_applied: profile={}, modules_changed={}",
	          profileName(profile), changed.size());
}

// == Memory Watchdog ==========================================================
// Reads /proc/<pid>/statm for each running module and kills any that exceed
// their RSS memory limit (from omniedge.ini [memory_limits_mb]).
// This prevents a single runaway module from consuming all host RAM and
// crashing WSL / the entire Windows system.

void OmniEdgeDaemon::memoryWatchdog()
{
	if (!iniConfig_.loaded()) return;
	const bool killGroup = iniConfig_.crashProtection().killProcessGroup;

	for (auto& mod : modules_) {
		if (!mod.isRunning()) continue;

		const std::size_t limitMb = iniConfig_.memoryLimitMb(mod.name);
		if (limitMb == 0) continue;  // No limit configured

		// Read RSS from /proc/<pid>/statm (field 1 = resident pages)
		const std::string statmPath = std::format("/proc/{}/statm", mod.pid);
		std::ifstream statmFile(statmPath);
		if (!statmFile.is_open()) continue;  // Process may have just exited

		std::size_t vmSize = 0;
		std::size_t rssPages = 0;
		statmFile >> vmSize >> rssPages;
		if (statmFile.fail()) continue;

		// Convert pages to MB (getpagesize() is typically 4096)
		const std::size_t pageSizeBytes = static_cast<std::size_t>(::getpagesize());
		const std::size_t rssMb = (rssPages * pageSizeBytes) / (1024 * 1024);

		if (rssMb > limitMb) {
			OE_LOG_ERROR("memory_limit_exceeded: module={}, pid={}, rss_mb={}, limit_mb={} "
			           "— KILLING to protect system stability",
			           mod.name, mod.pid, rssMb, limitMb);

			// Kill immediately — no grace period for OOM, system stability is at risk
			if (killGroup) {
				// Kill the entire process group to catch leaked children
				::kill(-mod.pid, SIGKILL);
			}
			::kill(mod.pid, SIGKILL);
			// Bounded reap: WNOHANG with deadline to avoid hanging on D-state
			{
				const auto reapDeadline =
					std::chrono::steady_clock::now()
					+ std::chrono::seconds{kKillReapTimeoutS};
				while (std::chrono::steady_clock::now() < reapDeadline) {
					if (::waitpid(mod.pid, nullptr, WNOHANG) != 0) break;
					std::this_thread::sleep_for(std::chrono::milliseconds{50});
				}
			}

			publishError(std::format(
				"{} killed: using {} MB RAM (limit: {} MB). "
				"Check for memory leaks or increase limit in omniedge.ini.",
				mod.name, rssMb, limitMb));
			publishModuleStatus(mod.name, "oom_killed");

			mod.pid = -1;
			mod.ready = false;
			if (vramTracker_.moduleBudgetMiB(mod.name) > 0) {
				vramTracker_.markModuleUnloaded(mod.name);
				eventBus_.publish(VramChanged{
					.moduleName = mod.name,
					.budgetMiB  = vramTracker_.moduleBudgetMiB(mod.name),
					.loaded     = false,
				});
			}
			eventBus_.publish(ModuleDisabled{
				.moduleName = mod.name,
				.reason     = std::format("OOM killed: {} MB RSS > {} MB limit", rssMb, limitMb),
			});

			// Don't auto-restart OOM-killed modules — they'll just OOM again.
			// The watchdogPoll() will detect the exit on the next cycle but
			// we've already set pid=-1, so checkExited will return true (already dead).
			OE_LOG_WARN("oom_killed_no_restart: module={} — fix the memory leak or "
			          "increase memory_limits_mb in omniedge.ini", mod.name);
		}
	}
}

// == VRAM Watchdog ============================================================
// Probes actual GPU VRAM usage via cudaMemGetInfo and evicts lowest-priority
// modules when usage exceeds the critical threshold from omniedge.ini.
// Evicts at most one module per cycle to avoid over-eviction while the CUDA
// driver reclaims memory from a just-killed process.

void OmniEdgeDaemon::vramWatchdog()
{
	if (!gpuProfiler_) return;

	const auto& vramConfig = iniConfig_.vramLimits();
	if (vramConfig.maxTotalVramMb == 0) return;  // Disabled

	const std::size_t usedMb = gpuProfiler_->liveUsedMb();

	// Warning threshold — log but take no action
	if (usedMb > vramConfig.warningThresholdMb &&
	    usedMb <= vramConfig.criticalThresholdMb) {
		OE_LOG_WARN("vram_warning: used_mb={}/{} (warning threshold: {} MB)",
		          usedMb, vramConfig.maxTotalVramMb, vramConfig.warningThresholdMb);
	}

	// Below critical — nothing to do
	if (usedMb <= vramConfig.criticalThresholdMb) return;

	OE_LOG_ERROR("vram_critical: used_mb={}/{} exceeds critical threshold {} MB — evicting",
	           usedMb, vramConfig.maxTotalVramMb, vramConfig.criticalThresholdMb);

	// Evict one lowest-priority module per watchdog cycle
	auto candidates = priorityScheduler_.evictionCandidates();
	for (const auto& candidate : candidates) {
		OE_LOG_WARN("vram_watchdog_evicting: module={}, budget_mb={}, priority={}",
		          candidate.moduleName, candidate.vramBudgetMiB, candidate.priority);

		// Find and stop the module process
		for (auto& mod : modules_) {
			if (mod.name == candidate.moduleName && mod.isRunning()) {
				launcher_.stopModule(mod);
				vramTracker_.markModuleUnloaded(mod.name);
				eventBus_.publish(VramEviction{
					.moduleName = mod.name,
					.usedMb     = usedMb,
					.capMb      = vramConfig.maxTotalVramMb,
				});
				eventBus_.publish(VramChanged{
					.moduleName = mod.name,
					.budgetMiB  = vramTracker_.moduleBudgetMiB(mod.name),
					.loaded     = false,
				});
				publishModuleStatus(mod.name, "vram_evicted");
				publishError(std::format(
					"{} evicted — GPU VRAM {} MB exceeds {} MB cap",
					mod.name, usedMb, vramConfig.maxTotalVramMb));

				OE_LOG_INFO("vram_watchdog_evicted: module={}, was_using_mb={}",
				          mod.name, usedMb);
				return;  // One eviction per cycle
			}
		}
	}

	// No evictable candidates — log severe warning
	OE_LOG_ERROR("vram_critical_no_candidates: used_mb={}/{} but no evictable "
	           "modules available — system may crash",
	           usedMb, vramConfig.maxTotalVramMb);
}

// == Four-Mode Architecture Handlers ==========================================
//
// The system operates in four mutually exclusive GPU inference modes:
//   1. Conversation (default) — Qwen2.5-Omni-7B / 3B / Gemma 4 E4B
//   2. Super Resolution — BasicVSR++ temporal video enhancement
//   3. Image Transform — Stable Diffusion + ControlNet/IP-Adapter
//
// Always-on: video_ingest, audio_ingest, background_blur (YOLO), ISP, ws_bridge.
//
// Switching models within Conversation mode follows the same kill-spawn cycle:
// the daemon terminates the previous model process and spawns the newly
// selected one.  Qwen2.5-Omni variants handle STT+LLM+TTS natively in a
// single process.  Gemma 4 E4B accepts audio+image natively but needs a
// separate TTS process, so when selected, the UI reveals a TTS model selector.

void OmniEdgeDaemon::handleSelectConversationModel(const nlohmann::json& cmd)
{
	const std::string newModel = cmd.value("model", "");
	if (newModel.empty()) return;

	// Validate model name against parsed conversation_models config
	if (conversationModelFlags_.find(newModel) == conversationModelFlags_.end()) {
		OE_LOG_WARN("invalid_conversation_model: '{}'", newModel);
		publishError(std::format("Unknown conversation model: {}", newModel));
		return;
	}

	// Validate engine directory exists before attempting to spawn — prevents
	// spawn-crash-restart loops for models that aren't installed.
	{
		auto dirIt = conversationModelFlags_.find(newModel);
		if (dirIt != conversationModelFlags_.end() && !dirIt->second.resolvedEngineDir.empty()) {
			if (!std::filesystem::exists(dirIt->second.resolvedEngineDir)) {
				OE_LOG_ERROR("conversation_model_dir_missing: model={}, dir={}",
				           newModel, dirIt->second.resolvedEngineDir);
				publishError(std::format("{} model not installed — directory not found: {}",
				           newModel, dirIt->second.resolvedEngineDir));
				return;
			}
		}
	}

	if (newModel == activeConversationModel_) return;

	const std::string previousModel = activeConversationModel_;
	OE_LOG_INFO("conversation_model_switch: {} -> {}", previousModel, newModel);

	// Determine companion module requirements from config flags (data-driven)
	auto prevFlags = conversationModelFlags_.find(previousModel);
	auto newFlags  = conversationModelFlags_.find(newModel);
	const bool previousNeedsTts = prevFlags != conversationModelFlags_.end() && prevFlags->second.needsTts;
	const bool newNeedsTts      = newFlags != conversationModelFlags_.end() && newFlags->second.needsTts;

	// ── Step 1: Kill the current conversation model process ──────────────
	auto convIt = moduleIndex_.find("conversation_model");
	if (convIt != moduleIndex_.end()) {
		ModuleDescriptor& convMod = modules_[convIt->second];
		if (convMod.isRunning()) {
			const pid_t savedPid = convMod.pid;
			launcher_.stopModule(convMod);
			// Release VRAM accounting for the old model
			vramTracker_.markModuleUnloaded("conversation_model");
			eventBus_.publish(VramChanged{
				.moduleName = "conversation_model",
				.budgetMiB  = vramTracker_.moduleBudgetMiB("conversation_model"),
				.loaded     = false,
			});
			OE_LOG_INFO("conversation_model_killed: model={}, pid={}",
			           previousModel, savedPid);
		}
	}

	// ── Step 2: Kill companion TTS if the previous model had one ─────────
	if (previousNeedsTts && !newNeedsTts) {
		auto ttsIt = moduleIndex_.find("tts");
		if (ttsIt != moduleIndex_.end()) {
			ModuleDescriptor& ttsMod = modules_[ttsIt->second];
			if (ttsMod.isRunning()) {
				launcher_.stopModule(ttsMod);
				vramTracker_.markModuleUnloaded("tts");
				eventBus_.publish(VramChanged{
					.moduleName = "tts",
					.budgetMiB  = iniConfig_.vramBudgetMiB("tts", kTtsMiB),
					.loaded     = false,
				});
				OE_LOG_INFO("companion_tts_killed: no longer needed for {}", newModel);
			}
		}
	}

	// ── Step 3: Update VRAM budget for the new model ─────────────────────
	// Budgets from INI [vram_budgets] with compile-time fallbacks
	std::size_t newBudgetMiB = iniConfig_.vramBudgetMiB("qwen_omni_7b", kQwenOmni7bMiB);
	if (newModel == "qwen_omni_3b") newBudgetMiB = iniConfig_.vramBudgetMiB("qwen_omni_3b", kQwenOmni3bMiB);
	else if (newModel == "gemma_e4b") newBudgetMiB = iniConfig_.vramBudgetMiB("gemma_e4b", kGemmaE4bMiB);

	// Verify VRAM available via GpuProfiler before spawning
	if (vramGate_) {
		auto vramResult = vramGate_->acquireVramForModule(
			"conversation_model", newBudgetMiB);
		if (!vramResult) {
			OE_LOG_ERROR("conversation_model_vram_failed: model={}, error={}",
			           newModel, vramResult.error());
			publishError(std::format("Cannot load {} — insufficient GPU memory", newModel));
			return;
		}
	}

	// ── Step 4: Spawn the new conversation model ─────────────────────────
	if (convIt != moduleIndex_.end()) {
		ModuleDescriptor& convMod = modules_[convIt->second];
		// Update args to specify which model variant to load
		// The binary reads --model flag to select the inferencer
		convMod.args = {"--config", "config/omniedge_config.yaml", "--model", newModel};
		// Reset restart counter — explicit model switch resets the crash budget
		convMod.restartCount = 0;
		auto spawnResult = launcher_.spawnModule(convMod);
		if (!spawnResult) {
			OE_LOG_ERROR("conversation_model_spawn_failed: model={}, error={}",
			           newModel, spawnResult.error());
			publishError(std::format("Failed to start {}", newModel));
			return;
		}
		moduleHeartbeats_["conversation_model"] = std::chrono::steady_clock::now();
		// Track VRAM allocation for the new model
		vramTracker_.markModuleLoaded("conversation_model");
		eventBus_.publish(VramChanged{
			.moduleName = "conversation_model",
			.budgetMiB  = newBudgetMiB,
			.loaded     = true,
		});
	}

	// ── Step 5: Spawn companion TTS if Gemma 4 E4B is selected ───────────
	if (newNeedsTts && !previousNeedsTts) {
		auto ttsIt = moduleIndex_.find("tts");
		if (ttsIt != moduleIndex_.end()) {
			ModuleDescriptor& ttsMod = modules_[ttsIt->second];
			if (!ttsMod.isRunning()) {
				if (vramGate_) {
					const auto ttsBudget = iniConfig_.vramBudgetMiB("tts", kTtsMiB);
					auto ttsVram = vramGate_->acquireVramForModule("tts", ttsBudget);
					if (!ttsVram) {
						OE_LOG_WARN("companion_tts_vram_failed: {}", ttsVram.error());
					}
				}
				auto ttsSpawn = launcher_.spawnModule(ttsMod);
				if (ttsSpawn) {
					vramTracker_.markModuleLoaded("tts");
					eventBus_.publish(VramChanged{
						.moduleName = "tts",
						.budgetMiB  = iniConfig_.vramBudgetMiB("tts", kTtsMiB),
						.loaded     = true,
					});
					moduleHeartbeats_["tts"] = std::chrono::steady_clock::now();
					OE_LOG_INFO("companion_tts_spawned: needed for {}", newModel);
				}
			}
		}
	}

	activeConversationModel_ = newModel;

	// Reset video conversation when switching models — user must re-enable
	if (videoConversationEnabled_) {
		videoConversationEnabled_ = false;
		if (messageRouter_) {
			messageRouter_->publish("video_conversation", nlohmann::json{
				{"v",       kSchemaVersion},
				{"type",    "video_conversation"},
				{"enabled", false},
			});
		}
		OE_LOG_INFO("video_conversation_disabled: model switch reset");
	}

	const bool newSupportsVision = newFlags != conversationModelFlags_.end() && newFlags->second.supportsVision;

	eventBus_.publish(ConversationModelChanged{
		.previousModel = previousModel,
		.newModel      = newModel,
		.needsTts      = newNeedsTts,
	});

	// Publish model change to frontend (includes vision capability)
	if (messageRouter_) {
		messageRouter_->publish("module_status", nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "conversation_model_changed"},
			{"model",  newModel},
			{"needs_tts", newNeedsTts},
			{"supports_vision", newSupportsVision},
		});
	}

	OE_LOG_INFO("conversation_model_active: model={}, needs_tts={}, vision={}, vram_mib={}",
	          newModel, newNeedsTts, newSupportsVision, newBudgetMiB);
}

// == handleToggleVideoConversation =============================================
// Toggles video conversation mode: when enabled, the conversation model
// receives the latest video frame from /oe.vid.ingest SHM alongside each
// text/audio prompt.  Only available when the active conversation model
// supports native vision input (Gemma 4 E4B, Qwen2.5-Omni).
//
// When Gemma 4 E4B is the active model, enabling video_conversation also
// ensures the companion TTS sidecar is running (Gemma text-only output).

void OmniEdgeDaemon::handleToggleVideoConversation(const nlohmann::json& cmd)
{
	const bool enabled = cmd.value("enabled", false);
	if (enabled == videoConversationEnabled_) return;

	// Verify the active conversation model supports vision
	auto flagsIt = conversationModelFlags_.find(activeConversationModel_);
	if (enabled && (flagsIt == conversationModelFlags_.end() || !flagsIt->second.supportsVision)) {
		OE_LOG_WARN("video_conversation_rejected: model {} does not support vision",
		          activeConversationModel_);
		publishError(std::format("Video conversation not available — {} does not support video input",
		           activeConversationModel_));
		return;
	}

	videoConversationEnabled_ = enabled;

	// Publish toggle to the conversation node
	if (messageRouter_) {
		messageRouter_->publish("video_conversation", nlohmann::json{
			{"v",       kSchemaVersion},
			{"type",    "video_conversation"},
			{"enabled", enabled},
		});
	}

	// Ensure TTS sidecar is running if Gemma 4 E4B is active and video conversation is enabled
	if (enabled && flagsIt != conversationModelFlags_.end() && flagsIt->second.needsTts) {
		auto ttsIt = moduleIndex_.find("tts");
		if (ttsIt != moduleIndex_.end()) {
			ModuleDescriptor& ttsMod = modules_[ttsIt->second];
			if (!ttsMod.isRunning()) {
				if (vramGate_) {
					const auto ttsBudget = iniConfig_.vramBudgetMiB("tts", kTtsMiB);
					(void)vramGate_->acquireVramForModule("tts", ttsBudget);
				}
				auto ttsSpawn = launcher_.spawnModule(ttsMod);
				if (ttsSpawn) {
					vramTracker_.markModuleLoaded("tts");
					moduleHeartbeats_["tts"] = std::chrono::steady_clock::now();
					OE_LOG_INFO("companion_tts_spawned: needed for video_conversation with {}",
					           activeConversationModel_);
				}
			}
		}
	}

	// Notify frontend
	if (messageRouter_) {
		messageRouter_->publish("module_status", nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "video_conversation_changed"},
			{"enabled", videoConversationEnabled_},
			{"model",  activeConversationModel_},
		});
	}

	OE_LOG_INFO("video_conversation_toggled: enabled={}, model={}",
	          videoConversationEnabled_, activeConversationModel_);
}

void OmniEdgeDaemon::handleSelectTtsModel(const nlohmann::json& cmd)
{
	// TTS model selection is only available when Gemma 4 E4B is active
	if (activeConversationModel_ != "gemma_e4b") {
		OE_LOG_WARN("tts_model_select_ignored: conversation model {} handles TTS natively",
		          activeConversationModel_);
		return;
	}

	const std::string ttsModel = cmd.value("model", "");
	if (ttsModel.empty()) return;

	OE_LOG_INFO("tts_model_selected: {}", ttsModel);

	// Forward TTS model selection to the TTS process via ZMQ
	if (messageRouter_) {
		messageRouter_->publish("tts_config", nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "tts_config"},
			{"model",  ttsModel},
		});
	}
}

void OmniEdgeDaemon::handleUploadImage(const nlohmann::json& cmd)
{
	if (!modeOrchestrator_) {
		publishError("Mode orchestrator not initialized");
		return;
	}

	// Auto-switch to image_transform mode if not already in it.
	// Call switchToMode() directly instead of recursive executeUiCommand().
	if (modeOrchestrator_->currentMode() != "image_transform") {
		OE_LOG_INFO("upload_image_auto_switch: from {} to image_transform",
		          modeOrchestrator_->currentMode());
		if (!switchToMode("image_transform")) {
			publishError("Failed to switch to Image Transform mode");
			return;
		}
	}

	OE_LOG_INFO("image_uploaded: forwarding to image_transform module");

	// Forward the upload data to the Stable Diffusion module via ZMQ
	if (messageRouter_) {
		messageRouter_->publish("image_upload", nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "image_upload"},
			{"data",   cmd.value("data", "")},
			{"width",  cmd.value("width", 0)},
			{"height", cmd.value("height", 0)},
		});
	}
}

void OmniEdgeDaemon::handleSelectImageStyle(const nlohmann::json& cmd)
{
	activeImageStyle_ = cmd.value("style", "anime");
	OE_LOG_INFO("image_style_selected: {}", activeImageStyle_);
}

void OmniEdgeDaemon::handleStartImageTransform(const nlohmann::json& /*cmd*/)
{
	if (!modeOrchestrator_) {
		publishError("Mode orchestrator not initialized");
		return;
	}

	// Auto-switch to image_transform mode if not already in it.
	// Call switchToMode() directly instead of recursive executeUiCommand().
	if (modeOrchestrator_->currentMode() != "image_transform") {
		OE_LOG_INFO("image_transform_auto_switch: from {} to image_transform",
		          modeOrchestrator_->currentMode());
		if (!switchToMode("image_transform")) {
			publishError("Failed to switch to Image Transform mode");
			return;
		}
	}

	OE_LOG_INFO("image_transform_triggered: style={}", activeImageStyle_);

	// Set timeout BEFORE firing FSM event — prevents stale deadline from
	// a previous transform causing immediate timeout on next tickPeriodicTasks().
	imageTransformTimeoutDeadline_ = std::chrono::steady_clock::now() +
	                                  std::chrono::seconds{kImageTransformTimeoutS};

	// Transition FSM to ImageTransform processing state
	(void)applyFsmEvent(fsm::ImageTransformRequest{});

	// Send the style transfer request to the Stable Diffusion module
	if (messageRouter_) {
		messageRouter_->publish("transform_request", nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "transform_request"},
			{"style",  activeImageStyle_},
		});
	}
}

void OmniEdgeDaemon::fsmOnImageTransformComplete()
{
	OE_LOG_INFO("image_transform_complete: style={}", activeImageStyle_);
	publishFsmState(StateIndex::kIdle);
}

void OmniEdgeDaemon::fsmOnImageTransformTimeout()
{
	OE_LOG_ERROR("image_transform_timeout: style={}", activeImageStyle_);
	publishError("Image transform timed out — Stable Diffusion inference took too long");
	publishFsmState(StateIndex::kIdle);
}

// == Config validation ========================================================

tl::expected<OmniEdgeDaemon::Config, std::string>
OmniEdgeDaemon::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("pubPort", raw.pubPort);
	v.requireRange("watchdogPollMs", raw.watchdogPollMs, 100, 60000);
	v.requirePositive("vadSilenceThresholdMs", raw.vadSilenceThresholdMs);
	v.requirePositive("llmGenerationTimeoutS", raw.llmGenerationTimeoutS);
	v.requirePositive("heartbeatIvlMs", raw.heartbeatIvlMs);
	v.requirePositive("heartbeatTimeoutMs", raw.heartbeatTimeoutMs);
	v.requirePositive("heartbeatTtlMs", raw.heartbeatTtlMs);
	v.requirePositive("moduleReadyTimeoutS", raw.moduleReadyTimeoutS);
	v.requireRange("gpuDeviceId", raw.gpuDeviceId, 0, 15);
	if (auto err = v.finish(); !err.empty())
		return tl::unexpected(err);
	return raw;
}

