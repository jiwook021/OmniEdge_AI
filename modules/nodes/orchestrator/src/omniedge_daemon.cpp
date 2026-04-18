#include "orchestrator/omniedge_daemon.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
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
#include "common/constants/conversation_constants.hpp"
#include "zmq/heartbeat_constants.hpp"
#include "zmq/zmq_endpoint.hpp"


// == Construction / Destruction ================================================

OmniEdgeDaemon::OmniEdgeDaemon(Config config)
	: config_(std::move(config))
{
}

OmniEdgeDaemon::~OmniEdgeDaemon()
{
	stop();
	screenCapture_.stop();
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
	// Five-Mode Architecture: conversation_model is registered with the
	// default Gemma-4 E4B budget.  On model switch, the daemon
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
			// Conversation mode — unified model (default: Gemma-4 E4B)
			ModuleVramDef{"conversation_model",  iniConfig_.vramBudgetMiB("gemma_e4b",         kGemmaE4bMiB)},
			ModuleVramDef{"tts",                 iniConfig_.vramBudgetMiB("tts",                kTtsMiB)},
			ModuleVramDef{"audio_denoise",       iniConfig_.vramBudgetMiB("audio_denoise",      kDtlnMiB)},
			// SAM2 Segmentation mode
			ModuleVramDef{"sam2",                iniConfig_.vramBudgetMiB("sam2",               kSam2MiB)},
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

	// ── Mode catalog — five mutually exclusive GPU inference modes ─────────
	//
	// Always-on modules (video_ingest, audio_ingest, background_blur,
	// websocket_bridge) are listed in EVERY mode so the orchestrator never
	// evicts them during a mode switch. Per-module VRAM budgets live in
	// [vram_budgets] in omniedge.ini and are consumed by VramGate/executeSwitchPlan.
	//
	// Mode 1 — Conversation (default): Gemma-4 (E4B default, E2B lightweight)
	//   handles STT+LLM natively and pairs with a Kokoro TTS sidecar for
	//   speech output.  Optional audio_denoise toggle.
	// Mode 2 — SAM2 Segmentation: interactive segmentation (no audio).
	// Mode 3 — Vision Model: VLM (vision+audio+text) using the conversation_model slot.
	// Mode 4 — Security: object detection and event logging.
	// Mode 5 — Beauty: face-mesh + landmark-driven effects.
	modes_ = {
		{"conversation",        {"video_ingest", "audio_ingest", "background_blur",
		                         "websocket_bridge", "conversation_model"}},
		{"sam2_segmentation",   {"video_ingest", "background_blur",
		                         "websocket_bridge", "sam2"}},
		{"vision_model",        {"video_ingest", "audio_ingest",
		                         "websocket_bridge", "conversation_model"}},
		{"security",            {"video_ingest", "audio_ingest", "background_blur",
		                         "websocket_bridge", "security_camera", "security_vlm"}},
		{"beauty",              {"video_ingest", "audio_ingest", "background_blur",
		                         "websocket_bridge", "beauty"}},
	};

	// Graph-based pipeline orchestrator — mode controlled by [orchestrator_graph] INI section.
	{
		const auto& og = iniConfig_.orchestratorGraph();
		core::pipeline_orchestrator::OrchestratorConfig orchConfig{
			.shadowMode    = og.shadowMode,
			.vramBudgetMiB = kVramBudgetMiB,
			.switchPlan    = {
				.drainTimeoutMs        = og.drainTimeoutMs,
				.connectRetryCount     = og.connectRetryCount,
				.connectRetryIntervalMs = og.connectRetryIntervalMs,
			},
			.crashHandler  = {
				.maxCrashesBeforeDisable = static_cast<std::size_t>(
					iniConfig_.crashProtection().maxSegfaultRestarts),
			},
		};
		pipelineOrchestrator_ =
			std::make_unique<core::pipeline_orchestrator::PipelineOrchestrator>(orchConfig);
	}

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
	// Conversation model subscriptions
	messageRouter_->subscribe(kConversationModel, "transcription",  false, handler);
	messageRouter_->subscribe(kConversationModel, kZmqTopicConversationResponse,   false, handler);
	messageRouter_->subscribe(kConversationModel, "audio_output",   false, handler);
	messageRouter_->subscribe(kScreenIngest,      "screen_health",   false, handler);

	// Subscribe to module_ready and heartbeat from every module's PUB port.
	static constexpr int kModulePorts[] = {
		kVideoIngest,       // 5555
		kAudioIngest,       // 5556
		kTts,              // 5565
		kFaceRecog,        // 5566
		kBgBlur,           // 5567
		kAudioDenoise,     // 5569
		kWsBridge,         // 5570
		kConversationModel,// 5572 (Gemma-4 E4B/E2B)
		kScreenIngest,     // 5577 (screen capture)
	};
	for (int port : kModulePorts) {
		messageRouter_->subscribe(port, kZmqTopicModuleReady, false, handler);
		messageRouter_->subscribe(port, kZmqTopicHeartbeat,    false, handler);
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
		// In the five-mode architecture, on-demand modules with priority >= 3
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

	// ── Load INI first (canonical source for daemon timing, screen ingest,
	//    boot modes, launch order). Module-registry parsing lives in YAML.
	{
		std::string iniPath = config_.iniFilePath;
		if (iniPath.empty()) {
			std::filesystem::path configDir =
				std::filesystem::path(config_.configFile).parent_path();
			iniPath = (configDir / "omniedge.ini").string();
		}
		if (!iniConfig_.loadFromFile(iniPath)) {
			OE_LOG_WARN("ini_config_load_failed: path={} — using defaults", iniPath);
		}
	}

	// ── Screen capture agent (Windows DXGI) from INI ──────────────────
	{
		const auto& si = iniConfig_.screenIngest();
		screenCapture_.configure({.exePath = si.screenCaptureExe, .tcpPort = si.tcpPort});
		if (screenCapture_.isConfigured()) {
			OE_LOG_INFO("screen_capture_agent: exe={}, tcp_port={}",
			          si.screenCaptureExe, si.tcpPort);
		}
	}

	YAML::Node root = YAML::LoadFile(config_.configFile);

	// Parse modules in launch order
	auto modulesNode = root["modules"];
	if (!modulesNode) return;

	// Determine launch order from INI: prefer boot_modes[default_mode] over launch_order.
	const auto& dt          = iniConfig_.daemonTiming();
	const auto& defaultMode = dt.defaultMode;
	const auto& iniBootModes = iniConfig_.bootModes().modes;
	std::vector<std::string> order;

	if (auto it = iniBootModes.find(defaultMode); it != iniBootModes.end() && !it->second.empty()) {
		order = it->second;
		std::string joined;
		for (const auto& n : order) { if (!joined.empty()) joined += ", "; joined += n; }
		OE_LOG_INFO("boot_mode: mode={}, modules={}", defaultMode, joined);
	} else if (!dt.launchOrder.empty()) {
		order = dt.launchOrder;
		OE_LOG_INFO("boot_mode: fallback to launch_order (mode '{}' not found in [boot_modes])",
		            defaultMode);
	}

	// Store all boot_modes for runtime switching
	currentBootMode_ = defaultMode;
	for (const auto& [modeName, modeModules] : iniBootModes) {
		bootModes_[modeName] = modeModules;
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
	// SHM paths come from INI (canonical); watchdog uses this to shm_unlink
	// stale /dev/shm/oe.* on module crash.
	{
		// Well-known ingest SHM names (hardcoded — not user-configurable)
		shmRegistry_.registerSegment("video_ingest", "/oe.vid.ingest");
		shmRegistry_.registerSegment("audio_ingest", "/oe.aud.ingest");

		shmRegistry_.registerSegment("tts",             iniConfig_.tts().shmOutput);
		shmRegistry_.registerSegment("background_blur", iniConfig_.backgroundBlur().shmOutput);
		shmRegistry_.registerSegment("audio_denoise",   iniConfig_.audioDenoise().shmOutput);

		OE_LOG_INFO("shm_registry: {} modules with registered SHM segments",
		            shmRegistry_.moduleCount());
	}

	// ── INI Config: filter disabled modules + apply timing overrides ────
	{
		if (iniConfig_.loaded()) {
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
			config_.restartBaseBackoffMs  = dt.restartBaseBackoffMs;
			config_.restartMaxBackoffMs   = dt.restartMaxBackoffMs;
			config_.bargeInEnabled        = dt.bargeInEnabled;

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
	// Also resolve model_dir to absolute paths for pre-spawn validation.
	if (auto convModels = root["conversation_models"]; convModels && convModels.IsMap()) {
		const std::string rawModelsRoot = root["models_root"].as<std::string>("");
		const auto resolveModelPath = makeModelPathResolver(rawModelsRoot);

		for (auto it = convModels.begin(); it != convModels.end(); ++it) {
			const std::string modelName = it->first.as<std::string>();
			const bool nativeTts = it->second["native_tts"].as<bool>(true);
			const std::string rawDir = it->second["model_dir"].as<std::string>("");
			const std::string resolvedDir = resolveModelPath(rawDir);
			const bool nativeVision = it->second["native_vision"].as<bool>(false);
			const std::string fallback = it->second["fallback"].as<std::string>("");
			const std::size_t vramBudgetMiB = it->second["vram_budget_mib"].as<std::size_t>(0);
			conversationModelFlags_[modelName] = {
				.needsTts = !nativeTts,
				.supportsVision = nativeVision,
				.resolvedModelDir = resolvedDir,
				.fallback = fallback,
				.vramBudgetMiB = vramBudgetMiB,
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
// ── FSM Transition Table ─────────────────────────────────────────────────
//
// 10 states, 35 numbered transitions (T1–T35).  Each operator() overload
// is one arrow in the diagram below.  CoverageProbe (below the struct)
// mirrors every entry for boot-time dead-state detection.
//
// Missing (state, event) pairs fall through to the catch-all → std::nullopt
// and are logged at WARN level by StateMachine::dispatch().
//
//                         ┌──────────────────────────────────────────────────┐
//                         │                   IDLE                          │
//                         │  (entry state after boot / error recovery)      │
//                         └───┬────┬────┬────┬────┬────┬────┬────┬────┬────┘
//                  T1 PttPress│    │T2  │T3  │T4  │T23 │T25 │T27 │T29 │T31
//                             ▼    │    │    │    │    │    │    │    │
//                        LISTENING  │    │    │    │    │    │    │    │
//                T5/T6 PttRelease│  │    │    │    │    │    │    │    │
//                  T7 VadSilence │  │    │    │    │    │    │    │    │
//                     T8 PttCancel│ │    │    │    │    │    │    │    │
//                     T9 Crash   │  │    │    │    │    │    │    │    │
//                             ▼   │    │    │    │    │    │    │    │
//                        PROCESSING │    │    │    │    │    │    │    │
//               T10 LlmFirstSent│  │    │    │    │    │    │    │    │
//                T11 LlmComplete│   │    │    │    │    │    │    │    │
//                 T12 LlmTimeout│   │    │    │    │    │    │    │    │
//                    T13 Crash  │   │    │    │    │    │    │    │    │
//                             ▼    │    │    │    │    │    │    │    │
//                         SPEAKING  │    │    │    │    │    │    │    │
//               T17/T18 PttPress│   │    │    │    │    │    │    │    │
//                 T19 TtsComplete│  │    │    │    │    │    │    │    │
//                    T20 Crash   │  │    │    │    │    │    │    │    │
//                             ▼   │    │    │    │    │    │    │    │
//                       INTERRUPTED │    │    │    │    │    │    │    │
//              T21 InterruptDone│   │    │    │    │    │    │    │    │
//                               ▼   │    │    │    │    │    │    │    │
//                      ERROR_RECOVERY│    │    │    │    │    │    │    │
//              T22 ModuleRecovered   │    │    │    │    │    │    │    │
//                                    │    │    │    │    │    │    │    │
//     Mode-switch transitions:       │    │    │    │    │    │    │    │
//       T27 Sam2SegmentationStart ───────────────────────┘    │    │    │
//       T28 Sam2SegmentationStop (mode error) ────────────────┘    │    │
//       T29 VisionModelStart ─────────────────────────────────────┘    │
//       T30 VisionModelStop (mode error) ──────────────────────────────┘
//
//       T32 Sam2Segmentation + Sam2SegmentationStop → IDLE
//       T33 Sam2Segmentation + ModuleCrash → ERROR_RECOVERY
//       T34 VisionModel + VisionModelStop → IDLE
//       T35 VisionModel + ModuleCrash → ERROR_RECOVERY
//
// Transition numbers T14–T16 are not used (historical gaps from removed
// states).  New transitions should continue from T36.
// ─────────────────────────────────────────────────────────────────────────

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
		d->fsmOnModuleCrash("conversation_model");
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

	// ── Five-Mode Architecture transitions ──────────────────────────────

	// T32: SAM2_SEGMENTATION → IDLE (user left SAM2 mode via mode switch)
	std::optional<State> operator()(fsm::Sam2Segmentation, fsm::Sam2SegmentationStop) const {
		d->publishFsmState(StateIndex::kIdle);
		return fsm::Idle{};
	}

	// T33: SAM2_SEGMENTATION → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::Sam2Segmentation, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("sam2");
		return fsm::ErrorRecovery{};
	}

	// T34: VISION_MODEL → IDLE (user left vision mode via mode switch)
	std::optional<State> operator()(fsm::VisionModel, fsm::VisionModelStop) const {
		d->publishFsmState(StateIndex::kIdle);
		return fsm::Idle{};
	}

	// T35: VISION_MODEL → ERROR_RECOVERY (module crash)
	std::optional<State> operator()(fsm::VisionModel, fsm::ModuleCrash) const {
		d->fsmOnModuleCrash("vision_model");
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
	// T32
	std::optional<StateIndex> operator()(fsm::Sam2Segmentation, fsm::Sam2SegmentationStop) const { return StateIndex::kIdle; }
	// T33
	std::optional<StateIndex> operator()(fsm::Sam2Segmentation, fsm::ModuleCrash) const  { return StateIndex::kErrorRecovery; }
	// T34
	std::optional<StateIndex> operator()(fsm::VisionModel, fsm::VisionModelStop) const   { return StateIndex::kIdle; }
	// T35
	std::optional<StateIndex> operator()(fsm::VisionModel, fsm::ModuleCrash) const       { return StateIndex::kErrorRecovery; }

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
	if (type == kZmqTopicConversationResponse) { processLlmTokenStream(msg); return; }

	// ── Five-Mode Architecture messages ─────────────────────────────────
	// Kokoro TTS sidecar publishes audio_output on behalf of Gemma-4 variants.
	// The bridge forwards it to the /audio WebSocket channel.
	if (type == "audio_output") {
		if (messageRouter_) messageRouter_->publish("audio_output", msg);
		return;
	}
	// Screen health — relay to WS bridge for frontend error banner.
	if (type == "screen_health") {
		if (messageRouter_) messageRouter_->publish("screen_health", msg);
		return;
	}

	if (type == kZmqTopicHeartbeat && msg.contains("module")) {
		const auto modName = msg["module"].get<std::string>();
		// Only accept heartbeats from known modules to prevent unbounded map growth.
		if (moduleIndex_.contains(modName)) {
			moduleHeartbeats_[modName] = std::chrono::steady_clock::now();
		}
		return;
	}

	if (type == kZmqTopicModuleReady && msg.contains("module")) {
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

	/* GET /graph introspection — returns pipeline graph as JSON */
	if (type == "get_graph" && pipelineOrchestrator_) {
		auto graphJson = pipelineOrchestrator_->introspectAsJson();
		if (messageRouter_) {
			messageRouter_->publish("graph_state", nlohmann::json::parse(graphJson));
		}
		return;
	}

}

// == LLM availability check ===================================================

bool OmniEdgeDaemon::isLlmAvailable() const
{
	// Check if conversation_model is running
	auto checkRunning = [&](const std::string& name) -> bool {
		auto it = moduleIndex_.find(name);
		return it != moduleIndex_.end() && modules_[it->second].isRunning();
	};
	return checkRunning("conversation_model");
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
		// Verify conversation model is running
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
		// Delegates to switchToMode() which handles profile switching, module
		// lifecycle, FSM state, and session persistence in one place.
		std::string requestedMode = cmd.value("mode", "");
		(void)switchToMode(requestedMode);
		break;
	}

	// ── Five-Mode Architecture commands ──────────────────────────────────
	case UiAction::kSelectConversationModel:
		handleSelectConversationModel(cmd);
		break;
	case UiAction::kSelectTtsModel:
		handleSelectTtsModel(cmd);
		break;
	case UiAction::kToggleVideoConversation:
		handleToggleVideoConversation(cmd);
		break;
	case UiAction::kSelectConversationSource:
		handleSelectConversationSource(cmd);
		break;

	// ── Security Camera mode commands ───────────────────────────────────
	case UiAction::kToggleSecurityMode: {
		bool enabled = cmd.value("enabled", false);
		if (enabled) {
			(void)switchToMode("security");
		} else {
			(void)switchToMode("conversation");
		}
		break;
	}
	case UiAction::kSecurityVlmAnalyze:
	case UiAction::kSecurityListRecordings:
	case UiAction::kSecurityListEvents:
	case UiAction::kSecurityUpdateClasses:
	case UiAction::kSecuritySetStyle:
	case UiAction::kSecuritySetRoi:
		if (messageRouter_) {
			messageRouter_->publish("security_command", cmd);
		}
		break;

	// ── Beauty mode commands ────────────────────────────────────────────
	case UiAction::kToggleBeauty: {
		bool enabled = cmd.value("enabled", false);
		if (enabled) {
			(void)switchToMode("beauty");
		} else {
			(void)switchToMode("conversation");
		}
		break;
	}
	case UiAction::kSetBeautySkin:
	case UiAction::kSetBeautyShape:
	case UiAction::kSetBeautyLight:
	case UiAction::kSetBeautyBg:
	case UiAction::kSetBeautyPreset:
		// Forward beauty commands directly — BeautyNode subscribes to
		// ui_command on the WS bridge port and handles them itself.
		break;

	case UiAction::kSwitchBootMode: {
		std::string targetMode = cmd.value("mode", "");
		handleBootModeSwitch(targetMode);
		break;
	}

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
		// Profile switches happen at mode level, not per individual module
		// toggle.  Module toggles within the current mode do not change the
		// interaction profile.

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

		// ── Profile stays at mode level ──────────────────────────────
		// Interaction profiles are tied to the active GPU mode, not
		// individual module toggles.
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
		upgradeWatchdog();
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

				// Kill the entire process group to catch child processes
				// (e.g., model_generate.py spawned by omniedge_conversation).
				::kill(-mod.pid, SIGKILL);
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

	// ── Phase 2.5a: Kill orphaned child processes ──────────────────────
	// When a module crashes or is killed, child processes (e.g., Python
	// workers) may survive as orphans, holding GPU VRAM and RAM.
	// Send SIGKILL to the process group to clean them up.
	for (const auto& crash : crashes) {
		if (crash.pidWas > 0) {
			::kill(-crash.pidWas, SIGKILL);
		}
	}

	// ── Phase 2.5b: Clean up stale SHM segments ─────────────────────────
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

	// Gate on conversation-model readiness — do not publish until the module
	// has sent module_ready. Publishing before the GemmaInferencer Python
	// subprocess finishes loading weights would drop the prompt silently.
	auto isModuleReady = [&](const std::string& name) -> bool {
		auto it = moduleIndex_.find(name);
		return it != moduleIndex_.end()
		    && modules_[it->second].isRunning()
		    && modules_[it->second].ready;
	};
	if (!isModuleReady("conversation_model")) {
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
	// Gemma-4 variants always pair with the Kokoro TTS sidecar; this helper
	// also covers any future native-TTS conversation model.
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
		{"mode",                currentProfileName_},
		{"conversation_model",  activeConversationModel_},
		{"tier",                gpuProfiler_ ? std::string(tierName(gpuProfiler_->tier())) : "unknown"},
		{"history_size",        promptAssembler_ ? promptAssembler_->historySize() : 0},
		{"modules",             modulesJson},
	};
}

void OmniEdgeDaemon::restoreSessionState(const nlohmann::json& state)
{
	if (state.contains("mode")) {
		currentProfileName_ = state["mode"].get<std::string>();
	}
}

// == Publish helpers ==========================================================

void OmniEdgeDaemon::publishFsmState(StateIndex s)
{
	if (!messageRouter_) return;
	messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
		{"v",      kSchemaVersion},
		{"type",   std::string(kZmqTopicModuleStatus)},
		{"module", "omniedge_orchestrator"},
		{"state",  std::string(stateName(s))},
	});
}

void OmniEdgeDaemon::publishModuleStatus(const std::string& module, const std::string& status)
{
	if (!messageRouter_) return;
	messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
		{"v",      kSchemaVersion},
		{"type",   std::string(kZmqTopicModuleStatus)},
		{"module", module},
		{"status", status},
	});
}

void OmniEdgeDaemon::publishError(const std::string& message)
{
	if (!messageRouter_) return;
	messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
		{"v",       kSchemaVersion},
		{"type",    "error"},
		{"message", message},
	});
}

// == executeSwitchPlan (graph-based module lifecycle) ==============================

tl::expected<std::unordered_set<std::string>, std::string>
OmniEdgeDaemon::executeSwitchPlan(
	const core::pipeline_orchestrator::SwitchPlan& plan)
{
	OE_ZONE_SCOPED;

	/* Phase 1: drain edges — no-op until edges are wired */
	/* Phase 2: evict vertices — sorted by evict priority (lowest first).
	 * With zero edges, topo sort returns arbitrary order, so we re-sort
	 * by priority here. */
	std::vector<std::string> sortedEvictions = plan.verticesToEvict;
	std::ranges::sort(sortedEvictions, [this](const std::string& a, const std::string& b) {
		auto ia = moduleIndex_.find(a);
		auto ib = moduleIndex_.find(b);
		if (ia == moduleIndex_.end() || ib == moduleIndex_.end()) return false;
		return modules_[ia->second].evictPriority < modules_[ib->second].evictPriority;
	});

	for (const auto& vertexId : sortedEvictions) {
		auto idxIt = moduleIndex_.find(vertexId);
		if (idxIt == moduleIndex_.end()) continue;
		auto& desc = modules_[idxIt->second];
		if (!desc.isRunning()) continue;

		OE_LOG_WARN("graph_evict: module={}, priority={}",
		            desc.name, desc.evictPriority);

		const pid_t savedPid = desc.pid;
		launcher_.stopModule(desc);

		/* Zombie process verification gate — WSL2 can leave SIGTERMed PIDs
		 * briefly uninterruptible, so confirm the process is truly dead
		 * before releasing its VRAM reservation. */
		if (savedPid > 0) {
			bool processVerifiedDead = false;
			for (int attempt = 0;
			     attempt < kEvictionVerifyMaxAttempts;
			     ++attempt) {
				if (::kill(savedPid, 0) == -1 && errno == ESRCH) {
					processVerifiedDead = true;
					break;
				}
				std::this_thread::sleep_for(kEvictionVerifyPollInterval);
			}

			if (!processVerifiedDead) {
				OE_LOG_ERROR(
					"graph_evict_zombie: {} (pid={}) refused to die — "
					"VRAM budget may be over-committed",
					desc.name, savedPid);
				desc.pid = savedPid;
				continue;
			}
		}

		vramGate_->notifyModuleEvicted(desc.name);
	}

	/* Phase 3: unlink SHM — no-op until edges are wired */
	/* Phase 4: spawn vertices (topo order from plan) */
	std::unordered_set<std::string> failedModuleNames;
	std::vector<ModuleDescriptor*> spawnedModules;

	for (const auto& vertexId : plan.verticesToSpawn) {
		auto idxIt = moduleIndex_.find(vertexId);
		if (idxIt == moduleIndex_.end()) {
			failedModuleNames.insert(vertexId);
			continue;
		}
		auto& desc = modules_[idxIt->second];
		if (desc.isRunning()) continue;  /* Already running — skip */

		/* Acquire VRAM via VramGate (evict→wait→verify cycle).
		 * Budget comes from INI [vram_budgets] with compile-time fallback. */
		const std::size_t moduleVramBudgetMiB =
			iniConfig_.vramBudgetMiB(desc.name, desc.vramBudgetMb);
		if (moduleVramBudgetMiB > 0) {
			auto vramResult = vramGate_->acquireVramForModule(
				desc.name, moduleVramBudgetMiB);
			if (!vramResult) {
				OE_LOG_ERROR("graph_spawn_vram_failed: module={}, error={}",
				             desc.name, vramResult.error());
				failedModuleNames.insert(desc.name);
				continue;
			}
		}

		auto spawnResult = launcher_.spawnModule(desc);
		if (!spawnResult) {
			OE_LOG_ERROR("graph_spawn_failed: module={}, error={}",
			             desc.name, spawnResult.error());
			failedModuleNames.insert(desc.name);
		} else {
			spawnedModules.push_back(&desc);
		}
	}

	/* Wait for readiness of all spawned modules */
	if (!spawnedModules.empty()) {
		auto notReady = launcher_.waitForReady(
			spawnedModules, messageRouter_->context());
		for (const auto& name : notReady) {
			failedModuleNames.insert(name);
		}
	}

	/* Phase 5: connect edges — no-op until edges are wired */

	OE_LOG_INFO("graph_switch_complete: evicted={}, spawned={}, failed={}",
	            plan.verticesToEvict.size(), plan.verticesToSpawn.size(),
	            failedModuleNames.size());
	return failedModuleNames;
}

// == switchToMode (non-recursive mode switch) ====================================

const ModeDefinition* OmniEdgeDaemon::findMode(const std::string& name) const noexcept
{
	for (const auto& mode : modes_) {
		if (mode.name == name) return &mode;
	}
	return nullptr;
}

bool OmniEdgeDaemon::switchToMode(const std::string& requestedMode)
{
	if (!vramGate_ || requestedMode.empty()) return false;

	// Apply interaction profile for the target mode
	if (requestedMode == "conversation") {
		applyInteractionProfile(InteractionProfile::kConversation);
	} else if (requestedMode == "sam2_segmentation") {
		applyInteractionProfile(InteractionProfile::kSam2Segmentation);
	} else if (requestedMode == "vision_model") {
		applyInteractionProfile(InteractionProfile::kVisionModel);
	} else if (requestedMode == "security") {
		applyInteractionProfile(InteractionProfile::kSecurity);
	} else if (requestedMode == "beauty") {
		applyInteractionProfile(InteractionProfile::kBeauty);
	}

	{
		/* ── Graph-based switch — the only path. ──────────────────── */
		const auto* modeDef = findMode(requestedMode);
		if (!modeDef) {
			OE_LOG_ERROR("mode_switch_failed: unknown mode '{}'", requestedMode);
			eventBus_.publish(ModeChanged{requestedMode, false});
			return false;
		}

		auto graphResult = GraphBuilder::buildForMode(*modeDef, modules_);
		if (!graphResult) {
			OE_LOG_ERROR("mode_switch_graph_failed: {}", graphResult.error());
			eventBus_.publish(ModeChanged{requestedMode, false});
			return false;
		}

		auto planResult = pipelineOrchestrator_->switchToProfile(*graphResult);
		if (!planResult) {
			if (planResult.error() == core::pipeline_orchestrator::OrchestratorError::kAlreadyAtTarget) {
				OE_LOG_INFO("mode_switch_noop: already at '{}'", requestedMode);
				/* Fall through to success — no plan to execute. */
			} else {
				OE_LOG_ERROR("mode_switch_plan_failed: mode={}", requestedMode);
				eventBus_.publish(ModeChanged{requestedMode, false});
				return false;
			}
		} else if (!planResult->isEmpty()) {
			auto execResult = executeSwitchPlan(*planResult);
			if (!execResult) {
				OE_LOG_ERROR("mode_switch_exec_failed: {}", execResult.error());
				eventBus_.publish(ModeChanged{requestedMode, false});
				return false;
			}
		}

		currentProfileName_ = requestedMode;
	}

	eventBus_.publish(ModeChanged{requestedMode, true});
	if (requestedMode == "sam2_segmentation") {
		stateMachine_.forceState(StateIndex::kSam2Segmentation);
		publishFsmState(StateIndex::kSam2Segmentation);
	} else if (requestedMode == "vision_model") {
		stateMachine_.forceState(StateIndex::kVisionModel);
		publishFsmState(StateIndex::kVisionModel);
	} else {
		stateMachine_.forceState(StateIndex::kIdle);
		publishFsmState(StateIndex::kIdle);
	}
	if (sessionPersistence_) {
		(void)sessionPersistence_->save(buildSessionState());
	}
	return true;
}

// == Boot Mode Switch (simple_llm ↔ full_conversation) ========================

void OmniEdgeDaemon::handleBootModeSwitch(const std::string& targetMode)
{
	if (targetMode == currentBootMode_) {
		OE_LOG_INFO("boot_mode_switch: already in mode={}", targetMode);
		return;
	}

	auto it = bootModes_.find(targetMode);
	if (it == bootModes_.end()) {
		OE_LOG_ERROR("boot_mode_switch_failed: unknown mode={}", targetMode);
		return;
	}

	OE_LOG_INFO("boot_mode_switch: {} → {}", currentBootMode_, targetMode);
	const auto& targetModules = it->second;

	// Build set of modules required by the target mode
	std::unordered_set<std::string> targetSet(targetModules.begin(), targetModules.end());

	// Build set of modules in the current mode
	std::unordered_set<std::string> currentSet;
	if (auto cit = bootModes_.find(currentBootMode_); cit != bootModes_.end()) {
		currentSet.insert(cit->second.begin(), cit->second.end());
	}

	// Stop modules that are in the current mode but not in the target mode
	for (auto& mod : modules_) {
		if (currentSet.contains(mod.name) && !targetSet.contains(mod.name) && mod.isRunning()) {
			OE_LOG_INFO("boot_mode_stopping: module={}", mod.name);
			launcher_.stopModule(mod);
		}
	}

	// Spawn modules that are in the target mode but not currently running
	for (const auto& modName : targetModules) {
		auto idx = moduleIndex_.find(modName);
		if (idx == moduleIndex_.end()) continue;
		auto& mod = modules_[idx->second];
		if (!mod.isRunning()) {
			OE_LOG_INFO("boot_mode_spawning: module={}", mod.name);
			auto result = launcher_.spawnModule(mod);
			if (!result) {
				OE_LOG_ERROR("boot_mode_spawn_failed: module={}, error={}", mod.name, result.error());
			}
		}
	}

	currentBootMode_ = targetMode;

	// Publish mode change to frontend
	if (messageRouter_) {
		messageRouter_->publish("boot_mode_changed", {
			{"type", "boot_mode_changed"},
			{"mode", targetMode}
		});
	}

	OE_LOG_INFO("boot_mode_switch_complete: mode={}", targetMode);
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

		// Update the ModuleDescriptor evictPriority for switch-plan sorting
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

// == Auto-Upgrade Watchdog ====================================================

void OmniEdgeDaemon::upgradeWatchdog()
{
	// Nothing to upgrade if not degraded
	if (userRequestedModel_.empty()) return;
	if (activeConversationModel_ == userRequestedModel_) return;
	if (!gpuProfiler_) return;

	auto flagsIt = conversationModelFlags_.find(userRequestedModel_);
	if (flagsIt == conversationModelFlags_.end()) return;

	constexpr std::size_t kUpgradeSafetyMarginMiB = 256;
	const std::size_t targetBudget = flagsIt->second.vramBudgetMiB;
	const std::size_t freeMiB = gpuProfiler_->liveFreeMb();
	const auto now = std::chrono::steady_clock::now();

	if (freeMiB >= targetBudget + kUpgradeSafetyMarginMiB) {
		// Start or continue the 60s hysteresis timer
		if (upgradeEligibleSince_ == std::chrono::steady_clock::time_point{})
			upgradeEligibleSince_ = now;

		if (now - upgradeEligibleSince_ >= std::chrono::seconds{60}) {
			OE_LOG_INFO("auto_upgrade: attempting {} (sustained 60s free VRAM, free={} MiB, need={} MiB)",
			          userRequestedModel_, freeMiB, targetBudget);
			const std::string target = userRequestedModel_;
			const std::string previousModel = activeConversationModel_;
			handleSelectConversationModel({{"model", target}});
			upgradeEligibleSince_ = {};

			if (activeConversationModel_ == target) {
				eventBus_.publish(ConversationModelUpgraded{
					.fromModel = previousModel,
					.toModel   = target,
				});
				publishModuleStatus("conversation_model", "upgraded");
			}
		}
	} else {
		upgradeEligibleSince_ = {};  // Reset — VRAM dropped below threshold
	}
}

// == Five-Mode Architecture Handlers ==========================================
//
// The system operates in five mutually exclusive GPU inference modes:
//   1. Conversation (default) — Gemma-4 E4B (default) / Gemma-4 E2B
//   2. SAM2 Segmentation — interactive mask generation (on-demand)
//   3. Vision Model — multimodal VLM using the conversation_model slot
//   4. Security — YOLO detection + NVENC recording + on-demand VLM
//   5. Beauty — FaceMesh V2 landmarks + bilateral filter + TPS warp
//
// Always-on: video_ingest, audio_ingest, background_blur, websocket_bridge.
//
// Switching models within Conversation mode follows the same kill-spawn cycle:
// the daemon terminates the previous model process and spawns the newly
// selected one.  Gemma-4 variants accept audio+image natively and pair with
// the Kokoro TTS sidecar for speech output.

void OmniEdgeDaemon::handleSelectConversationModel(const nlohmann::json& cmd)
{
	std::string newModel = cmd.value("model", "");
	if (newModel.empty()) return;

	// Validate model name against parsed conversation_models config
	if (conversationModelFlags_.find(newModel) == conversationModelFlags_.end()) {
		OE_LOG_WARN("invalid_conversation_model: '{}'", newModel);
		publishError(std::format("Unknown conversation model: {}", newModel));
		return;
	}

	// Validate model directory exists before attempting to spawn — prevents
	// spawn-crash-restart loops for models that aren't installed.
	{
		auto dirIt = conversationModelFlags_.find(newModel);
		if (dirIt != conversationModelFlags_.end() && !dirIt->second.resolvedModelDir.empty()) {
			if (!std::filesystem::exists(dirIt->second.resolvedModelDir)) {
				OE_LOG_ERROR("conversation_model_dir_missing: model={}, dir={}",
				           newModel, dirIt->second.resolvedModelDir);
				publishError(std::format("{} model not installed — directory not found: {}",
				           newModel, dirIt->second.resolvedModelDir));
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
	bool newNeedsTts = newFlags != conversationModelFlags_.end() && newFlags->second.needsTts;

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

	// ── Step 3: Acquire VRAM, with auto-downgrade fallback chain ─────────
	const std::string originalRequest = newModel;
	std::size_t newBudgetMiB = conversationModelFlags_.at(newModel).vramBudgetMiB;

	if (vramGate_) {
		auto vramResult = vramGate_->acquireVramForModule(
			"conversation_model", newBudgetMiB);

		// Walk the fallback chain if primary model doesn't fit
		if (!vramResult) {
			OE_LOG_WARN("conversation_model_vram_failed: model={}, error={} — trying fallback chain",
			          newModel, vramResult.error());

			std::string candidate = conversationModelFlags_.at(newModel).fallback;
			while (!candidate.empty()) {
				newBudgetMiB = conversationModelFlags_.at(candidate).vramBudgetMiB;
				vramResult = vramGate_->acquireVramForModule(
					"conversation_model", newBudgetMiB);
				if (vramResult) {
					OE_LOG_WARN("auto_downgrade: requested={}, actual={}", originalRequest, candidate);
					newModel = candidate;
					break;
				}
				candidate = conversationModelFlags_.at(candidate).fallback;
			}

			if (!vramResult) {
				OE_LOG_ERROR("conversation_model_vram_exhausted: no model in fallback chain fits");
				publishError(std::format(
					"Cannot load {} or any fallback — insufficient GPU memory", originalRequest));
				return;
			}

			// Re-lookup flags for the downgraded model (needsTts may differ)
			newFlags  = conversationModelFlags_.find(newModel);
			newNeedsTts = newFlags != conversationModelFlags_.end() && newFlags->second.needsTts;
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
		// Note: vramGate_->acquireVramForModule() already called markModuleLoaded()
		// internally — do NOT call vramTracker_.markModuleLoaded() again here.
		eventBus_.publish(VramChanged{
			.moduleName = "conversation_model",
			.budgetMiB  = newBudgetMiB,
			.loaded     = true,
		});

		// Track auto-downgrade state for upgrade watchdog
		userRequestedModel_ = originalRequest;
		if (newModel != originalRequest) {
			eventBus_.publish(ConversationModelDegraded{
				.requestedModel = originalRequest,
				.actualModel    = newModel,
				.reason         = "vram_insufficient",
			});
			publishModuleStatus("conversation_model", "degraded");
			OE_LOG_WARN("conversation_model_degraded: user wanted {}, running {}",
			          originalRequest, newModel);
		} else if (!userRequestedModel_.empty() &&
		           userRequestedModel_ == activeConversationModel_) {
			// User got what they wanted — clear prior degradation
			upgradeEligibleSince_ = {};
		}
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
					// Note: vramGate_->acquireVramForModule() already called markModuleLoaded()
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
		messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
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
// supports native vision input (Gemma-4 E4B, Gemma-4 E2B).
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
		messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "video_conversation_changed"},
			{"enabled", videoConversationEnabled_},
			{"model",  activeConversationModel_},
		});
	}

	OE_LOG_INFO("video_conversation_toggled: enabled={}, model={}",
	          videoConversationEnabled_, activeConversationModel_);
}

// == handleSelectConversationSource ============================================
// Switches the conversation model's video frame source between camera
// (/oe.vid.ingest) and screen (/oe.screen.ingest).  No process spawn/kill —
// ScreenIngestNode is always-on (CPU-only, 0 VRAM).  This only relays the
// source selection to ConversationNode via the video_conversation ZMQ topic.

void OmniEdgeDaemon::handleSelectConversationSource(const nlohmann::json& cmd)
{
	const auto source = cmd.value("source", "camera");
	if (source != "camera" && source != "screen") {
		OE_LOG_WARN("invalid_conversation_source: {}", source);
		return;
	}

	// Launch or kill the Windows screen capture agent as needed.
	if (source == "screen") {
		screenCapture_.start();
	} else {
		screenCapture_.stop();
	}

	// Enable video conversation with the selected source.
	videoConversationEnabled_ = true;

	if (messageRouter_) {
		messageRouter_->publish("video_conversation", nlohmann::json{
			{"v",       kSchemaVersion},
			{"type",    "video_conversation"},
			{"enabled", true},
			{"source",  source},
		});
	}

	// Notify frontend of the source change.
	if (messageRouter_) {
		messageRouter_->publish(kZmqTopicModuleStatus, nlohmann::json{
			{"v",      kSchemaVersion},
			{"type",   "conversation_source_changed"},
			{"source", source},
			{"model",  activeConversationModel_},
		});
	}

	OE_LOG_INFO("conversation_source_selected: source={}, model={}",
	          source, activeConversationModel_);
}

void OmniEdgeDaemon::handleSelectTtsModel(const nlohmann::json& cmd)
{
	// TTS model selection is only meaningful when the active conversation
	// model lacks native TTS (i.e. needs the Kokoro sidecar). Gemma-4 (both
	// E2B and E4B) has no native TTS, so the sidecar is always in play; if a
	// future model has native TTS, it sets `needs_tts=false` in the flags map
	// and we skip the forward here.
	auto flagsIt = conversationModelFlags_.find(activeConversationModel_);
	if (flagsIt != conversationModelFlags_.end() && !flagsIt->second.needsTts) {
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

