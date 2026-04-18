#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/event_bus.hpp"
#include "common/ini_config.hpp"
#include "common/runtime_defaults.hpp"
#include "zmq/port_settings.hpp"
#include "vram/vram_thresholds.hpp"
#include "common/validated_config.hpp"
#include "gpu/gpu_profiler.hpp"
#include "zmq/message_router.hpp"
#include "vram/vram_gate.hpp"
#include "vram/vram_tracker.hpp"
#include "vram/priority_scheduler.hpp"
#include "orchestrator/graph_builder.hpp"
#include "orchestrator/mode_definition.hpp"
#include "orchestrator/module_launcher.hpp"
#include "orchestrator/shm_registry.hpp"
#include "pipeline_orchestrator/pipeline_orchestrator.hpp"
#include "statemachine/prompt_assembler.hpp"
#include "statemachine/session_persistence.hpp"
#include "statemachine/state_machine.hpp"
#include "common/ui_action.hpp"
#include "orchestrator/screen_capture_agent.hpp"


/// Master orchestrator — spawns, monitors, and coordinates all modules.
///
/// Single-threaded event loop driven by zmq_poll().
/// stop() is the only method safe to call from a signal handler.
class OmniEdgeDaemon {
public:
	struct Config {
		std::string configFile;
		int         watchdogPollMs{kWatchdogPollMs};
		int         pubPort{kDaemon};
		std::vector<std::string> subEndpoints;

		// State machine
		int  vadSilenceThresholdMs{static_cast<int>(kVadSilenceDurationMs)};
		bool bargeInEnabled{true};
		int  llmGenerationTimeoutS{kLlmGenerationTimeoutS};

		// Heartbeat
		int heartbeatIvlMs{kHeartbeatIntervalMs};
		int heartbeatTimeoutMs{kHeartbeatTimeoutMs};
		int heartbeatTtlMs{kHeartbeatTtlMs};

		int moduleReadyTimeoutS{kModuleReadyTimeoutS};

		// Restart backoff
		int restartBaseBackoffMs{static_cast<int>(kRestartBaseBackoffMs)};
		int restartMaxBackoffMs{static_cast<int>(kRestartMaxBackoffMs)};

		// GPU
		int         gpuDeviceId{0};
		std::string gpuOverrideProfile;
		std::size_t gpuHeadroomMb{kHeadroomMiB};

		// Session
		std::string sessionFilePath{"session_state.json"};

		// INI config file (empty = config/omniedge.ini relative to config dir)
		std::string iniFilePath;

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	explicit OmniEdgeDaemon(Config config);
	OmniEdgeDaemon(const OmniEdgeDaemon&)            = delete;
	OmniEdgeDaemon& operator=(const OmniEdgeDaemon&) = delete;
	~OmniEdgeDaemon();

	/// Parse YAML, probe GPU, set up ZMQ, register state machine transitions, launch modules.
	void initialize();

	/// Blocking poll loop — returns when stop() is called or SIGTERM.
	void run();

	/// Signal run() to exit. Thread-safe (atomic).
	void stop() noexcept;

	// Accessors (tests + debugging)
	[[nodiscard]] StateIndex currentState() const noexcept { return stateMachine_.currentState(); }
	[[nodiscard]] const std::vector<ModuleDescriptor>& modules() const noexcept { return modules_; }
	[[nodiscard]] StateMachine& stateMachine() noexcept { return stateMachine_; }
	[[nodiscard]] PromptAssembler& promptAssembler() noexcept { return *promptAssembler_; }
	[[nodiscard]] EventBus& eventBus() noexcept { return eventBus_; }
	[[nodiscard]] const std::string& currentMode() const noexcept { return currentProfileName_; }

private:
	Config config_;

	// Networking — owns ZMQ context, PUB socket, all SUB sockets, poll loop.
	std::unique_ptr<MessageRouter> messageRouter_;

	// In-process event bus — decouples cross-component notifications
	EventBus                              eventBus_;

	// Sub-systems
	StateMachine                          stateMachine_;
	std::unique_ptr<PromptAssembler>      promptAssembler_;
	std::unique_ptr<GpuProfiler>          gpuProfiler_;
	VramTracker                           vramTracker_;
	PriorityScheduler                     priorityScheduler_;
	std::unique_ptr<VramGate>             vramGate_;
	std::unique_ptr<core::pipeline_orchestrator::PipelineOrchestrator> pipelineOrchestrator_;
	std::unique_ptr<SessionPersistence>   sessionPersistence_;
	ModuleLauncher                        launcher_;
	ShmRegistry                           shmRegistry_;
	IniConfig                             iniConfig_;

	// Module descriptors (from YAML)
	std::vector<ModuleDescriptor> modules_;

	// Interaction mode catalog (populated in the constructor). The daemon
	// uses these to drive switchToMode() through the graph orchestrator.
	std::vector<ModeDefinition> modes_;
	std::string                 currentProfileName_{"conversation"};

	[[nodiscard]] const ModeDefinition* findMode(const std::string& name) const noexcept;

	// Boot mode definitions from YAML boot_modes section (e.g. simple_llm, full_conversation)
	std::unordered_map<std::string, std::vector<std::string>> bootModes_;
	std::string currentBootMode_;

	// Name→index map for O(1) module lookup (built once in loadModuleConfigFromYaml)
	std::unordered_map<std::string, std::size_t> moduleIndex_;

	// Conversation context
	std::string lastTranscription_;
	std::string lastLlmResponse_;
	std::string pendingSentence_;
	std::string lastCrashedModule_;

	// ── Five-Mode Architecture state ────────────────────────────────────
	// Modes: Conversation, SAM2 Segmentation, Vision Model, Security, Beauty
	// Active conversation model: "gemma_e4b" (default), "gemma_e2b"
	std::string activeConversationModel_{"gemma_e4b"};

	// Per-model config flags parsed from conversation_models YAML section.
	// Used by handleSelectConversationModel to decide companion module spawning.
	struct ConversationModelFlags {
		bool needsTts{false};           // !native_tts — needs companion TTS process
		bool supportsVision{false};     // native_vision — accepts video/image frames
		std::string resolvedModelDir;   // absolute path from YAML model_dir + models_root
		std::string fallback;           // next-smaller model name, empty = terminal
		std::size_t vramBudgetMiB{0};   // from YAML vram_budget_mib
	};
	std::unordered_map<std::string, ConversationModelFlags> conversationModelFlags_;
	bool videoConversationEnabled_{false};  ///< True when camera toggle is active

	// ── Auto-downgrade state ───────────────────────────────────────────
	std::string userRequestedModel_;  ///< What the user asked for (may differ from active if degraded)
	std::chrono::steady_clock::time_point upgradeEligibleSince_{};  ///< Hysteresis timer for auto-upgrade

	// ── Windows screen capture agent lifecycle ─────────────────────────
	ScreenCaptureAgent screenCapture_;

	// Deferred state-machine event (avoids reentrant dispatch)
	std::optional<AnyEvent> pendingEvent_;

	// Segfault tracking — modules that keep crashing with SIGSEGV/SIGBUS
	// are permanently disabled after maxSegfaultRestarts to prevent WSL crash loops.
	std::unordered_map<std::string, int> segfaultCounts_;

	// Heartbeat tracking — last time each module published a heartbeat.
	// Initialised when a module publishes module_ready; erased on SIGKILL.
	std::unordered_map<std::string, std::chrono::steady_clock::time_point> moduleHeartbeats_;

	// Timers
	std::chrono::steady_clock::time_point llmTimeoutDeadline_;
	std::chrono::steady_clock::time_point lastWatchdogPoll_;
	std::chrono::steady_clock::time_point lastMemoryCheck_;
	std::chrono::steady_clock::time_point lastVramCheck_;

	// Type-safe transition table (defined in .cpp)
	struct Transitions;

	// Init helpers
	void loadModuleConfigFromYaml();
	void validateFsmCoverage() const;

	// FSM — feed an event into the state machine transition table
	std::optional<StateIndex> applyFsmEvent(const AnyEvent& event);

	// Message handling (inbound ZMQ subscribers)
	void handleMessage(const std::string& type, const nlohmann::json& msg);
	void executeUiCommand(const nlohmann::json& cmd);
	void handleToggleModule(UiAction action, const nlohmann::json& cmd);
	void processLlmTokenStream(const nlohmann::json& msg);

	// Periodic tasks (called from message handler on each poll cycle)
	void tickPeriodicTasks();

	// Watchdog
	void watchdogPoll();
	void memoryWatchdog();   ///< Check /proc/pid/statm, kill OOM modules
	void vramWatchdog();     ///< Check GPU VRAM via cudaMemGetInfo, evict if over cap
	void upgradeWatchdog();  ///< Auto-upgrade degraded conversation model when VRAM frees up

	// Dynamic priority-based GPU allocation
	void applyInteractionProfile(InteractionProfile profile);

	// FSM transition callbacks (called from Transitions overloads)
	void fsmOnStartListening();
	void fsmOnStopListeningAndSubmitPrompt();
	void fsmOnReturnToIdle();
	void buildAndPublishLlmPrompt(const std::string& userText);
	void publishSentenceToTts(const std::string& sentence);
	void fsmOnFinishTurn();
	void fsmOnBargeIn();
	void fsmOnModuleCrash(const std::string& moduleName);

	// Check if the conversation_model module is running
	[[nodiscard]] bool isLlmAvailable() const;

	// ── Five-Mode Architecture actions ──────────────────────────────────
	/// Non-recursive mode switch (avoids reentrant executeUiCommand dispatch).
	/// Returns true if the mode switch succeeded.
	[[nodiscard]] bool switchToMode(const std::string& mode);

	/// Execute a SwitchPlan's evict/spawn phases.
	/// Edge phases (drain, connect) are no-ops until edges are wired.
	/// Returns set of module names that failed to start.
	[[nodiscard]] tl::expected<std::unordered_set<std::string>, std::string>
	executeSwitchPlan(const core::pipeline_orchestrator::SwitchPlan& plan);

	/// Switch between boot modes (simple_llm / full_conversation).
	/// Spawns or stops A/V infrastructure modules accordingly.
	void handleBootModeSwitch(const std::string& targetMode);

	void handleSelectConversationModel(const nlohmann::json& cmd);
	void handleSelectTtsModel(const nlohmann::json& cmd);
	void handleToggleVideoConversation(const nlohmann::json& cmd);
	void handleSelectConversationSource(const nlohmann::json& cmd);

	// Session state
	[[nodiscard]] nlohmann::json buildSessionState() const;
	void restoreSessionState(const nlohmann::json& state);

	// Publish helpers
	void publishFsmState(StateIndex s);
	void publishModuleStatus(const std::string& module, const std::string& status);
	void publishError(const std::string& message);
};

