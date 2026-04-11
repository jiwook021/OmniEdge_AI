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
#include "orchestrator/mode_orchestrator.hpp"
#include "orchestrator/module_launcher.hpp"
#include "orchestrator/shm_registry.hpp"
#include "statemachine/prompt_assembler.hpp"
#include "statemachine/session_persistence.hpp"
#include "statemachine/state_machine.hpp"
#include "common/ui_action.hpp"


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
	[[nodiscard]] ModeOrchestrator& modeOrchestrator() noexcept { return *modeOrchestrator_; }
	[[nodiscard]] EventBus& eventBus() noexcept { return eventBus_; }

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
	std::unique_ptr<ModeOrchestrator>     modeOrchestrator_;
	std::unique_ptr<SessionPersistence>   sessionPersistence_;
	ModuleLauncher                        launcher_;
	ShmRegistry                           shmRegistry_;
	IniConfig                             iniConfig_;

	// Module descriptors (from YAML)
	std::vector<ModuleDescriptor> modules_;

	// Name→index map for O(1) module lookup (built once in loadModuleConfigFromYaml)
	std::unordered_map<std::string, std::size_t> moduleIndex_;

	// Conversation context
	std::string lastTranscription_;
	std::string lastLlmResponse_;
	std::string pendingSentence_;
	std::string lastCrashedModule_;

	// ── Four-mode architecture state ────────────────────────────────────
	// Active conversation model: "qwen_omni_7b" (default), "qwen_omni_3b", "gemma_e4b"
	std::string activeConversationModel_{"qwen_omni_7b"};
	// Active style for Image Transform mode
	std::string activeImageStyle_{"anime"};

	// Per-model config flags parsed from conversation_models YAML section.
	// Used by handleSelectConversationModel to decide companion module spawning.
	struct ConversationModelFlags {
		bool needsTts{false};           // !native_tts — needs companion TTS process
		bool supportsVision{false};     // native_vision — accepts video/image frames
		std::string resolvedEngineDir;  // absolute path from YAML engine_dir/model_dir + models_root
	};
	std::unordered_map<std::string, ConversationModelFlags> conversationModelFlags_;
	bool videoConversationEnabled_{false};  ///< True when camera toggle is active
	// Image transform timeout deadline
	std::chrono::steady_clock::time_point imageTransformTimeoutDeadline_;

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

	// Check if any LLM-capable module (legacy "llm" or "conversation_model") is running
	[[nodiscard]] bool isLlmAvailable() const;

	// ── Four-mode architecture actions ──────────────────────────────────
	/// Non-recursive mode switch (avoids reentrant executeUiCommand dispatch).
	/// Returns true if the mode switch succeeded.
	[[nodiscard]] bool switchToMode(const std::string& mode);

	void handleSelectConversationModel(const nlohmann::json& cmd);
	void handleSelectTtsModel(const nlohmann::json& cmd);
	void handleToggleVideoConversation(const nlohmann::json& cmd);
	void handleStartImageTransform(const nlohmann::json& cmd);
	void handleSelectImageStyle(const nlohmann::json& cmd);
	void handleUploadImage(const nlohmann::json& cmd);
	void fsmOnImageTransformComplete();
	void fsmOnImageTransformTimeout();

	// Session state
	[[nodiscard]] nlohmann::json buildSessionState() const;
	void restoreSessionState(const nlohmann::json& state);

	// Publish helpers
	void publishFsmState(StateIndex s);
	void publishModuleStatus(const std::string& module, const std::string& status);
	void publishError(const std::string& message);
};

