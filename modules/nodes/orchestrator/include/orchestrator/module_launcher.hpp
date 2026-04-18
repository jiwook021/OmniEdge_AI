#pragma once

#include <chrono>
#include <string>
#include <sys/types.h>
#include <unordered_set>
#include <vector>

#include <zmq.hpp>

#include <tl/expected.hpp>
#include "common/runtime_defaults.hpp"
#include "orchestrator/pipeline_graph.hpp"


/// Describes one managed child module process (populated from YAML).
struct ModuleDescriptor {
	std::string              name;
	std::string              binaryPath;
	std::vector<std::string> args;
	std::string              zmqPubEndpoint;
	int                      zmqPubPort{0};
	pid_t                    pid{-1};
	int                      restartCount{0};
	int                      maxRestarts{kMaxModuleRestarts};
	int                      evictPriority{0};   ///< Lower = evicted first
	std::size_t              vramBudgetMb{0};
	bool                     degraded{false};
	bool                     ready{false};
	bool                     onDemand{false};  ///< True for modules spawned via UI toggle, not at launch

	[[nodiscard]] bool isRunning() const noexcept { return pid > 0; }
};

/// posix_spawn wrapper + module readiness polling.
/// NOT thread-safe — all calls from orchestrator poll loop.
class ModuleLauncher {
public:
	ModuleLauncher() = default;
	ModuleLauncher(const ModuleLauncher&)            = delete;
	ModuleLauncher& operator=(const ModuleLauncher&) = delete;

	[[nodiscard]] tl::expected<pid_t, std::string> spawnModule(ModuleDescriptor& desc);

	void stopModule(ModuleDescriptor& desc,
	                std::chrono::seconds graceSec = std::chrono::seconds{5});

	/// Launch all in order, wait for module_ready. Returns names not ready.
	[[nodiscard]] std::unordered_set<std::string> launchAll(
		std::vector<ModuleDescriptor>& modules,
		zmq::context_t&                zmqCtx,
		std::chrono::seconds           timeout = std::chrono::seconds{30});

	/// Wait for already-spawned modules to publish module_ready.
	/// Unlike launchAll(), this does NOT call spawnModule() — it only waits.
	/// Returns names of modules that did not become ready within timeout.
	[[nodiscard]] std::unordered_set<std::string> waitForReady(
		std::vector<ModuleDescriptor*>& modules,
		zmq::context_t&                 zmqCtx,
		std::chrono::seconds            timeout = std::chrono::seconds{30});

	/// Non-blocking reap via waitpid(WNOHANG).
	[[nodiscard]] bool checkExited(ModuleDescriptor& desc);

	/// Like checkExited but also reports the signal that killed the process.
	/// exitSignal is set to the signal number if the process was killed by a
	/// signal (SIGSEGV, SIGBUS, etc.), or 0 if it exited normally.
	/// exitCode is set to the process exit code (only meaningful when exitSignal==0).
	[[nodiscard]] bool checkExitedWithSignal(ModuleDescriptor& desc, int& exitSignal, int& exitCode);

	/// Restart a crashed module (increments restartCount).
	/// Applies exponential backoff before respawning to avoid thrashing
	/// on deterministic crashes (e.g., corrupt model file).
	[[nodiscard]] tl::expected<pid_t, std::string> restartModule(ModuleDescriptor& desc);

	/// Launch a pipeline chain in topological order (upstream first).
	///
	/// For each stage in the pipeline:
	///   1. Append pipeline-generated CLI args to the module's descriptor
	///   2. Call spawnModule()
	///   3. Wait for module_ready (up to timeout per stage)
	///   4. Only then proceed to the next stage
	///
	/// If any stage fails to spawn or become ready, the entire pipeline
	/// is aborted and already-launched stages are stopped.
	///
	/// @param pipeline     The resolved pipeline descriptor
	/// @param graph        The pipeline graph (for CLI arg generation)
	/// @param modules      Module descriptors (indexed by module name)
	/// @param zmqCtx       ZMQ context for readiness polling
	/// @param timeout      Timeout per stage for module_ready
	/// @return Names of modules that failed to become ready, or empty on success
	[[nodiscard]] std::unordered_set<std::string> launchPipeline(
		const PipelineDesc& pipeline,
		const PipelineGraph& graph,
		std::unordered_map<std::string, ModuleDescriptor>& modules,
		zmq::context_t& zmqCtx,
		std::chrono::seconds timeout = std::chrono::seconds{30});

	/// Configure restart backoff timing. Both values in milliseconds.
	void setBackoffConfig(unsigned baseMs, unsigned maxMs) noexcept {
		baseBackoffMs_ = baseMs;
		maxBackoffMs_  = maxMs;
	}

private:
	unsigned baseBackoffMs_{kRestartBaseBackoffMs};
	unsigned maxBackoffMs_{kRestartMaxBackoffMs};
};

