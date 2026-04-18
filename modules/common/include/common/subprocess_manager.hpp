#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — SubprocessManager
//
// Single source of truth for all child process lifecycle: spawn, wait,
// health-check, kill.  Used by both one-shot inference scripts and the
// orchestrator's ModuleLauncher (via the static primitives).
//
// Three layers:
//
//   1. Static primitives (spawnProcess, terminateProcess, checkProcess):
//      Low-level building blocks.  ModuleLauncher uses these to manage
//      module processes while keeping its own descriptor/restart/readiness
//      logic.  SpawnOptions controls process-group creation, stderr
//      redirection, and PATH-vs-absolute spawn.
//
//   2. One-shot (static runOnce):
//      Spawn a subprocess, wait for it to exit with a deadline, return
//      success or error.  The subprocess is killed (SIGKILL) on timeout.
//
//   3. Managed daemon (spawn → isAlive/restart → ~SubprocessManager):
//      RAII owner of a long-running subprocess.  The destructor sends
//      SIGTERM, waits a grace period, then escalates to SIGKILL.
//
// Thread safety: NOT thread-safe.  Caller must serialize all calls.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <sys/types.h>

#include <tl/expected.hpp>
#include "common/runtime_defaults.hpp"


class SubprocessManager {
public:
	// -----------------------------------------------------------------------
	// Spawn options — shared by all spawn paths
	// -----------------------------------------------------------------------

	struct SpawnOptions {
		bool        useProcessGroup{false}; ///< POSIX_SPAWN_SETPGROUP — enables process-group kill
		std::string stderrFile;             ///< Redirect child stderr to file (empty = inherit)
		bool        searchPath{true};       ///< true = posix_spawnp (PATH), false = posix_spawn
	};

	// -----------------------------------------------------------------------
	// Exit information from a non-blocking waitpid check
	// -----------------------------------------------------------------------

	struct ExitInfo {
		bool exited{false};   ///< true if the process has exited
		int  exitCode{0};     ///< Exit code (meaningful when exited && signal == 0)
		int  signal{0};       ///< Signal number (0 if exited normally)
	};

	// -----------------------------------------------------------------------
	// Static primitives — used by ModuleLauncher and internally
	// -----------------------------------------------------------------------

	/// Spawn a child process.  Returns the child PID on success.
	///
	/// @param command  Executable name or absolute path.
	/// @param args     Arguments (do NOT include the program name — it is
	///                 prepended automatically).
	/// @param options  Process-group, stderr redirect, PATH search.
	[[nodiscard]] static tl::expected<pid_t, std::string>
	spawnProcess(const std::string& command,
	             const std::vector<std::string>& args,
	             SpawnOptions options);

	/// Overload using default SpawnOptions (PATH search, no process group).
	[[nodiscard]] static tl::expected<pid_t, std::string>
	spawnProcess(const std::string& command,
	             const std::vector<std::string>& args);

	/// SIGTERM → poll(gracePeriod) → SIGKILL → reap.
	/// If killProcessGroup is true, signals are also sent to the entire
	/// process group (negative PID).
	static void terminateProcess(pid_t pid,
	                             std::chrono::seconds gracePeriod,
	                             bool killProcessGroup = false) noexcept;

	/// Non-blocking waitpid check.  Returns exit status if the process
	/// has exited, or {.exited=false} if still running.
	[[nodiscard]] static ExitInfo checkProcess(pid_t pid);

	// -----------------------------------------------------------------------
	// One-shot execution — spawn, wait for exit, return result
	// -----------------------------------------------------------------------

	/// Spawn a subprocess and block until it exits or the deadline expires.
	/// On timeout the child is sent SIGKILL and reaped.
	[[nodiscard]] static tl::expected<void, std::string>
	runOnce(const std::string& command,
	        const std::vector<std::string>& args,
	        std::chrono::seconds timeout);

	// -----------------------------------------------------------------------
	// Managed daemon — long-running subprocess with RAII lifecycle
	// -----------------------------------------------------------------------

	struct Config {
		std::string              command;                                                  ///< Executable name or path
		std::vector<std::string> args;                                                     ///< Command-line arguments
		SpawnOptions             spawnOptions;                                              ///< Process-group, stderr redirect, PATH search
		std::chrono::seconds     startupTimeout{kSubprocessStartupTimeoutS};    ///< Max wait for healthProbe to return true
		std::chrono::seconds     gracePeriod{kSubprocessGracePeriodS};           ///< SIGTERM → SIGKILL grace period
		std::function<bool()>    healthProbe;                                              ///< Returns true when subprocess is ready (optional)
	};

	/// Spawn a long-running subprocess.  If Config::healthProbe is set,
	/// waits up to Config::startupTimeout for it to return true.
	[[nodiscard]] static tl::expected<SubprocessManager, std::string>
	spawn(Config cfg);

	/// SIGTERM → wait(gracePeriod) → SIGKILL → reap.
	~SubprocessManager();

	SubprocessManager(SubprocessManager&& other) noexcept;
	SubprocessManager& operator=(SubprocessManager&& other) noexcept;

	SubprocessManager(const SubprocessManager&)            = delete;
	SubprocessManager& operator=(const SubprocessManager&) = delete;

	/// Non-blocking check: is the subprocess still running?
	/// Invalidates pid_ if the child has exited or was already reaped.
	[[nodiscard]] bool isAlive();

	/// Kill the current subprocess and re-spawn using the original Config.
	[[nodiscard]] tl::expected<void, std::string> restart();

	/// Get the subprocess PID (-1 if not running).
	[[nodiscard]] pid_t pid() const noexcept { return pid_; }

private:
	SubprocessManager() = default;

	/// Delegates to terminateProcess() and resets pid_.
	void terminate() noexcept;

	pid_t  pid_{-1};
	Config cfg_;
};

