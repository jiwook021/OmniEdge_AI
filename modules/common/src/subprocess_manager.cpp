#include "common/subprocess_manager.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <utility>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

extern char** environ;


// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Build a null-terminated argv array: [command, args..., nullptr].
[[nodiscard]] std::vector<const char*> buildArgv(
	const std::string& command,
	const std::vector<std::string>& args)
{
	std::vector<const char*> argv;
	argv.reserve(args.size() + 2);
	argv.push_back(command.c_str());
	for (const auto& arg : args) {
		argv.push_back(arg.c_str());
	}
	argv.push_back(nullptr);
	return argv;
}

/// Poll waitpid until the child exits or the deadline passes.
/// On timeout, sends SIGKILL, reaps, and returns an error.
/// On success (exit code 0), returns void.
[[nodiscard]] tl::expected<void, std::string> waitForExit(
	pid_t childPid,
	std::chrono::steady_clock::time_point deadline)
{
	constexpr auto kPollInterval = std::chrono::milliseconds{10};

	while (true) {
		int status = 0;
		const pid_t result = ::waitpid(childPid, &status, WNOHANG);

		if (result == childPid) {
			if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
				return {};
			}
			if (WIFEXITED(status)) {
				return tl::unexpected(
					std::format("subprocess (pid={}) exited with code {}",
					            childPid, WEXITSTATUS(status)));
			}
			if (WIFSIGNALED(status)) {
				return tl::unexpected(
					std::format("subprocess (pid={}) killed by signal {}",
					            childPid, WTERMSIG(status)));
			}
			return tl::unexpected(
				std::format("subprocess (pid={}) exited abnormally", childPid));
		}

		if (result == -1 && errno != EINTR) {
			return tl::unexpected(
				std::format("waitpid(pid={}) error: {}", childPid, std::strerror(errno)));
		}

		if (std::chrono::steady_clock::now() >= deadline) {
			OE_LOG_WARN("subprocess_timeout: pid={} — sending SIGKILL", childPid);
			::kill(childPid, SIGKILL);
			::waitpid(childPid, nullptr, 0);  // reap
			return tl::unexpected(
				std::format("subprocess (pid={}) timed out", childPid));
		}

		std::this_thread::sleep_for(kPollInterval);
	}
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Static primitives
// ---------------------------------------------------------------------------

tl::expected<pid_t, std::string> SubprocessManager::spawnProcess(
	const std::string& command,
	const std::vector<std::string>& args)
{
	return spawnProcess(command, args, SpawnOptions{});
}

tl::expected<pid_t, std::string> SubprocessManager::spawnProcess(
	const std::string& command,
	const std::vector<std::string>& args,
	SpawnOptions options)
{
	auto argv = buildArgv(command, args);

	// -- file actions (stderr redirect) -----------------------------------
	posix_spawn_file_actions_t fileActions;
	posix_spawn_file_actions_init(&fileActions);
	int logFd = -1;

	if (!options.stderrFile.empty()) {
		logFd = ::open(options.stderrFile.c_str(),
		               O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
		if (logFd >= 0) {
			posix_spawn_file_actions_adddup2(&fileActions, logFd, STDERR_FILENO);
			posix_spawn_file_actions_addclose(&fileActions, logFd);
		}
	}

	// -- spawn attributes (process group) ---------------------------------
	posix_spawnattr_t attrs;
	posix_spawnattr_init(&attrs);
	if (options.useProcessGroup) {
		posix_spawnattr_setflags(&attrs, POSIX_SPAWN_SETPGROUP);
		posix_spawnattr_setpgroup(&attrs, 0);  // 0 = new group with child's PID
	}

	// -- spawn ------------------------------------------------------------
	pid_t childPid = -1;
	int err = 0;

	if (options.searchPath) {
		err = posix_spawnp(
			&childPid, command.c_str(),
			&fileActions, &attrs,
			const_cast<char* const*>(argv.data()), environ);
	} else {
		err = posix_spawn(
			&childPid, command.c_str(),
			&fileActions, &attrs,
			const_cast<char* const*>(argv.data()), environ);
	}

	posix_spawnattr_destroy(&attrs);
	posix_spawn_file_actions_destroy(&fileActions);
	if (logFd >= 0) ::close(logFd);

	if (err != 0) {
		return tl::unexpected(
			std::format("posix_spawn{}('{}') failed: {} (errno={})",
			            options.searchPath ? "p" : "",
			            command, std::strerror(err), err));
	}

	OE_LOG_DEBUG("subprocess_spawned: cmd={}, pid={}", command, childPid);
	return childPid;
}

void SubprocessManager::terminateProcess(pid_t pid,
                                         std::chrono::seconds gracePeriod,
                                         bool killProcessGroup) noexcept
{
	if (pid <= 0) return;

	OE_LOG_DEBUG("subprocess_terminate: pid={}, grace_sec={}, pgroup={}",
	             pid, gracePeriod.count(), killProcessGroup);

	if (killProcessGroup) ::kill(-pid, SIGTERM);
	::kill(pid, SIGTERM);

	const auto deadline = std::chrono::steady_clock::now() + gracePeriod;
	while (std::chrono::steady_clock::now() < deadline) {
		int status = 0;
		if (::waitpid(pid, &status, WNOHANG) > 0) {
			OE_LOG_DEBUG("subprocess_terminated_gracefully: pid={}", pid);
			return;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds{100});
	}

	OE_LOG_WARN("subprocess_sigkill_escalation: pid={}", pid);
	if (killProcessGroup) ::kill(-pid, SIGKILL);
	::kill(pid, SIGKILL);
	::waitpid(pid, nullptr, 0);  // blocking reap
}

SubprocessManager::ExitInfo SubprocessManager::checkProcess(pid_t pid)
{
	ExitInfo info;
	if (pid <= 0) {
		info.exited = true;
		return info;
	}

	int status = 0;
	if (::waitpid(pid, &status, WNOHANG) > 0) {
		info.exited = true;
		if (WIFEXITED(status)) {
			info.exitCode = WEXITSTATUS(status);
		} else if (WIFSIGNALED(status)) {
			info.signal = WTERMSIG(status);
		}
	}
	return info;
}

// ---------------------------------------------------------------------------
// One-shot execution
// ---------------------------------------------------------------------------

tl::expected<void, std::string> SubprocessManager::runOnce(
	const std::string& command,
	const std::vector<std::string>& args,
	std::chrono::seconds timeout)
{
	OE_ZONE_SCOPED;

	auto childResult = spawnProcess(command, args);
	if (!childResult) {
		return tl::unexpected(childResult.error());
	}

	const pid_t childPid = *childResult;
	const auto deadline = std::chrono::steady_clock::now() + timeout;

	auto waitResult = waitForExit(childPid, deadline);
	if (!waitResult) {
		OE_LOG_ERROR("subprocess_run_once_failed: cmd={}, error={}",
		             command, waitResult.error());
		return tl::unexpected(waitResult.error());
	}

	OE_LOG_DEBUG("subprocess_run_once_done: cmd={}, pid={}", command, childPid);
	return {};
}

// ---------------------------------------------------------------------------
// Managed daemon — spawn
// ---------------------------------------------------------------------------

tl::expected<SubprocessManager, std::string> SubprocessManager::spawn(Config cfg)
{
	OE_ZONE_SCOPED;

	auto childResult = spawnProcess(cfg.command, cfg.args, cfg.spawnOptions);
	if (!childResult) {
		return tl::unexpected(childResult.error());
	}

	SubprocessManager mgr;
	mgr.pid_ = *childResult;
	mgr.cfg_ = std::move(cfg);

	// If a health probe is provided, wait for it to report healthy.
	if (mgr.cfg_.healthProbe) {
		const auto deadline =
			std::chrono::steady_clock::now() + mgr.cfg_.startupTimeout;

		while (!mgr.cfg_.healthProbe()) {
			if (!mgr.isAlive()) {
				return tl::unexpected(
					std::format("subprocess '{}' (pid={}) died during startup",
					            mgr.cfg_.command, mgr.pid_));
			}
			if (std::chrono::steady_clock::now() >= deadline) {
				mgr.terminate();
				return tl::unexpected(
					std::format("subprocess '{}' (pid={}) startup timed out after {}s",
					            mgr.cfg_.command, mgr.pid_,
					            mgr.cfg_.startupTimeout.count()));
			}
			std::this_thread::sleep_for(std::chrono::milliseconds{100});
		}
		OE_LOG_INFO("subprocess_healthy: cmd={}, pid={}", mgr.cfg_.command, mgr.pid_);
	}

	return mgr;
}

// ---------------------------------------------------------------------------
// Destructor / move / instance methods — delegate to static primitives
// ---------------------------------------------------------------------------

SubprocessManager::~SubprocessManager()
{
	terminate();
}

SubprocessManager::SubprocessManager(SubprocessManager&& other) noexcept
	: pid_(std::exchange(other.pid_, -1))
	, cfg_(std::move(other.cfg_))
{}

SubprocessManager& SubprocessManager::operator=(SubprocessManager&& other) noexcept
{
	if (this != &other) {
		terminate();
		pid_ = std::exchange(other.pid_, -1);
		cfg_ = std::move(other.cfg_);
	}
	return *this;
}

bool SubprocessManager::isAlive() const
{
	if (pid_ <= 0) return false;

	int status = 0;
	const pid_t result = ::waitpid(pid_, &status, WNOHANG);
	// result == 0 means child is still running
	// result == pid means child has exited
	// result == -1 means error (ECHILD = no such child = already reaped)
	return result == 0;
}

tl::expected<void, std::string> SubprocessManager::restart()
{
	OE_LOG_INFO("subprocess_restart: cmd={}, old_pid={}", cfg_.command, pid_);
	terminate();

	auto childResult = spawnProcess(cfg_.command, cfg_.args, cfg_.spawnOptions);
	if (!childResult) {
		return tl::unexpected(childResult.error());
	}

	pid_ = *childResult;

	if (cfg_.healthProbe) {
		const auto deadline =
			std::chrono::steady_clock::now() + cfg_.startupTimeout;

		while (!cfg_.healthProbe()) {
			if (!isAlive()) {
				pid_ = -1;
				return tl::unexpected(
					std::format("subprocess '{}' died during restart", cfg_.command));
			}
			if (std::chrono::steady_clock::now() >= deadline) {
				terminate();
				return tl::unexpected(
					std::format("subprocess '{}' restart timed out after {}s",
					            cfg_.command, cfg_.startupTimeout.count()));
			}
			std::this_thread::sleep_for(std::chrono::milliseconds{100});
		}
	}

	return {};
}

void SubprocessManager::terminate() noexcept
{
	if (pid_ <= 0) return;
	terminateProcess(pid_, cfg_.gracePeriod, cfg_.spawnOptions.useProcessGroup);
	pid_ = -1;
}

