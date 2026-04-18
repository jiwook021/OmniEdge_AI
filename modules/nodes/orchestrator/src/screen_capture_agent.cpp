#include "orchestrator/screen_capture_agent.hpp"

#include <format>
#include <utility>

#include "common/oe_logger.hpp"
#include "common/subprocess_manager.hpp"


namespace {
constexpr std::chrono::seconds kTaskkillTimeout{5};
}


ScreenCaptureAgent::~ScreenCaptureAgent()
{
	stop();
}

void ScreenCaptureAgent::configure(Config cfg)
{
	cfg_ = std::move(cfg);
}

void ScreenCaptureAgent::start()
{
	if (cfg_.exePath.empty()) {
		OE_LOG_WARN("screen_capture_agent: no exe path configured (screen_ingest.screen_capture_exe)");
		return;
	}

	// Already running — verify the WSL-side wrapper is still alive.
	if (pid_) {
		const auto info = SubprocessManager::checkProcess(*pid_);
		if (!info.exited) {
			OE_LOG_DEBUG("screen_capture_agent: already running (pid={})", *pid_);
			return;
		}
		OE_LOG_WARN("screen_capture_agent: previous instance exited (pid={})", *pid_);
		pid_.reset();
	}

	// cmd.exe understands Windows paths natively — no /mnt/c conversion.
	const std::string cmdLine = std::format(
		"\"{}\" --port {} --fps 1", cfg_.exePath, cfg_.tcpPort);

	OE_LOG_INFO("screen_capture_agent: launching cmd.exe /C {}", cmdLine);

	SubprocessManager::SpawnOptions opts;
	opts.useProcessGroup = true;    // detach from daemon's process group
	opts.searchPath      = true;    // cmd.exe is on PATH via WSL interop

	auto spawned = SubprocessManager::spawnProcess(
		"cmd.exe",
		{"/C", cmdLine},
		opts);

	if (!spawned) {
		OE_LOG_ERROR("screen_capture_agent: spawn failed: {}", spawned.error());
		return;
	}

	pid_ = *spawned;
	OE_LOG_INFO("screen_capture_agent: launched pid={}", *pid_);
}

void ScreenCaptureAgent::stop()
{
	if (!pid_) return;

	const pid_t wrapperPid = *pid_;
	OE_LOG_INFO("screen_capture_agent: stopping pid={}", wrapperPid);

	// taskkill targets the Windows process by image name — SIGTERM on the
	// WSL wrapper alone does not reliably kill the underlying Windows process.
	auto tkResult = SubprocessManager::runOnce(
		"taskkill.exe",
		{"/IM", "oe_screen_capture.exe", "/F"},
		kTaskkillTimeout);
	if (!tkResult) {
		OE_LOG_WARN("screen_capture_agent: taskkill failed: {}", tkResult.error());
	}

	// Reap the WSL-side wrapper.  If it hasn't exited after taskkill,
	// SubprocessManager::terminateProcess escalates SIGTERM → SIGKILL.
	const auto info = SubprocessManager::checkProcess(wrapperPid);
	if (!info.exited) {
		SubprocessManager::terminateProcess(wrapperPid, cfg_.gracePeriod, /*killProcessGroup=*/false);
	}

	pid_.reset();
}
