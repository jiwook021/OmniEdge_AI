#pragma once

// ---------------------------------------------------------------------------
// ScreenCaptureAgent — lifecycle owner for oe_screen_capture.exe on WSL2.
//
// The Windows-side agent captures the desktop via DXGI Duplication and
// streams JPEG frames over TCP to ScreenIngestNode.  We launch it through
// cmd.exe /C (WSL→Windows interop) and kill it via taskkill.exe because
// SIGTERM on the WSL interop wrapper does not propagate reliably to the
// underlying Windows process.
// ---------------------------------------------------------------------------

#include <chrono>
#include <optional>
#include <string>

#include <sys/types.h>

#include "common/constants/ingest_constants.hpp"


class ScreenCaptureAgent {
public:
	struct Config {
		std::string          exePath;          ///< Windows path, e.g. C:\dev\...\oe_screen_capture.exe
		int                  tcpPort{kScreenIngestTcpPort};    ///< TCP port the agent listens on
		std::chrono::seconds gracePeriod{1};   ///< SIGTERM → SIGKILL grace on the WSL wrapper after taskkill
	};

	ScreenCaptureAgent() = default;
	~ScreenCaptureAgent();

	ScreenCaptureAgent(const ScreenCaptureAgent&)            = delete;
	ScreenCaptureAgent& operator=(const ScreenCaptureAgent&) = delete;

	void configure(Config cfg);

	/// Launch oe_screen_capture.exe via WSL→Windows interop (cmd.exe /C).
	/// Idempotent: if the wrapper is still alive, returns without re-spawning;
	/// if the previous wrapper has exited, re-launches.
	/// No-op with warning log when exePath is empty.
	void start();

	/// Terminate the Windows process via taskkill.exe, then reap the
	/// WSL-side interop wrapper.  No-op if not running.
	void stop();

	[[nodiscard]] bool isConfigured() const noexcept { return !cfg_.exePath.empty(); }

private:
	Config               cfg_;
	std::optional<pid_t> pid_;
};
