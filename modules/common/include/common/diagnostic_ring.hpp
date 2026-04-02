#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — DiagnosticRing + CrashDumpGuard
//
// Lock-free, async-signal-safe ring buffer that captures the last N
// diagnostic breadcrumbs before a module crash.  When the process
// receives SIGSEGV / SIGBUS / SIGABRT, the ring contents are written
// to a pre-opened file descriptor and the process exits.
//
// Usage:
//   OE_DIAG("loading model weights");        // any thread, any time
//   OE_DIAG("processing frame 42");
//   CrashDumpGuard guard("stt");             // once, at module init
//
// The daemon reads logs/crash_<name>.dump after watchdogPoll() reaps
// the child and logs the diagnostic trail alongside the crash event.
// ---------------------------------------------------------------------------

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <csignal>
#include <time.h>


class DiagnosticRing {
public:
	static constexpr int kSlots = 16;

	struct Slot {
		char     msg[256];
		uint64_t ts{0};      // CLOCK_MONOTONIC nanoseconds
	};

	/// Record.  Lock-free, safe from any thread.
	void record(std::string_view msg) noexcept
	{
		const int idx = head_.fetch_add(1, std::memory_order_relaxed) % kSlots;
		const auto len = std::min(msg.size(), std::size_t{255});
		std::memcpy(slots_[idx].msg, msg.data(), len);
		slots_[idx].msg[len] = '\0';
		slots_[idx].ts = steadyNanos();
	}

	/// Dump ring contents to fd, oldest entry first.
	/// Async-signal-safe: uses only write() and stack arithmetic.
	void dumpToFd(int fd) const noexcept;

private:
	static uint64_t steadyNanos() noexcept
	{
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
		     + static_cast<uint64_t>(ts.tv_nsec);
	}

	std::array<Slot, kSlots> slots_{};
	std::atomic<int>         head_{0};
};

// ---------------------------------------------------------------------------
// Global singleton — one ring per process (modules are separate processes).
// ---------------------------------------------------------------------------
DiagnosticRing& globalDiagRing() noexcept;

// ---------------------------------------------------------------------------
// CrashDumpGuard — RAII: opens crash dump file, installs signal handlers.
// On SIGSEGV/SIGBUS/SIGABRT the global ring is dumped and the process exits.
// Restores previous handlers on destruction.
// ---------------------------------------------------------------------------
class CrashDumpGuard {
public:
	/// @param moduleName  Used to build the dump path: logs/crash_<name>.dump
	explicit CrashDumpGuard(std::string_view moduleName);
	~CrashDumpGuard();

	CrashDumpGuard(const CrashDumpGuard&)            = delete;
	CrashDumpGuard& operator=(const CrashDumpGuard&) = delete;

private:
	int              fd_{-1};
	struct sigaction prevSigsegv_{};
	struct sigaction prevSigbus_{};
	struct sigaction prevSigabrt_{};
};


/// Record a diagnostic breadcrumb.  Safe from any thread, any time.
#define OE_DIAG(msg) ::globalDiagRing().record(msg)
