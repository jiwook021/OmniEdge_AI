#include "common/diagnostic_ring.hpp"

#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

// All write() calls in this file are async-signal-safe crash-dump code
// where errors are unrecoverable. Suppress warn_unused_result.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"


// == DiagnosticRing ===========================================================

void DiagnosticRing::dumpToFd(int fd) const noexcept
{
	if (fd < 0) return;

	// Header — includes PID for the daemon to verify freshness.
	// getpid() and write() are async-signal-safe.
	{
		char hdr[80];
		int pos = 0;
		const char* p = "--- OmniEdge Crash Dump (pid=";
		while (*p) hdr[pos++] = *p++;

		// Integer-to-string for PID (async-signal-safe, stack only)
		char digits[16];
		int nd = 0;
		auto pid = static_cast<unsigned>(::getpid());
		if (pid == 0) { digits[nd++] = '0'; }
		else { while (pid > 0) { digits[nd++] = '0' + static_cast<char>(pid % 10); pid /= 10; } }
		for (int d = nd - 1; d >= 0; --d) hdr[pos++] = digits[d];

		const char* suf = ") ---\n";
		while (*suf) hdr[pos++] = *suf++;
		(void)::write(fd, hdr, pos);
	}

	// Iterate ring oldest→newest.
	const int h     = head_.load(std::memory_order_relaxed);
	const int start = (h >= kSlots) ? (h % kSlots) : 0;
	const int count = (h >= kSlots) ? kSlots : h;

	for (int i = 0; i < count; ++i) {
		const auto& slot = slots_[static_cast<std::size_t>((start + i) % kSlots)];
		if (slot.ts == 0) continue;

		char line[320]; // 256 msg + ~64 framing
		int  pos = 0;
		line[pos++] = '[';

		// uint64 → decimal (stack-only, async-signal-safe)
		char digits[20];
		int nd = 0;
		uint64_t t = slot.ts;
		if (t == 0) { digits[nd++] = '0'; }
		else { while (t > 0) { digits[nd++] = '0' + static_cast<char>(t % 10); t /= 10; } }
		for (int d = nd - 1; d >= 0; --d) line[pos++] = digits[d];

		line[pos++] = ']';
		line[pos++] = ' ';

		for (int m = 0; slot.msg[m] != '\0' && m < 255 && pos < 316; ++m)
			line[pos++] = slot.msg[m];
		line[pos++] = '\n';

		(void)::write(fd, line, pos);
	}
}

// == Global singleton =========================================================

DiagnosticRing& globalDiagRing() noexcept
{
	static DiagnosticRing ring;
	return ring;
}

// == Signal handler (file-scope) ==============================================

namespace {

int gCrashDumpFd = -1;   // Pre-opened by CrashDumpGuard

void crashSignalHandler(int sig)
{
	// 1. Dump the diagnostic ring
	globalDiagRing().dumpToFd(gCrashDumpFd);

	// 2. Append the signal number
	if (gCrashDumpFd >= 0) {
		char buf[48];
		int pos = 0;
		const char* p = "signal: ";
		while (*p) buf[pos++] = *p++;

		char digits[16];
		int nd = 0;
		int s = (sig < 0) ? -sig : sig;
		if (s == 0) { digits[nd++] = '0'; }
		else { while (s > 0) { digits[nd++] = '0' + static_cast<char>(s % 10); s /= 10; } }
		if (sig < 0) buf[pos++] = '-';
		for (int d = nd - 1; d >= 0; --d) buf[pos++] = digits[d];
		buf[pos++] = '\n';

		(void)::write(gCrashDumpFd, buf, pos);
		::fsync(gCrashDumpFd);
	}

	// 3. Exit immediately — do NOT return into the faulting frame.
	_exit(128 + sig);
}

} // anonymous namespace

// == CrashDumpGuard ===========================================================

CrashDumpGuard::CrashDumpGuard(std::string_view moduleName)
{
	// Ensure logs/ directory exists (idempotent).
	(void)::mkdir("logs", 0755);

	// Build path: logs/crash_<name>.dump
	std::string path;
	path.reserve(32 + moduleName.size());
	path += "logs/crash_";
	path += moduleName;
	path += ".dump";

	fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
	if (fd_ < 0) return;   // Best-effort — module still runs, just no dump on crash

	gCrashDumpFd = fd_;

	// Install one-shot signal handlers (SA_RESETHAND avoids recursive crash).
	struct sigaction sa{};
	sa.sa_handler = crashSignalHandler;
	sa.sa_flags   = SA_RESETHAND;
	sigemptyset(&sa.sa_mask);

	if (sigaction(SIGSEGV, &sa, &prevSigsegv_) != 0) {
		// Best-effort — log to stderr since OeLogger may not be initialized
		(void)::write(STDERR_FILENO, "CrashDumpGuard: sigaction(SIGSEGV) failed\n", 42);
	}
	if (sigaction(SIGBUS, &sa, &prevSigbus_) != 0) {
		(void)::write(STDERR_FILENO, "CrashDumpGuard: sigaction(SIGBUS) failed\n", 41);
	}
	if (sigaction(SIGABRT, &sa, &prevSigabrt_) != 0) {
		(void)::write(STDERR_FILENO, "CrashDumpGuard: sigaction(SIGABRT) failed\n", 42);
	}
}

CrashDumpGuard::~CrashDumpGuard()
{
	// Restore previous handlers.
	sigaction(SIGSEGV, &prevSigsegv_, nullptr);
	sigaction(SIGBUS,  &prevSigbus_,  nullptr);
	sigaction(SIGABRT, &prevSigabrt_, nullptr);

	gCrashDumpFd = -1;
	if (fd_ >= 0) ::close(fd_);
}

#pragma GCC diagnostic pop

