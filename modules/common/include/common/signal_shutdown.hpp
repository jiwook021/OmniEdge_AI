#pragma once

#include <atomic>
#include <csignal>
#include <cstring>
#include <thread>
#include <sys/signalfd.h>
#include <unistd.h>


/// Global shutdown flag — set when SIGINT/SIGTERM is received.
inline std::atomic<bool> gShutdownRequested{false};

/// Blocks SIGINT/SIGTERM in the calling thread and waits on a signalfd.
/// Calls stopper() once a signal arrives. Runs in a detached thread.
/// Typical usage:
///   launchShutdownWatcher([&node]() { node.stop(); });
template<typename StopFn>
void launchShutdownWatcher(StopFn stopper)
{
	sigset_t mask;
	sigemptyset(&mask);
	sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGTERM);

	// Block these signals so they are delivered via signalfd, not handlers.
	pthread_sigmask(SIG_BLOCK, &mask, nullptr);

	int sfd = signalfd(-1, &mask, SFD_CLOEXEC);
	if (sfd < 0) {
		// Fallback: mark shutdown immediately so callers still stop.
		gShutdownRequested.store(true, std::memory_order_release);
		stopper();
		return;
	}

	std::thread([sfd, stopper = std::move(stopper)]() {
		struct signalfd_siginfo info;
		// Blocks until SIGINT or SIGTERM is delivered.
		ssize_t n = read(sfd, &info, sizeof(info));
		close(sfd);

		if (n == static_cast<ssize_t>(sizeof(info))) {
			gShutdownRequested.store(true, std::memory_order_release);
			stopper();
		}
	}).detach();
}

