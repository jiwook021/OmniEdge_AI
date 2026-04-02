#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — ModuleNodeBase<Derived>  
//
// Formalises the node lifecycle that every module follows:
//   1. initialize()  →  onConfigure() + onLoadInferencer() + publishModuleReady()
//   2. run()         →  onBeforeRun() (optional) + messageRouter_.run()
//   3. stop()        →  onBeforeStop() (optional) + running_ = false + messageRouter_.stop()
//
// Derived types MUST implement:
//   tl::expected<void, std::string> onConfigure();
//   tl::expected<void, std::string> onLoadInferencer();
//   MessageRouter& router();
//   std::string_view moduleName() const;
//
// Derived types MAY implement:
//   void onBeforeRun();              // called once before entering the poll loop
//   void onAfterStop();              // called after the poll loop exits
//   void onBeforeStop() noexcept;    // called before messageRouter_.stop() (e.g. cancel in-flight work)
//   void onPublishReady();           // override default publishModuleReady() (e.g. degraded mode)
//
// Why CRTP:
//   - Zero virtual dispatch overhead on the hot path.
//   - The compiler enforces that every Derived implements onConfigure() and
//     onLoadInferencer().  A new module that forgets a hook is a compile error,
//     not a runtime bug.
//   - Shared lifecycle logic (ready signal, stop, poll loop) is written once.
//
// Thread safety:
//   initialize() and run() must be called from the same thread.
//   stop() is safe from any thread, including SIGTERM signal handlers.
// ---------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>

#include "common/diagnostic_ring.hpp"
#include <tl/expected.hpp>
#include "common/oe_tracy.hpp"


// Forward-declare — Derived must provide router() returning MessageRouter&.
class MessageRouter;

template <typename Derived>
class ModuleNodeBase {
public:
	ModuleNodeBase(const ModuleNodeBase&)            = delete;
	ModuleNodeBase& operator=(const ModuleNodeBase&) = delete;

	/// Non-virtual — enforces the init sequence across all modules.
	/// Calls Derived::onConfigure() → Derived::onLoadInferencer() →
	/// publishModuleReady() (or Derived::onPublishReady()).
	/// @throws std::runtime_error if any hook returns an error.
	void initialize()
	{
		OE_ZONE_SCOPED;
		auto& self = static_cast<Derived&>(*this);

		// Install crash dump handler — captures diagnostic ring on
		// SIGSEGV/SIGBUS/SIGABRT so the daemon can log what we were
		// doing when we died.
		crashDumpGuard_ = std::make_unique<CrashDumpGuard>(self.moduleName());
		OE_DIAG("initialize: start");

		spdlog::info("[{}] initialize: configuring...", self.moduleName());

		if (auto result = self.onConfigure(); !result) {
			throw std::runtime_error(
				"[" + std::string{self.moduleName()} + "] onConfigure failed: "
				+ result.error());
		}

		OE_DIAG("initialize: loading inferencer");
		spdlog::info("[{}] initialize: loading inferencer...", self.moduleName());

		if (auto result = self.onLoadInferencer(); !result) {
			throw std::runtime_error(
				"[" + std::string{self.moduleName()} + "] onLoadInferencer failed: "
				+ result.error());
		}

		if constexpr (requires { self.onPublishReady(); }) {
			self.onPublishReady();
		} else {
			self.router().publishModuleReady();
		}

		OE_DIAG("initialize: ready");
		spdlog::info("[{}] initialize: ready", self.moduleName());
	}

	/// Non-virtual — calls optional Derived::onBeforeRun(), then enters
	/// the MessageRouter poll loop.  Calls optional onAfterStop() when
	/// the loop exits.
	void run()
	{
		auto& self = static_cast<Derived&>(*this);
		running_.store(true, std::memory_order_release);

		if constexpr (requires { self.onBeforeRun(); }) {
			self.onBeforeRun();
		}

		OE_DIAG("run: entering poll loop");
		spdlog::info("[{}] entering poll loop", self.moduleName());
		self.router().run();

		if constexpr (requires { self.onAfterStop(); }) {
			self.onAfterStop();
		}

		running_.store(false, std::memory_order_release);
		spdlog::info("[{}] poll loop exited", self.moduleName());
	}

	/// Thread-safe stop — safe from signal handlers.
	/// Calls optional Derived::onBeforeStop() before signalling the router.
	/// Safe to call before initialize() — guards against uninitialised router.
	void stop() noexcept
	{
		if constexpr (requires(Derived& d) { d.onBeforeStop(); }) {
			static_cast<Derived&>(*this).onBeforeStop();
		}
		const bool wasRunning = running_.exchange(false, std::memory_order_acq_rel);
		if (wasRunning) {
			static_cast<Derived&>(*this).router().stop();
		}
	}

	[[nodiscard]] bool isRunning() const noexcept
	{
		return running_.load(std::memory_order_acquire);
	}

protected:
	ModuleNodeBase() = default;
	~ModuleNodeBase() = default;

	std::atomic<bool>                  running_{false};
	std::unique_ptr<CrashDumpGuard>    crashDumpGuard_;
};

