#include "orchestrator/module_launcher.hpp"

#include <format>
#include <thread>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/subprocess_manager.hpp"
#include "zmq/zmq_endpoint.hpp"


tl::expected<pid_t, std::string> ModuleLauncher::spawnModule(ModuleDescriptor& moduleDescriptor)
{
	OE_ZONE_SCOPED;
	SubprocessManager::SpawnOptions opts;
	opts.useProcessGroup = true;
	opts.stderrFile = OeLogger::resolveLogDir() + "/" + std::string{kConsolidatedLogFile};
	opts.searchPath = false;  // ModuleLauncher uses absolute binaryPath

	auto result = SubprocessManager::spawnProcess(
		moduleDescriptor.binaryPath, moduleDescriptor.args, opts);

	if (!result) {
		return tl::make_unexpected(std::format(
			"spawn failed for {}: {}", moduleDescriptor.name, result.error()));
	}

	moduleDescriptor.pid   = *result;
	moduleDescriptor.ready = false;
	OE_LOG_INFO("module_spawned: name={}, pid={}", moduleDescriptor.name, *result);
	return *result;
}

void ModuleLauncher::stopModule(ModuleDescriptor& moduleDescriptor,
                               std::chrono::seconds gracePeriod)
{
	OE_ZONE_SCOPED;
	if (moduleDescriptor.pid <= 0) return;

	OE_LOG_INFO("module_stop_requested: name={}, pid={}, grace_sec={}",
	          moduleDescriptor.name, moduleDescriptor.pid, gracePeriod.count());

	SubprocessManager::terminateProcess(
		moduleDescriptor.pid, gracePeriod, /*killProcessGroup=*/true);

	moduleDescriptor.pid   = -1;
	moduleDescriptor.ready = false;
}

bool ModuleLauncher::checkExited(ModuleDescriptor& moduleDescriptor)
{
	int unusedSignal = 0;
	int unusedCode = 0;
	return checkExitedWithSignal(moduleDescriptor, unusedSignal, unusedCode);
}

bool ModuleLauncher::checkExitedWithSignal(ModuleDescriptor& moduleDescriptor,
                                           int& exitSignal, int& exitCode)
{
	exitSignal = 0;
	exitCode   = 0;
	if (moduleDescriptor.pid <= 0) return true;

	auto info = SubprocessManager::checkProcess(moduleDescriptor.pid);
	if (info.exited) {
		if (info.signal != 0) {
			exitSignal = info.signal;
			OE_LOG_ERROR("module_killed_by_signal: name={}, pid={}, signal={}",
			           moduleDescriptor.name, moduleDescriptor.pid, info.signal);
		} else {
			exitCode = info.exitCode;
			OE_LOG_WARN("module_exited_unexpectedly: name={}, pid={}, exit_code={}",
			          moduleDescriptor.name, moduleDescriptor.pid, info.exitCode);
		}
		moduleDescriptor.pid   = -1;
		moduleDescriptor.ready = false;
		return true;
	}
	return false;
}

tl::expected<pid_t, std::string> ModuleLauncher::restartModule(ModuleDescriptor& moduleDescriptor)
{
	OE_ZONE_SCOPED;
	if (moduleDescriptor.restartCount >= moduleDescriptor.maxRestarts) {
		return tl::make_unexpected(std::format(
			"{}: max restarts exceeded ({}/{})",
			moduleDescriptor.name, moduleDescriptor.restartCount, moduleDescriptor.maxRestarts));
	}

	// Exponential backoff: 500, 1000, 2000, 4000, 8000, 16000 ms (capped at maxBackoffMs_).
	// Prevents deterministic crashes (e.g., corrupt model file) from burning through
	// all max_restarts in seconds, wasting VRAM allocation cycles each time.
	const auto backoffMs = std::min(
		baseBackoffMs_ * (1u << std::min(static_cast<unsigned>(moduleDescriptor.restartCount), 5u)),
		maxBackoffMs_
	);
	OE_LOG_INFO("restart_backoff: module={}, attempt={}/{}, wait={}ms",
	            moduleDescriptor.name, moduleDescriptor.restartCount + 1,
	            moduleDescriptor.maxRestarts, backoffMs);
	std::this_thread::sleep_for(std::chrono::milliseconds{backoffMs});

	stopModule(moduleDescriptor, std::chrono::seconds{kSubprocessGracePeriodS});
	moduleDescriptor.restartCount++;
	OE_LOG_WARN("module_restarting: name={}, attempt={}/{}",
	          moduleDescriptor.name, moduleDescriptor.restartCount, moduleDescriptor.maxRestarts);
	return spawnModule(moduleDescriptor);
}

std::unordered_set<std::string> ModuleLauncher::launchAll(
	std::vector<ModuleDescriptor>& modulesToLaunch,
	zmq::context_t&                zmqContext,
	std::chrono::seconds           readinessTimeout)
{
	OE_ZONE_SCOPED;
	std::unordered_set<std::string> modulesNotReady;

	for (auto& moduleDescriptor : modulesToLaunch) {
		if (moduleDescriptor.onDemand) continue;  // spawned via UI toggle, not at launch
		if (auto spawnResult = spawnModule(moduleDescriptor); !spawnResult) {
			OE_LOG_ERROR("launch_spawn_failed: name={}, error={}",
			           moduleDescriptor.name, spawnResult.error());
			modulesNotReady.insert(moduleDescriptor.name);
		}
	}

	// Collect names of modules awaiting readiness confirmation
	std::unordered_set<std::string> modulesPendingReady;
	for (const auto& moduleDescriptor : modulesToLaunch) {
		if (moduleDescriptor.isRunning()) {
			modulesPendingReady.insert(moduleDescriptor.name);
		}
	}

	// Subscribe to module_ready messages from all running modules
	zmq::socket_t readinessSubscriber(zmqContext, ZMQ_SUB);
	readinessSubscriber.set(zmq::sockopt::subscribe, "module_ready");
	readinessSubscriber.set(zmq::sockopt::rcvtimeo, 1000);

	for (const auto& moduleDescriptor : modulesToLaunch) {
		if (moduleDescriptor.isRunning() && moduleDescriptor.zmqPubPort > 0) {
			readinessSubscriber.connect(
				zmqEndpoint(moduleDescriptor.zmqPubPort));
		}
	}

	// ZMQ slow-joiner mitigation — brief delay for subscriptions to propagate.
	// Modules re-publish module_ready after entering their poll loop, so even
	// if the first publish was missed, the second one will be caught.
	constexpr auto kZmqSlowJoinerDelayMs = std::chrono::milliseconds{250};
	std::this_thread::sleep_for(kZmqSlowJoinerDelayMs);

	const auto readinessDeadline = std::chrono::steady_clock::now() + readinessTimeout;
	while (!modulesPendingReady.empty() &&
	       std::chrono::steady_clock::now() < readinessDeadline) {
		zmq::message_t receivedMessage;
		if (!readinessSubscriber.recv(receivedMessage, zmq::recv_flags::none)) continue;

		std::string rawFrame(static_cast<char*>(receivedMessage.data()),
		                     receivedMessage.size());
		auto topicSeparatorPos = rawFrame.find(' ');
		if (topicSeparatorPos == std::string::npos) continue;

		try {
			auto parsedJson = nlohmann::json::parse(rawFrame.substr(topicSeparatorPos + 1));
			if (parsedJson.contains("module")) {
				std::string readyModuleName = parsedJson["module"].get<std::string>();
				if (modulesPendingReady.erase(readyModuleName) > 0) {
					OE_LOG_INFO("module_ready_confirmed: name={}", readyModuleName);
					for (auto& moduleDescriptor : modulesToLaunch) {
						if (moduleDescriptor.name == readyModuleName) {
							moduleDescriptor.ready = true;
							break;
						}
					}
				}
			}
		} catch (const nlohmann::json::exception&) {
			// Malformed frame — skip
		}
	}

	for (const auto& pendingModuleName : modulesPendingReady) {
		// Check if the module process is still alive — if so, treat it as
		// ready despite missing the ZMQ confirmation (slow-joiner race).
		bool stillRunning = false;
		for (auto& moduleDescriptor : modulesToLaunch) {
			if (moduleDescriptor.name == pendingModuleName &&
			    moduleDescriptor.isRunning()) {
				stillRunning = true;
				moduleDescriptor.ready = true;
				OE_LOG_INFO("module_ready_assumed: name={} — process alive, "
				          "ZMQ confirmation missed (slow joiner)",
				          pendingModuleName);
				break;
			}
		}
		if (!stillRunning) {
			OE_LOG_WARN("module_ready_timeout: name={}, timeout_sec={} — process dead",
			          pendingModuleName, readinessTimeout.count());
			modulesNotReady.insert(pendingModuleName);
		}
	}
	return modulesNotReady;
}

std::unordered_set<std::string> ModuleLauncher::waitForReady(
	std::vector<ModuleDescriptor*>& modulePtrs,
	zmq::context_t&                 zmqContext,
	std::chrono::seconds            readinessTimeout)
{
	std::unordered_set<std::string> modulesNotReady;

	// Collect names of modules awaiting readiness confirmation
	std::unordered_set<std::string> modulesPendingReady;
	for (const auto* moduleDescriptor : modulePtrs) {
		if (moduleDescriptor->isRunning()) {
			modulesPendingReady.insert(moduleDescriptor->name);
		}
	}

	if (modulesPendingReady.empty()) return modulesNotReady;

	// Subscribe to module_ready messages from all running modules
	zmq::socket_t readinessSubscriber(zmqContext, ZMQ_SUB);
	readinessSubscriber.set(zmq::sockopt::subscribe, "module_ready");
	readinessSubscriber.set(zmq::sockopt::rcvtimeo, 1000);

	for (const auto* moduleDescriptor : modulePtrs) {
		if (moduleDescriptor->isRunning() && moduleDescriptor->zmqPubPort > 0) {
			readinessSubscriber.connect(
				zmqEndpoint(moduleDescriptor->zmqPubPort));
		}
	}

	// ZMQ slow-joiner mitigation — brief delay for subscriptions to propagate
	std::this_thread::sleep_for(std::chrono::milliseconds{250});

	const auto readinessDeadline = std::chrono::steady_clock::now() + readinessTimeout;
	while (!modulesPendingReady.empty() &&
	       std::chrono::steady_clock::now() < readinessDeadline) {
		zmq::message_t receivedMessage;
		if (!readinessSubscriber.recv(receivedMessage, zmq::recv_flags::none)) continue;

		std::string rawFrame(static_cast<char*>(receivedMessage.data()),
		                     receivedMessage.size());
		auto topicSeparatorPos = rawFrame.find(' ');
		if (topicSeparatorPos == std::string::npos) continue;

		try {
			auto parsedJson = nlohmann::json::parse(rawFrame.substr(topicSeparatorPos + 1));
			if (parsedJson.contains("module")) {
				std::string readyModuleName = parsedJson["module"].get<std::string>();
				if (modulesPendingReady.erase(readyModuleName) > 0) {
					OE_LOG_INFO("module_ready_confirmed: name={}", readyModuleName);
					for (auto* moduleDescriptor : modulePtrs) {
						if (moduleDescriptor->name == readyModuleName) {
							moduleDescriptor->ready = true;
							break;
						}
					}
				}
			}
		} catch (const nlohmann::json::exception&) {
			// Malformed frame — skip
		}
	}

	for (const auto& pendingModuleName : modulesPendingReady) {
		OE_LOG_WARN("module_ready_timeout: name={}, timeout_sec={}",
		          pendingModuleName, readinessTimeout.count());
		modulesNotReady.insert(pendingModuleName);
	}
	return modulesNotReady;
}

