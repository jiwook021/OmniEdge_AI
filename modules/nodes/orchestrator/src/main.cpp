#include "orchestrator/omniedge_daemon.hpp"

#include <string>

#include <CLI/CLI.hpp>

#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"
#include "zmq/zmq_endpoint.hpp"

int main(int argc, char* argv[])
{
	std::string configPath;
	std::string profileOverride;
	CLI::App app{"omniedge_daemon"};
	app.add_option("--config", configPath, "Path to YAML config file")->required();
	app.add_option("--profile", profileOverride, "GPU tier profile override");
	CLI11_PARSE(app, argc, argv);

	try {
		oe::main::initLogger("omniedge_orchestrator", "orchestrator");

		OmniEdgeDaemon::Config config{
			.configFile         = configPath,
			.pubPort            = kDaemon,
			.subEndpoints       = {
				zmqEndpoint(kWsBridge),    // ui_command (WS Bridge)
				zmqEndpoint(kStt),         // transcription (WhisperSTT)
				zmqEndpoint(kFaceRecog),   // identity (FaceRecognition)
				zmqEndpoint(kAudioIngest), // vad_status (AudioIngest)
				zmqEndpoint(kLlm),         // llm_response (QwenLLM)
			},
			.gpuOverrideProfile = profileOverride,
			.sessionFilePath    = "session_state.json",
			.iniFilePath        = "",
		};

		// Validate config
		if (auto v = OmniEdgeDaemon::Config::validate(config); !v) {
			spdlog::error("Config validation failed: {}", v.error());
			return 1;
		}

		OmniEdgeDaemon daemon(config);

		daemon.initialize();
		OE_LOG_INFO("module_ready: module=omniedge_orchestrator");

		// Shutdown watcher (detached thread)
		launchShutdownWatcher([&daemon]() { daemon.stop(); });

		daemon.run();

		return 0;
	} catch (const std::exception& ex) {
		OE_LOG_ERROR("fatal: {}", ex.what());
		return 1;
	}
}
