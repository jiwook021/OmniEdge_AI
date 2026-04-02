#include "orchestrator/omniedge_daemon.hpp"

#include <string>

#include "common/signal_shutdown.hpp"
#include "zmq/zmq_endpoint.hpp"

int main(int argc, char* argv[])
{
	std::string configPath;
	std::string profileOverride;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "--config" || arg == "-c") && i + 1 < argc)
			configPath = argv[++i];
		else if (arg == "--profile" && i + 1 < argc)
			profileOverride = argv[++i];
	}

	try {
		OeLogger::instance().setModule("omniedge_orchestrator");
		OeLogger::instance().initFile(OeLogger::resolveLogDir());
		OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "orchestrator");

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
