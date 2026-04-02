#include "audio_denoise/audio_denoise_node.hpp"

#include <cstdlib>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — omniedge_audio_denoise entry point
//
// Usage:
//   omniedge_audio_denoise --config <path/to/omniedge_config.yaml>
//
// The process:
//   1. Parses --config CLI argument.
//   2. Loads the YAML file and reads the [audio_denoise] section.
//   3. Installs SIGTERM / SIGINT handlers.
//   4. Constructs AudioDenoiseNode, calls initialize(), then run().
//   5. Returns 0 on clean shutdown, 1 on init failure.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	// -- 1. CLI parsing --
	std::string configPath;
	for (int i = 1; i < argc - 1; ++i) {
		if (std::string(argv[i]) == "--config") {
			configPath = argv[i + 1];
		}
	}

	if (configPath.empty()) {
		std::cerr << "Usage: omniedge_audio_denoise --config <path>\n";
		return 1;
	}

	// -- 2. Load YAML --
	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& ex) {
		std::cerr << std::format("Failed to load config '{}': {}\n",
		                         configPath, ex.what());
		return 1;
	}

	const YAML::Node& dnCfg = yaml["audio_denoise"];
	if (!dnCfg) {
		std::cerr << "Config is missing [audio_denoise] section.\n";
		return 1;
	}

	// -- 3. Build Config struct from YAML --
	AudioDenoiseNode::Config config;

	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve    = makeModelPathResolver(modelsRoot);

	// Model paths (resolved against models_root)
	config.model1Path = resolve(dnCfg["model_stage1"].as<std::string>(config.model1Path));
	config.model2Path = resolve(dnCfg["model_stage2"].as<std::string>(config.model2Path));

	// Shared-memory ring names
	config.shmInput  = dnCfg["shm_input"].as<std::string>(config.shmInput);
	config.shmOutput = dnCfg["shm_output"].as<std::string>(config.shmOutput);

	// ZMQ ports
	config.pubPort         = dnCfg["zmq_pub_port"].as<int>(config.pubPort);
	config.subAudioPort    = yaml["audio_ingest"]["pub_port"].as<int>(config.subAudioPort);
	config.subWsBridgePort = yaml["websocket_bridge"]["zmq_pub_port"].as<int>(config.subWsBridgePort);

	// CUDA
	config.cudaStreamPriority = dnCfg["cuda_stream_priority"].as<int>(config.cudaStreamPriority);

	// -- 3b. Initialize logger --
	OeLogger::instance().setModule("omniedge_audio_denoise");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "audio_denoise");

	// -- 3c. Validate config --
	if (auto v = AudioDenoiseNode::Config::validate(config); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// -- 4. Construct the node --
	AudioDenoiseNode node(config, createOnnxDtlnInferencer());

	// -- 5. Initialize --
	try {
		node.initialize();
	} catch (const std::exception& ex) {
		std::cerr << std::format("AudioDenoiseNode init failed: {}\n", ex.what());
		return 1;
	}

	// -- 6. Shutdown watcher (detached thread) --
	launchShutdownWatcher([&node]() { node.stop(); });

	// -- 7. Run --
	node.run();

	// -- 8. Clean exit --
	return 0;
}
