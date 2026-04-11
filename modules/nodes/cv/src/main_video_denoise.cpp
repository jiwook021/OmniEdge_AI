#include "cv/video_denoise_node.hpp"

#include <cstdlib>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — omniedge_video_denoise entry point
//
// Usage:
//   omniedge_video_denoise --config <path/to/omniedge_config.yaml>
//
// The process:
//   1. Parses --config CLI argument.
//   2. Loads the YAML file and reads the [video_denoise] section.
//   3. Installs SIGTERM / SIGINT handlers.
//   4. Constructs VideoDenoiseNode, calls initialize(), then run().
//   5. Returns 0 on clean shutdown, 1 on init failure.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	// -- 1. CLI parsing --
	std::string configPath;
	CLI::App app{"omniedge_video_denoise"};
	app.add_option("--config", configPath, "Path to YAML config file")->required();
	CLI11_PARSE(app, argc, argv);

	// -- 2. Load YAML --
	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& ex) {
		std::cerr << std::format("Failed to load config '{}': {}\n",
		                         configPath, ex.what());
		return 1;
	}

	const YAML::Node& dnCfg = yaml["video_denoise"];
	if (!dnCfg) {
		std::cerr << "Config is missing [video_denoise] section.\n";
		return 1;
	}

	// -- 3. Build Config struct from YAML --
	VideoDenoiseNode::Config config;

	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve = makeModelPathResolver(modelsRoot);

	config.onnxModelPath = resolve(dnCfg["onnx_model"]
		.as<std::string>(config.onnxModelPath));
	config.shmInput      = dnCfg["shm_input"]
		.as<std::string>(config.shmInput);
	config.shmOutput     = dnCfg["shm_output"]
		.as<std::string>(config.shmOutput);
	config.pubPort       = dnCfg["zmq_pub_port"]
		.as<int>(config.pubPort);
	config.subVideoPort  = yaml["video_ingest"]["pub_port"]
		.as<int>(config.subVideoPort);
	config.subWsBridgePort = yaml["websocket_bridge"]["zmq_pub_port"]
		.as<int>(config.subWsBridgePort);
	config.temporalWindowSize = dnCfg["temporal_window"]
		.as<uint32_t>(config.temporalWindowSize);
	config.cudaStreamPriority = dnCfg["cuda_stream_priority"]
		.as<int>(config.cudaStreamPriority);

	// -- 3b. Initialize logger --
	OeLogger::instance().setModule("omniedge_video_denoise");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "video_denoise");

	// -- 3c. Validate config --
	if (auto v = VideoDenoiseNode::Config::validate(config); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// -- 4. Construct the node --
	VideoDenoiseNode node(config, createOnnxBasicVsrppInferencer());

	// -- 5. Initialize --
	try {
		node.initialize();
	} catch (const std::exception& ex) {
		std::cerr << std::format("VideoDenoiseNode init failed: {}\n", ex.what());
		return 1;
	}

	// -- 6. Shutdown watcher (detached thread) --
	launchShutdownWatcher([&node]() { node.stop(); });

	// -- 7. Run --
	node.run();

	// -- 8. Clean exit --
	return 0;
}
