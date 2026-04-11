#include "cv/face_filter_node.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// Entry point for the omniedge_face_filter binary.
//
// CLI:  omniedge_face_filter --config <path/to/omniedge_config.yaml>
//
// Lifecycle:
//   1. Parse CLI -> load YAML config
//   2. Create node with inferencer (stub or ONNX)
//   3. initialize() — load model, open SHM, bind ZMQ
//   4. run() — blocking poll loop (processes frames until shutdown)
//   5. SIGTERM/SIGINT -> stop() via watcher thread -> run() returns -> exit
// ---------------------------------------------------------------------------

namespace {

// Load YAML and populate the FaceFilterNode::Config struct.
FaceFilterNode::Config loadConfig(const std::string& configPath)
{
	YAML::Node yaml = YAML::LoadFile(configPath);

	FaceFilterNode::Config cfg;
	cfg.moduleName = "face_filter";

	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve = makeModelPathResolver(modelsRoot);

	if (const auto& ff = yaml["face_filter"]) {
		cfg.faceMeshOnnxPath    = resolve(ff["model_path"].as<std::string>(cfg.faceMeshOnnxPath));
		cfg.filterManifestPath  = ff["filter_manifest"].as<std::string>(cfg.filterManifestPath);
		cfg.pubPort             = ff["zmq_pub_port"].as<int>(cfg.pubPort);
		cfg.jpegQuality         = ff["jpeg_quality"].as<int>(cfg.jpegQuality);
		cfg.outputShmName       = ff["shm_output"].as<std::string>(cfg.outputShmName);
		cfg.inputShmName        = ff["shm_input"].as<std::string>(cfg.inputShmName);
		cfg.activeFilterId      = ff["active_filter"].as<std::string>(cfg.activeFilterId);
		cfg.enabledAtStartup    = ff["enabled"].as<bool>(cfg.enabledAtStartup);
	}
	if (const auto& vi = yaml["video_ingest"]) {
		cfg.videoSubPort = vi["pub_port"].as<int>(cfg.videoSubPort);
	}
	if (const auto& ws = yaml["websocket_bridge"]) {
		cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
	}

	return cfg;
}

} // anonymous namespace

// Forward-declared factory for the inferencer.
// In production, links OnnxFaceFilterInferencer or the stub.
	[[nodiscard]] std::unique_ptr<FaceFilterInferencer> createStubFaceFilterInferencer();

int main(int argc, char* argv[])
{
	// 1. Parse CLI
	std::string configPath;
	CLI::App app{"omniedge_face_filter"};
	app.add_option("--config", configPath, "Path to YAML config file")->required();
	CLI11_PARSE(app, argc, argv);

	// 2. Load config from YAML
	FaceFilterNode::Config cfg;
	try {
		cfg = loadConfig(configPath);
	} catch (const std::exception& e) {
		std::cerr << "Failed to load config: " << e.what() << "\n";
		return 1;
	}

	// 3. Set up logging
	OeLogger::instance().setModule("omniedge_face_filter");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "face_filter");

	// 3b. Validate config
	if (auto v = FaceFilterNode::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// 4. Create and initialize the node
	FaceFilterNode node(cfg);
	node.setInferencer(createStubFaceFilterInferencer());
	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << "[omniedge_face_filter] Initialize failed: " << e.what() << "\n";
		return 1;
	}

	// 5. Shutdown watcher (detached thread)
	launchShutdownWatcher([&node]() { node.stop(); });

	// 6. Run the event loop
	node.run();

	return 0;
}
