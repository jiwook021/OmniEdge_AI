#include "ingest/video_ingest_node.hpp"

#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// main_video.cpp — entry point for omniedge_video_ingest binary
//
// CLI:
//   omniedge_video_ingest --config <path/to/omniedge_config.yaml>
//                        [--profile <tier>]
//
// No business logic here — only config parsing, signal wiring, and
// construct / initialize / run.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	// 1 — CLI parsing: --config <path>
	std::string configPath;
	for (int i = 1; i < argc - 1; ++i) {
		if (std::string_view{argv[i]} == "--config") {
			configPath = argv[i + 1];
		}
	}
	if (configPath.empty()) {
		std::cerr << "Usage: omniedge_video_ingest --config <path>\n";
		return 1;
	}

	// 2 — Load YAML
	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& e) {
		std::cerr << "Failed to load config '" << configPath << "': "
				  << e.what() << "\n";
		return 1;
	}

	// 3 — Build Config from YAML with oe_defaults fallbacks
	VideoIngestNode::Config cfg;

	const auto& vi = yaml["video_ingest"];
	if (vi) {
		cfg.v4l2Device      = vi["v4l2_device"].as<std::string>(cfg.v4l2Device);
		cfg.frameWidth      = vi["frame_width"].as<uint32_t>(cfg.frameWidth);
		cfg.frameHeight     = vi["frame_height"].as<uint32_t>(cfg.frameHeight);
		cfg.pubPort         = vi["pub_port"].as<int>(cfg.pubPort);
		cfg.zmqSendHighWaterMark       = vi["zmq_sndhwm"].as<int>(cfg.zmqSendHighWaterMark);
	}
	const auto& ws = yaml["websocket_bridge"];
	if (ws) {
		cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
	}
	cfg.moduleName = "video_ingest";

	// 3b — Initialize logger with module name and file output
	OeLogger::instance().setModule("omniedge_video_ingest");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "video_ingest");

	// 3c — Validate config
	if (auto v = VideoIngestNode::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// 4 — Construct, initialise, run
	VideoIngestNode node(cfg);

	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << "[omniedge_video_ingest] Initialize failed: "
				  << e.what() << "\n";
		return 1;
	}

	// Shutdown watcher (detached thread)
	launchShutdownWatcher([&node]() { node.stop(); });

	node.run();  // blocks until stop() is called

	return 0;
}
