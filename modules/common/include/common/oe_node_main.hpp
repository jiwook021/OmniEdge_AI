#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI �� oeNodeMain: unified main() template for node binaries
//
// Encapsulates the 7-step entry point pattern repeated in 10+ main.cpp files:
//   1. Parse --config CLI argument
//   2. Load YAML file
//   3. Build Config struct (caller provides a lambda)
//   4. Initialize logger
//   5. Validate config
//   6. Construct node (caller provides a lambda)
//   7. Initialize, launch shutdown watcher, run
//
// Usage:
//   return oeNodeMain<AudioDenoiseNode>(argc, argv,
//       "omniedge_audio_denoise", "audio_denoise",
//       [](const YAML::Node& yaml) -> AudioDenoiseNode::Config { ... },
//       [](AudioDenoiseNode::Config cfg) { return AudioDenoiseNode(cfg, ...); }
//   );
//
// For nodes that don't need custom construction (no inferencer injection):
//   return oeNodeMain<VideoIngestNode>(argc, argv,
//       "omniedge_video_ingest", "video_ingest",
//       [](const YAML::Node& yaml) -> VideoIngestNode::Config { ... }
//   );
// ---------------------------------------------------------------------------

#include <format>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"


// ---------------------------------------------------------------------------
// oeNodeMain — full template with custom node factory
// ---------------------------------------------------------------------------
template<typename NodeT, typename ConfigLoader, typename NodeFactory>
int oeNodeMain(int argc, char* argv[],
               std::string_view binaryName,
               std::string_view iniSection,
               ConfigLoader     configLoader,
               NodeFactory      nodeFactory)
{
	// 1. CLI parsing
	std::string configPath;
	CLI::App app{std::string{binaryName}};
	app.add_option("--config", configPath, "Path to YAML config file")->required();
	CLI11_PARSE(app, argc, argv);

	// 2. Load YAML
	YAML::Node yaml = oe::main::loadYaml(configPath, binaryName);
	if (!yaml) return 1;

	// 3. Build Config struct (caller's lambda)
	typename NodeT::Config cfg;
	try {
		cfg = configLoader(yaml);
	} catch (const std::exception& e) {
		std::cerr << std::format("[{}] Config load failed: {}\n",
		                         binaryName, e.what());
		return 1;
	}

	// 4. Initialize logger
	oe::main::initLogger(binaryName, iniSection);

	// 5. Validate config
	if (auto v = NodeT::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// 6. Construct node (caller's factory)
	auto node = nodeFactory(std::move(cfg));

	// 7. Initialize
	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << std::format("[{}] Initialize failed: {}\n",
		                         binaryName, e.what());
		return 1;
	}

	// 8. Shutdown watcher + run
	launchShutdownWatcher([&node]() { node.stop(); });

	try {
		node.run();
	} catch (const std::exception& e) {
		spdlog::error("[{}] run() failed: {}", binaryName, e.what());
		return 1;
	}

	return 0;
}

// ---------------------------------------------------------------------------
// oeNodeMain — simplified overload for nodes with default construction
//
// Uses NodeT(std::move(cfg)) as the factory — suitable for nodes that don't
// need inferencer injection at construction time.
// ---------------------------------------------------------------------------
template<typename NodeT, typename ConfigLoader>
int oeNodeMain(int argc, char* argv[],
               std::string_view binaryName,
               std::string_view iniSection,
               ConfigLoader     configLoader)
{
	return oeNodeMain<NodeT>(argc, argv, binaryName, iniSection,
		std::move(configLoader),
		[](typename NodeT::Config cfg) { return NodeT(std::move(cfg)); });
}
