#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — oeNodeMain: unified main() template for node binaries
//
// Encapsulates the 7-step entry point pattern repeated in 10+ main.cpp files:
//   1. Parse --config CLI argument
//   2. Load YAML file
//   3. Build Config struct (caller provides a lambda)
//   4. Initialize logger
//   5. Validate config
//   6. Construct node (caller provides a factory OR a configurator)
//   7. Initialize, launch shutdown watcher, run
//
// Two factory forms are supported (SFINAE-dispatched on the lambda signature):
//
//   A) Factory form — lambda returns NodeT as a prvalue:
//      return oeNodeMain<AudioDenoiseNode>(argc, argv,
//          "omniedge_audio_denoise", "audio_denoise",
//          [](const YAML::Node& y) -> AudioDenoiseNode::Config { ... },
//          [](AudioDenoiseNode::Config cfg) {
//              return AudioDenoiseNode(std::move(cfg), createInferencer());  // prvalue
//          }
//      );
//
//   B) Configurator form — lambda takes NodeT& after in-place construction.
//      Use this when the node is non-movable (deleted copy + user destructor)
//      and needs post-construction inferencer injection:
//      return oeNodeMain<BeautyNode>(argc, argv,
//          "omniedge_beauty", "beauty",
//          [](const YAML::Node& y) -> BeautyNode::Config { ... },
//          [](BeautyNode& node) { node.setInferencer(createBeautyInferencer()); }
//      );
//
// Simplified overload (no custom factory, default construction):
//   return oeNodeMain<VideoIngestNode>(argc, argv,
//       "omniedge_video_ingest", "video_ingest",
//       [](const YAML::Node& y) -> VideoIngestNode::Config { ... }
//   );
// ---------------------------------------------------------------------------

#include <format>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

#include "common/ini_config.hpp"
#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"


namespace oe_node_main_detail {

template<typename NodeT>
int runLifecycle(NodeT& node, std::string_view binaryName)
{
	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << std::format("[{}] Initialize failed: {}\n",
		                         binaryName, e.what());
		return 1;
	}

	launchShutdownWatcher([&node]() { node.stop(); });

	try {
		node.run();
	} catch (const std::exception& e) {
		spdlog::error("[{}] run() failed: {}", binaryName, e.what());
		return 1;
	}

	return 0;
}

template<typename NodeT, typename ConfigLoader>
bool prepareConfig(int argc, char* argv[],
                   std::string_view binaryName,
                   std::string_view iniSection,
                   ConfigLoader& configLoader,
                   typename NodeT::Config& cfgOut)
{
	std::string configPath;
	CLI::App app{std::string{binaryName}};
	app.add_option("--config", configPath, "Path to YAML config file")->required();
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError& e) {
		std::exit(app.exit(e));
	}

	YAML::Node yaml = oe::main::loadYaml(configPath, binaryName);
	if (!yaml) return false;

	IniConfig ini;
	(void)ini.loadFromFile("config/omniedge.ini");

	try {
		if constexpr (std::is_invocable_r_v<typename NodeT::Config,
		                                    ConfigLoader,
		                                    const YAML::Node&,
		                                    const IniConfig&>) {
			cfgOut = configLoader(yaml, ini);
		} else {
			cfgOut = configLoader(yaml);
		}
	} catch (const std::exception& e) {
		std::cerr << std::format("[{}] Config load failed: {}\n",
		                         binaryName, e.what());
		return false;
	}

	oe::main::initLogger(binaryName, iniSection);

	if (auto v = NodeT::Config::validate(cfgOut); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return false;
	}

	return true;
}

} // namespace oe_node_main_detail


// ---------------------------------------------------------------------------
// Factory form — lambda returns NodeT as a prvalue
// ---------------------------------------------------------------------------
template<typename NodeT, typename ConfigLoader, typename NodeFactory>
auto oeNodeMain(int argc, char* argv[],
                std::string_view binaryName,
                std::string_view iniSection,
                ConfigLoader     configLoader,
                NodeFactory      nodeFactory)
	-> std::enable_if_t<std::is_invocable_r_v<NodeT, NodeFactory, typename NodeT::Config>, int>
{
	typename NodeT::Config cfg;
	if (!oe_node_main_detail::prepareConfig<NodeT>(argc, argv, binaryName, iniSection,
	                                               configLoader, cfg)) {
		return 1;
	}

	auto node = nodeFactory(std::move(cfg));
	return oe_node_main_detail::runLifecycle(node, binaryName);
}


// ---------------------------------------------------------------------------
// Configurator form — lambda configures an already-constructed NodeT&
// Required for non-movable node types (deleted copy + user-declared destructor).
// ---------------------------------------------------------------------------
template<typename NodeT, typename ConfigLoader, typename Configurator>
auto oeNodeMain(int argc, char* argv[],
                std::string_view binaryName,
                std::string_view iniSection,
                ConfigLoader     configLoader,
                Configurator     configurator)
	-> std::enable_if_t<std::is_invocable_v<Configurator, NodeT&>
	                 && !std::is_invocable_r_v<NodeT, Configurator, typename NodeT::Config>, int>
{
	typename NodeT::Config cfg;
	if (!oe_node_main_detail::prepareConfig<NodeT>(argc, argv, binaryName, iniSection,
	                                               configLoader, cfg)) {
		return 1;
	}

	NodeT node(std::move(cfg));
	configurator(node);
	return oe_node_main_detail::runLifecycle(node, binaryName);
}


// ---------------------------------------------------------------------------
// Simplified overload — no custom factory, default construction
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
