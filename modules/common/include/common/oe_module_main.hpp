#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Module Bootstrap Helpers
//
// Eliminates duplicated YAML loading and logger initialization
// across all 13+ main.cpp entry points.
//
// CLI argument parsing is handled by CLI11 (added to all node executables
// via oe_add_node_exe() in OmniEdgeHelpers.cmake).
// ---------------------------------------------------------------------------

#include <format>
#include <iostream>
#include <string>
#include <string_view>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"


namespace oe::main {

// ---------------------------------------------------------------------------
// loadYaml — load and return a YAML::Node from a config file path.
//
// Prints error to stderr and returns empty node on failure.
//
// @param configPath  Path to the YAML config file
// @param binaryName  Binary name for error messages (e.g. "omniedge_bg_blur")
// @return Loaded YAML node, or empty node on failure (caller should check)
// ---------------------------------------------------------------------------
[[nodiscard]] inline YAML::Node loadYaml(const std::string& configPath,
                                         std::string_view binaryName)
{
	try {
		return YAML::LoadFile(configPath);
	} catch (const YAML::Exception& ex) {
		std::cerr << std::format("{}: failed to load config '{}': {}\n",
		                         binaryName, configPath, ex.what());
		return {};
	}
}

// ---------------------------------------------------------------------------
// initLogger — initialize the OeLogger singleton for this module.
//
// @param moduleName  Full module name (e.g. "omniedge_bg_blur")
// @param iniSection  INI section name for log level (e.g. "background_blur")
// ---------------------------------------------------------------------------
inline void initLogger(std::string_view moduleName,
                       std::string_view iniSection)
{
	OeLogger::instance().setModule(std::string{moduleName});
	OeLogger::instance().initFile(OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini",
	                                           std::string{iniSection});
}

} // namespace oe::main
