#pragma once

#include <string>
#include <unordered_set>

/// Named set of modules that must run in a given interaction mode.
/// Shared by the daemon's profile switcher and the graph builder.
struct ModeDefinition {
	std::string                     name;
	std::unordered_set<std::string> required;
};
