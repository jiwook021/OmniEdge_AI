#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Model Path Resolution
//
// Resolves model paths from omniedge_config.yaml. Paths that are already
// absolute are returned as-is. Relative paths are resolved against
// the models_root key from the YAML root node.
//
// Usage in main.cpp:
//   const auto resolve = makeModelPathResolver(yaml);
//   config.onnxModelPath = resolve(ttsCfg["onnx_model"].as<std::string>());
// ---------------------------------------------------------------------------

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>


/// Returns a callable that resolves a model path against models_root.
/// If the input path is absolute, it is returned unchanged.
/// If relative, models_root is prepended.
///
/// models_root itself supports ${OE_MODELS_DIR:-<fallback>} syntax:
///   - If OE_MODELS_DIR env var is set, it is used.
///   - Otherwise the YAML value (with env syntax stripped) is used.
[[nodiscard]] inline std::function<std::string(const std::string&)>
makeModelPathResolver(const std::string& modelsRoot)
{
	namespace fs = std::filesystem;

	// Resolve the models_root value.
	// Support "${OE_MODELS_DIR:-<default>}" shell-style syntax in YAML.
	std::string resolvedRoot = modelsRoot;

	// Check OE_MODELS_DIR environment variable first.
	if (const char* envDir = std::getenv("OE_MODELS_DIR"); envDir != nullptr) {
		resolvedRoot = envDir;
	} else if (resolvedRoot.starts_with("${")) {
		// Strip shell variable syntax: "${OE_MODELS_DIR:-/path}" -> "/path"
		const auto dashPos = resolvedRoot.find(":-");
		const auto bracePos = resolvedRoot.rfind('}');
		if (dashPos != std::string::npos && bracePos != std::string::npos) {
			resolvedRoot = resolvedRoot.substr(dashPos + 2,
			                                   bracePos - dashPos - 2);
		}
		// Expand $HOME if present in the fallback
		if (resolvedRoot.starts_with("${HOME}") || resolvedRoot.starts_with("$HOME")) {
			if (const char* home = std::getenv("HOME"); home != nullptr) {
				if (resolvedRoot.starts_with("${HOME}")) {
					resolvedRoot.replace(0, 7, home);
				} else {
					resolvedRoot.replace(0, 5, home);
				}
			}
		}
	}

	const fs::path rootPath{resolvedRoot};

	return [rootPath](const std::string& modelPath) -> std::string {
		if (modelPath.empty()) {
			return modelPath;
		}
		const fs::path p{modelPath};
		if (p.is_absolute()) {
			return modelPath;
		}
		return (rootPath / p).string();
	};
}

