#include "tts/tts_node.hpp"

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
// OmniEdge_AI — omniedge_tts entry point
//
// Usage:
//   omniedge_tts --config <path/to/omniedge_config.yaml>
//
// The process:
//   1. Parses --config CLI argument.
//   2. Loads the YAML file and reads the [tts] section (fallback: [kokoro_tts]).
//   3. Installs SIGTERM / SIGINT handlers.
//   4. Constructs TTSNode, calls initialize(), then run().
//   5. Returns 0 on clean shutdown, 1 on init failure.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	// ── 1. CLI parsing ────────────────────────────────────────────────────
	std::string configPath;
	for (int i = 1; i < argc - 1; ++i) {
		if (std::string(argv[i]) == "--config") {
			configPath = argv[i + 1];
		}
	}

	if (configPath.empty()) {
		std::cerr << "Usage: omniedge_tts --config <path>\n";
		return 1;
	}

	// ── 2. Load YAML ──────────────────────────────────────────────────────
	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& ex) {
		std::cerr << std::format("Failed to load config '{}': {}\n",
		                         configPath, ex.what());
		return 1;
	}

	const YAML::Node& ttsYaml = yaml["tts"] ? yaml["tts"] : yaml["kokoro_tts"];
	if (!ttsYaml) {
		std::cerr << "Config is missing [tts] section.\n";
		return 1;
	}
	const YAML::Node& ttsCfg = ttsYaml;

	// ── 3. Build Config struct from YAML ──────────────────────────────────
	TTSNode::Config config;

	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve = makeModelPathResolver(modelsRoot);

	config.onnxModelPath = resolve(ttsCfg["onnx_model"]
		.as<std::string>(config.onnxModelPath));
	config.voiceDir      = resolve(ttsCfg["voice_dir"]
		.as<std::string>(config.voiceDir));
	config.defaultVoice  = ttsCfg["default_voice"]
		.as<std::string>(config.defaultVoice);
	config.speed         = ttsCfg["speed"]
		.as<float>(config.speed);
	config.shmOutput     = ttsCfg["shm_output"]
		.as<std::string>(config.shmOutput);
	config.pubPort       = ttsCfg["zmq_pub_port"]
		.as<int>(config.pubPort);
	const YAML::Node& llmCfg = yaml["llm"] ? yaml["llm"] : yaml["qwen_llm"];
	if (llmCfg) {
		config.subLlmPort = llmCfg["zmq_pub_port"]
			.as<int>(config.subLlmPort);
	}
	config.subDaemonPort = yaml["daemon"]["zmq_pub_port"]
		.as<int>(config.subDaemonPort);
	config.cudaStreamPriority = ttsCfg["cuda_stream_priority"]
		.as<int>(config.cudaStreamPriority);

	// ── 3b. Initialize logger with module name and file output ───────────
	OeLogger::instance().setModule("omniedge_tts");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "tts");

	// ── 3c. Validate config ──────────────────────────────────────────────
	if (auto v = TTSNode::Config::validate(config); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// ── 4. Construct the node (does NOT load GPU resources yet) ──────────
	TTSNode node(config, createOnnxKokoroInferencer());

	// ── 5. Initialize (throws on failure — caught below) ─────────────────
	try {
		node.initialize();
	} catch (const std::exception& ex) {
		std::cerr << std::format("TTSNode init failed: {}\n", ex.what());
		return 1;
	}

	// ── 6. Shutdown watcher (detached thread) ────────────────────────────
	launchShutdownWatcher([&node]() { node.stop(); });

	// ── 7. Run — blocks until stop() or SIGTERM ───────────────────────────
	node.run();

	// ── 8. Clean exit ─────────────────────────────────────────────────────
	return 0;
}
