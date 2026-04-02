#include "ingest/audio_ingest_node.hpp"

#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// main_audio.cpp — entry point for omniedge_audio_ingest binary
//
// CLI:
//   omniedge_audio_ingest --config <path/to/omniedge_config.yaml>
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	std::string configPath;
	for (int i = 1; i < argc - 1; ++i) {
		if (std::string_view{argv[i]} == "--config") {
			configPath = argv[i + 1];
		}
	}
	if (configPath.empty()) {
		std::cerr << "Usage: omniedge_audio_ingest --config <path>\n";
		return 1;
	}

	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& e) {
		std::cerr << "Failed to load config: " << e.what() << "\n";
		return 1;
	}

	AudioIngestNode::Config cfg;

	const auto& ai = yaml["audio_ingest"];
	if (ai) {
		cfg.windowsHostIp      = ai["windows_host_ip"].as<std::string>(cfg.windowsHostIp);
		cfg.audioTcpPort       = ai["tcp_port"].as<int>(cfg.audioTcpPort);
		cfg.sampleRateHz       = ai["sample_rate_hz"].as<uint32_t>(cfg.sampleRateHz);
		cfg.chunkSamples       = ai["chunk_samples"].as<uint32_t>(cfg.chunkSamples);
		cfg.zmqSendHighWaterMark          = ai["zmq_sndhwm"].as<int>(cfg.zmqSendHighWaterMark);
		cfg.pubPort            = ai["pub_port"].as<int>(cfg.pubPort);
	}
	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve = makeModelPathResolver(modelsRoot);

	const auto& vad = ai["vad"];
	if (vad) {
		cfg.vadModelPath       = resolve(vad["model_path"].as<std::string>(cfg.vadModelPath));
		cfg.vadSpeechThreshold = vad["speech_threshold"].as<float>(cfg.vadSpeechThreshold);
		cfg.silenceDurationMs  = vad["silence_duration_ms"].as<uint32_t>(cfg.silenceDurationMs);
	}
	const auto& ws = yaml["websocket_bridge"];
	if (ws) {
		cfg.wsBridgeSubPort    = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
	}
	cfg.moduleName = "audio_ingest";

	// Initialize logger with module name and file output
	OeLogger::instance().setModule("omniedge_audio_ingest");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "audio_ingest");

	// Validate config
	if (auto v = AudioIngestNode::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	AudioIngestNode node(cfg);

	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << "[omniedge_audio_ingest] Initialize failed: "
				  << e.what() << "\n";
		return 1;
	}

	// Shutdown watcher (detached thread)
	launchShutdownWatcher([&node]() { node.stop(); });

	node.run();  // blocks until stop() is called

	return 0;
}
