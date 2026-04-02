#include "stt/stt_node.hpp"

#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// main.cpp — entry point for omniedge_stt binary
//
// CLI:
//   omniedge_stt --config <path/to/omniedge_config.yaml>
//
// Relevant YAML keys (new: stt.*, legacy fallback: whisper_stt.*):
//   stt.encoder_engine        — TRT encoder engine directory
//   stt.decoder_engine        — TRT decoder engine directory
//   stt.tokenizer_dir         — Whisper tokenizer assets directory
//   stt.zmq_pub               — PUB port (default: 5563)
//   stt.shm_input             — SHM name (default: /oe.aud.ingest)
//   stt.hallucination_filter  — filter threshold sub-block
//   audio_ingest.pub_port             — AudioIngestNode PUB port to subscribe to
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
		std::cerr << "Usage: omniedge_stt --config <path>\n";
		return 1;
	}

	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& e) {
		std::cerr << "Failed to load config: " << e.what() << "\n";
		return 1;
	}

	STTNode::Config cfg;

	// Try new "stt" key first, fall back to legacy "whisper_stt"
	const YAML::Node& sttYaml = yaml["stt"] ? yaml["stt"] : yaml["whisper_stt"];
	auto stt = sttYaml;
	if (!stt || !stt["encoder_engine"]) {
		stt = yaml["modules"]["stt"] ? yaml["modules"]["stt"] : yaml["modules"]["whisper_stt"];
	}
	const auto modelsRoot = yaml["models_root"].as<std::string>("");
	const auto resolve = makeModelPathResolver(modelsRoot);

	if (stt) {
		cfg.encoderEngineDir =
			resolve(stt["encoder_engine"].as<std::string>(cfg.encoderEngineDir));
		cfg.decoderEngineDir =
			resolve(stt["decoder_engine"].as<std::string>(cfg.decoderEngineDir));
		cfg.tokenizerDir =
			resolve(stt["tokenizer_dir"].as<std::string>(cfg.tokenizerDir));
		cfg.pubPort =
			stt["zmq_pub_port"].as<int>(cfg.pubPort);
		cfg.inputShmName =
			stt["shm_input"].as<std::string>(cfg.inputShmName);

		if (stt["hallucination_filter"]) {
			const auto& hf = stt["hallucination_filter"];
			cfg.noSpeechProbThreshold =
				hf["no_speech_prob_threshold"].as<float>(cfg.noSpeechProbThreshold);
			cfg.minAvgLogprob =
				hf["min_avg_logprob"].as<float>(cfg.minAvgLogprob);
			cfg.maxConsecutiveRepeats =
				hf["max_consecutive_repeats"].as<int>(cfg.maxConsecutiveRepeats);
		}
	}

	// Try top-level audio_ingest first, then modules.audio_ingest
	auto ai = yaml["audio_ingest"];
	if (!ai || !ai["pub_port"]) {
		ai = yaml["modules"]["audio_ingest"];
	}
	if (ai) {
		cfg.audioSubPort = ai["pub_port"].as<int>(cfg.audioSubPort);
	}

	cfg.moduleName = "stt";

	// Initialize logger with module name and file output
	OeLogger::instance().setModule("omniedge_stt");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "stt");

	// Validate config before constructing node
	if (auto v = STTNode::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	// Construct node with production TRT inferencer
	STTNode node(cfg);
	node.setInferencer(createTrtWhisperInferencer());

	try {
		node.initialize();
	} catch (const std::exception& e) {
		std::cerr << "[omniedge_stt] Initialize failed: " << e.what() << "\n";
		return 1;
	}

	// Shutdown watcher (detached thread)
	launchShutdownWatcher([&node]() { node.stop(); });

	node.run();  // blocks until stop() is called

	return 0;
}
