#include "stt/stt_node.hpp"

#include <iostream>

#include <CLI/CLI.hpp>

#include "common/ini_config.hpp"
#include "common/model_path.hpp"
#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// main.cpp — entry point for omniedge_stt binary
//
// CLI:
//   omniedge_stt --config <path/to/omniedge.ini>
//
// Relevant INI sections:
//   [paths]                  — models_root
//   [stt]                    — encoder_engine, decoder_engine, tokenizer_dir,
//                              shm_input
//   [stt.hallucination_filter] — no_speech_prob_threshold, min_avg_logprob,
//                              max_consecutive_repeats
//   [ports] stt              — PUB port for "transcription"
//   [ports] audio_ingest     — AudioIngest PUB port to subscribe to
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	std::string configPath;
	CLI::App app{"omniedge_stt"};
	app.add_option("--config", configPath, "Path to INI config file")->required();
	CLI11_PARSE(app, argc, argv);

	IniConfig ini;
	if (!ini.loadFromFile(configPath)) {
		std::cerr << "[omniedge_stt] Failed to load INI: " << configPath << "\n";
		return 1;
	}

	STTNode::Config cfg;

	const auto resolve = makeModelPathResolver(ini.paths().modelsRoot);
	const auto& s = ini.stt();

	cfg.encoderEngineDir      = resolve(s.encoderEngineDir);
	cfg.decoderEngineDir      = resolve(s.decoderEngineDir);
	cfg.tokenizerDir          = resolve(s.tokenizerDir);
	cfg.inputShmName          = s.shmInput;
	cfg.noSpeechProbThreshold = s.noSpeechProbThreshold;
	cfg.minAvgLogprob         = s.minAvgLogprob;
	cfg.maxConsecutiveRepeats = s.maxConsecutiveRepeats;

	cfg.pubPort      = ini.port("stt",          cfg.pubPort);
	cfg.audioSubPort = ini.port("audio_ingest", cfg.audioSubPort);
	cfg.moduleName   = "stt";

	// Initialize logger with module name and file output
	oe::main::initLogger("omniedge_stt", "stt");

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

	try {
		node.run();  // blocks until stop() is called
	} catch (const std::exception& e) {
		spdlog::error("[omniedge_stt] run() failed: {}", e.what());
		return 1;
	}

	return 0;
}
