// main.cpp -- omniedge_conversation entry point
//
// Unified conversation module replacing the separate oe_stt + oe_llm + oe_tts
// pipeline. A single process handles multimodal inference via GemmaInferencer:
//   - Gemma-4 E2B  (2B effective, lighter — native STT + vision, text out)
//   - Gemma-4 E4B  (4B effective, default — native STT + vision, text out)
//
// Native TTS is not provided; the KokoroTTSNode sidecar consumes llm_response.
//
// Pattern 4 from class-patterns.md:
//   1. Parse CLI args (--config <path> --model <variant> [--degraded])
//   2. Load INI (canonical)
//   3. Populate Config struct
//   4. Install signal handlers
//   5. Construct ConversationNode, initialize(), run()
//   6. Exit 0 on clean shutdown, 1 on runtime failure, 78 (EX_CONFIG) on config

#include "conversation/conversation_node.hpp"
#include "conversation/gemma_inferencer.hpp"
#include "common/constants/conversation_constants.hpp"

#include <cstring>
#include <iostream>
#include <string>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>

#include "common/ini_config.hpp"
#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"

int main(int argc, char* argv[])
{
    // ---- Parse CLI ---------------------------------------------------------
    std::string configPath;
    std::string modelVariant;
    bool degradedMode = false;
    CLI::App app{"omniedge_conversation"};
    app.add_option("--config", configPath, "Path to INI config file")->required();
    app.add_option("--model", modelVariant, "Conversation model variant");
    app.add_flag("--degraded", degradedMode, "Run in degraded mode (stub inferencer)");
    CLI11_PARSE(app, argc, argv);

    // ---- Load INI (canonical) + YAML (per-variant model block) ------------
    IniConfig ini;
    if (!ini.loadFromFile(configPath)) {
        std::cerr << "[omniedge_conversation] Failed to load INI: "
                  << configPath << "\n";
        return 78;  // EX_CONFIG
    }

    YAML::Node yaml = oe::main::loadYaml("config/omniedge_config.yaml",
                                         "omniedge_conversation");

    // ---- Resolve model variant ---------------------------------------------
    if (modelVariant.empty()) {
        modelVariant = ini.conversation().defaultVariant;
    }

    YAML::Node modelYaml;
    if (yaml && yaml["conversation_models"]
             && yaml["conversation_models"][modelVariant]) {
        modelYaml = yaml["conversation_models"][modelVariant];
        std::cerr << "[omniedge_conversation] model_selected: " << modelVariant << "\n";
    } else {
        std::cerr << "[omniedge_conversation] WARNING: conversation_models[" << modelVariant
                  << "] not found in YAML, using defaults\n";
    }

    // ---- Build Config struct -----------------------------------------------
    ConversationNode::Config cfg;
    cfg.moduleName   = "conversation_model";  // Must match daemon's module registry
    cfg.modelVariant = modelVariant;

    cfg.pubPort          = ini.port("conversation_model", kConversationModel);
    cfg.daemonSubPort    = ini.port("daemon",             kDaemon);
    cfg.uiCommandSubPort = ini.port("websocket_bridge",   kWsBridge);
    cfg.audioSubPort     = ini.port("audio_ingest",       kAudioIngest);

    const auto resolve = makeModelPathResolver(ini.paths().modelsRoot);

    // HuggingFace transformers model directory. Per-variant fields come from YAML.
    cfg.modelDir = resolve(modelYaml["model_dir"].as<std::string>(""));

    cfg.cudaStreamPriority =
        modelYaml["cuda_stream_priority"].as<int>(kCudaPriorityLlm);

    cfg.audioShmName = ini.conversation().shmInputAudio;
    cfg.videoShmName = ini.videoIngest().shmInput;
    cfg.pollTimeout  = std::chrono::milliseconds(kPollTimeoutMs);
    cfg.degraded     = degradedMode;

    // ---- Initialize logger -------------------------------------------------
    oe::main::initLogger("omniedge_conversation", "conversation");

    // ---- Generation defaults from INI --------------------------------------
    {
        const auto& gen = ini.generation();
        cfg.generationDefaults.temperature = gen.temperature;
        cfg.generationDefaults.topP        = gen.topP;
        cfg.generationDefaults.maxTokens   = gen.maxTokens;
        cfg.generationDefaults.clamp();
    }

    // ---- Validate config ---------------------------------------------------
    if (auto v = ConversationNode::Config::validate(cfg); !v) {
        spdlog::error("Config validation failed: {}", v.error());
        return 78;  // EX_CONFIG
    }

    // ---- Create inferencer based on model variant --------------------------
    const auto modelId = parseConversationModelId(modelVariant);
    if (!modelId) {
        std::cerr << "[omniedge_conversation] Unknown model variant: "
                  << modelVariant
                  << " (expected one of: " << kModelGemmaE2b
                  << ", " << kModelGemmaE4b << ")\n";
        return 78;  // EX_CONFIG
    }

    GemmaInferencer::Variant variant = GemmaInferencer::Variant::kE4B;
    switch (*modelId) {
    case ConversationModelId::kGemmaE2B:
        variant = GemmaInferencer::Variant::kE2B;
        break;
    case ConversationModelId::kGemmaE4B:
        variant = GemmaInferencer::Variant::kE4B;
        break;
    }
    std::unique_ptr<ConversationInferencer> inferencer =
        createGemmaInferencer(variant);

    // ---- Construct and initialize node -------------------------------------
    ConversationNode node(cfg, std::move(inferencer));

    try {
        node.initialize();
    } catch (const std::exception& e) {
        std::cerr << "[omniedge_conversation] Initialize failed: " << e.what() << "\n";
        if (std::string_view{e.what()}.find("stub inferencer") != std::string_view::npos) {
            return 78;  // EX_CONFIG
        }
        return 1;
    }

    // ---- Shutdown watcher --------------------------------------------------
    launchShutdownWatcher([&node]() { node.stop(); });

    // ---- Run (blocks until stop()) -----------------------------------------
    try {
        node.run();
    } catch (const std::exception& e) {
        spdlog::error("[omniedge_conversation] run() failed: {}", e.what());
        return 1;
    }

    return 0;
}
