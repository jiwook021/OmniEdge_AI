// main.cpp -- omniedge_conversation entry point
//
// Unified conversation module replacing the separate oe_stt + oe_llm + oe_tts
// pipeline.  A single process handles multimodal inference via one of:
//   - Qwen2.5-Omni-7B  (default, native audio I/O)
//   - Qwen2.5-Omni-3B  (lighter, native audio I/O)
//   - Gemma 4 E4B       (text-only output, TTS sidecar required)
//
// Pattern 4 from class-patterns.md:
//   1. Parse CLI args (--config <path> --model <variant> [--degraded])
//   2. Load YAML
//   3. Populate Config struct
//   4. Install signal handlers
//   5. Construct ConversationNode, initialize(), run()
//   6. Exit 0 on clean shutdown, 1 on init failure

#include "conversation/conversation_node.hpp"
#include "conversation/qwen_omni_inferencer.hpp"
#include "conversation/gemma_e4b_inferencer.hpp"
#include "common/constants/conversation_constants.hpp"

#include <cstring>
#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "common/ini_config.hpp"
#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"

int main(int argc, char* argv[])
{
    // ---- Parse CLI ---------------------------------------------------------
    std::string configPath;
    std::string modelVariant;
    bool        degradedMode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            configPath = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelVariant = argv[++i];
        } else if (std::strcmp(argv[i], "--degraded") == 0) {
            degradedMode = true;
        }
    }
    if (configPath.empty()) {
        std::cerr << "Usage: omniedge_conversation --config <path> --model <variant> [--degraded]\n";
        return 1;
    }

    // ---- Load YAML ---------------------------------------------------------
    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(configPath);
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load config " << configPath << ": " << e.what() << "\n";
        return 1;
    }

    // ---- Resolve model variant ---------------------------------------------
    if (modelVariant.empty()) {
        modelVariant = yaml["default_conversation_model"].as<std::string>(
            std::string(kModelQwenOmni7b));
    }

    YAML::Node modelYaml;
    if (yaml["conversation_models"] && yaml["conversation_models"][modelVariant]) {
        modelYaml = yaml["conversation_models"][modelVariant];
        std::cerr << "[omniedge_conversation] model_selected: " << modelVariant << "\n";
    } else {
        std::cerr << "[omniedge_conversation] WARNING: conversation_models[" << modelVariant
                  << "] not found in YAML, using defaults\n";
    }

    // ---- Build Config struct -----------------------------------------------
    ConversationNode::Config cfg;
    cfg.modelVariant = modelVariant;

    cfg.pubPort =
        modelYaml["zmq_pub_port"]
            .as<int>(kConversationModel);

    cfg.daemonSubPort =
        yaml["daemon"]["zmq_pub_port"]
            .as<int>(kDaemon);

    cfg.uiCommandSubPort =
        yaml["websocket_bridge"]["zmq_pub_port"]
            .as<int>(kWsBridge);

    const auto modelsRoot = yaml["models_root"].as<std::string>("");
    const auto resolve = makeModelPathResolver(modelsRoot);

    cfg.modelDir = resolve(
        modelYaml["model_dir"].as<std::string>(""));

    cfg.cudaStreamPriority =
        modelYaml["cuda_stream_priority"]
            .as<int>(kCudaPriorityLlm);

    cfg.pollTimeout = std::chrono::milliseconds(
        kPollTimeoutMs);

    cfg.degraded = degradedMode;

    // ---- Initialize logger -------------------------------------------------
    OeLogger::instance().setModule("omniedge_conversation");
    OeLogger::instance().initFile(
        OeLogger::resolveLogDir());
    OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "conversation");

    // ---- Load generation defaults from INI --------------------------------
    {
        IniConfig ini;
        if (ini.loadFromFile("config/omniedge.ini")) {
            const auto& gen = ini.generation();
            cfg.generationDefaults.temperature = gen.temperature;
            cfg.generationDefaults.topP        = gen.topP;
            cfg.generationDefaults.maxTokens   = gen.maxTokens;
            cfg.generationDefaults.clamp();
        }
    }

    // ---- Validate config ---------------------------------------------------
    if (auto v = ConversationNode::Config::validate(cfg); !v) {
        spdlog::error("Config validation failed: {}", v.error());
        return 1;
    }

    // ---- Create inferencer based on model variant --------------------------
    std::unique_ptr<ConversationInferencer> inferencer;
    if (modelVariant == kModelQwenOmni7b) {
        inferencer = createQwenOmni7bInferencer();
    } else if (modelVariant == kModelQwenOmni3b) {
        inferencer = createQwenOmni3bInferencer();
    } else if (modelVariant == kModelGemmaE4b) {
        inferencer = createGemmaE4BInferencer();
    } else {
        std::cerr << "[omniedge_conversation] Unknown model variant: " << modelVariant << "\n";
        return 1;
    }

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
    node.run();

    return 0;
}
