#include "cv/background_blur_node.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// Entry point for the omniedge_bg_blur binary.
//
// CLI:  omniedge_bg_blur --config <path/to/omniedge_config.yaml>
//
// Lifecycle:
//   1. Parse CLI → load YAML config
//   2. Create node with TensorRT inferencer
//   3. initialize() — load engine, open SHM, bind ZMQ
//   4. run() — blocking poll loop (processes frames until shutdown)
//   5. SIGTERM/SIGINT → stop() via watcher thread → run() returns → exit
// ---------------------------------------------------------------------------

namespace {

// Load YAML and populate the BackgroundBlurNode::Config struct.
// Uses YAML values when present, falls back to compiled defaults.
BackgroundBlurNode::Config loadConfig(const std::string& configPath)
{
    YAML::Node yaml = YAML::LoadFile(configPath);

    BackgroundBlurNode::Config cfg;
    cfg.moduleName = "background_blur";

    const auto modelsRoot = yaml["models_root"].as<std::string>("");
    const auto resolve = makeModelPathResolver(modelsRoot);

    if (const auto& bg = yaml["background_blur"]) {
        cfg.yolov8EnginePath = resolve(bg["engine_path"].as<std::string>(cfg.yolov8EnginePath));
        cfg.pubPort          = bg["zmq_pub_port"].as<int>(cfg.pubPort);
        cfg.jpegQuality      = bg["jpeg_quality"].as<int>(cfg.jpegQuality);
        cfg.blurKernelSize   = bg["blur_kernel_size"].as<int>(cfg.blurKernelSize);
        cfg.blurSigma        = bg["blur_sigma"].as<float>(cfg.blurSigma);
        cfg.outputShmName    = bg["shm_output"].as<std::string>(cfg.outputShmName);
        cfg.inputShmName     = bg["shm_input"].as<std::string>(cfg.inputShmName);
    }
    if (const auto& vi = yaml["video_ingest"]) {
        cfg.videoSubPort = vi["pub_port"].as<int>(cfg.videoSubPort);
    }
    if (const auto& ws = yaml["websocket_bridge"]) {
        cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
    }

    return cfg;
}

} // anonymous namespace

// Forward-declared factory for the production TensorRT inferencer.
// Defined in tensorrt_blur_inferencer.cpp (built only in production targets).
    [[nodiscard]] std::unique_ptr<BlurInferencer> createTensorRTBlurInferencer();

int main(int argc, char* argv[])
{
    // 1. Parse CLI
    std::string configPath;
    CLI::App app{"omniedge_bg_blur"};
    app.add_option("--config", configPath, "Path to YAML config file")->required();
    CLI11_PARSE(app, argc, argv);

    // 2. Load config from YAML
    BackgroundBlurNode::Config cfg;
    try {
        cfg = loadConfig(configPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    // 3. Set up logging
    OeLogger::instance().setModule("omniedge_bg_blur");
    OeLogger::instance().initFile(
        OeLogger::resolveLogDir());
    OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "background_blur");

    // 3b. Validate config
    if (auto v = BackgroundBlurNode::Config::validate(cfg); !v) {
        spdlog::error("Config validation failed: {}", v.error());
        return 1;
    }

    // 4. Create and initialize the node
    BackgroundBlurNode node(cfg);
    node.setInferencer(createTensorRTBlurInferencer());
    try {
        node.initialize();
    } catch (const std::exception& e) {
        std::cerr << "[omniedge_bg_blur] Initialize failed: " << e.what() << "\n";
        return 1;
    }

    // 5. Shutdown watcher (detached thread)
    launchShutdownWatcher([&node]() { node.stop(); });

    // 6. Run the event loop
    node.run();

    return 0;
}
