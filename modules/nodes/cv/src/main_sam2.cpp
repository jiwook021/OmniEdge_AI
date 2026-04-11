#include "cv/sam2_node.hpp"
#include "cv/onnx_sam2_inferencer.hpp"
#include "common/oe_tracy.hpp"
#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"

#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// omniedge_sam2 — SAM2 (Segment Anything Model 2) interactive segmentation
//
// Reads camera frames from /oe.vid.ingest, receives prompts (point/box/mask)
// from the UI via ZMQ, produces pixel-perfect segmentation masks, and
// publishes JPEG overlays on port 5576.
//
// Usage:
//   omniedge_sam2 --config config/omniedge_config.yaml
// ---------------------------------------------------------------------------

namespace {

Sam2Node::Config loadConfig(const std::string& configPath)
{
    YAML::Node yaml = YAML::LoadFile(configPath);

    Sam2Node::Config cfg;
    cfg.moduleName = "sam2";

    const auto modelsRoot = yaml["models_root"].as<std::string>("");
    const auto resolve = makeModelPathResolver(modelsRoot);

    if (const auto& sam2 = yaml["sam2"]) {
        if (sam2["encoder_model"])
            cfg.encoderOnnxPath = resolve(sam2["encoder_model"].as<std::string>());
        if (sam2["decoder_model"])
            cfg.decoderOnnxPath = resolve(sam2["decoder_model"].as<std::string>());
        if (sam2["shm_input"])
            cfg.inputShmName = sam2["shm_input"].as<std::string>();
        if (sam2["shm_output"])
            cfg.outputShmName = sam2["shm_output"].as<std::string>();
        if (sam2["jpeg_quality"])
            cfg.jpegQuality = sam2["jpeg_quality"].as<int>();
        if (sam2["enabled"])
            cfg.enabledAtStartup = sam2["enabled"].as<bool>();
        if (sam2["zmq_pub_port"])
            cfg.pubPort = sam2["zmq_pub_port"].as<int>();
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

int main(int argc, char* argv[])
{
    // 1. Parse CLI
    std::string configPath;
    CLI::App app{"omniedge_sam2"};
    app.add_option("--config", configPath, "Path to YAML config file")->required();
    CLI11_PARSE(app, argc, argv);

    // 2. Load config from YAML
    Sam2Node::Config cfg;
    try {
        cfg = loadConfig(configPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    // 3. Set up logging
    oe::main::initLogger("omniedge_sam2", "sam2");

    // 3b. Validate config
    auto validated = Sam2Node::Config::validate(cfg);
    if (!validated.has_value()) {
        spdlog::error("Config validation failed: {}", validated.error());
        return 1;
    }

    // 4. Create and initialize the node
    Sam2Node node(validated.value());
    node.setInferencer(createOnnxSam2Inferencer());
    try {
        node.initialize();
    } catch (const std::exception& e) {
        std::cerr << "[omniedge_sam2] Initialize failed: " << e.what() << "\n";
        return 1;
    }

    // 5. Shutdown watcher (detached thread)
    launchShutdownWatcher([&node]() { node.stop(); });

    // 6. Run the event loop
    node.run();

    return 0;
}
