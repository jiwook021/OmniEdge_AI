#include "cv/sam2_node.hpp"
#include "cv/onnx_sam2_inferencer.hpp"
#include "common/oe_tracy.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <yaml-cpp/yaml.h>

#include <csignal>
#include <cstdlib>
#include <filesystem>
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

static std::atomic<bool> g_running{true};

static void signalHandler(int sig) {
    spdlog::info("[SAM2] Received signal {}, shutting down...", sig);
    g_running.store(false, std::memory_order_release);
}

int main(int argc, char* argv[])
{
    // Parse --config argument
    std::string configPath = "config/omniedge_config.yaml";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--config") {
            configPath = argv[i + 1];
        }
    }

    // Set up logging
    try {
        auto logger = spdlog::basic_logger_mt("omniedge_sam2", "logs/omniedge.log", true);
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::debug);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log init failed: " << ex.what() << std::endl;
    }

    spdlog::info("[SAM2] Starting omniedge_sam2...");
    spdlog::info("[SAM2] Config: {}", configPath);

    // Install signal handlers
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGINT, signalHandler);

    // Load YAML config
    Sam2Node::Config nodeCfg;
    try {
        if (std::filesystem::exists(configPath)) {
            YAML::Node yaml = YAML::LoadFile(configPath);
            if (yaml["sam2"]) {
                auto sam2 = yaml["sam2"];
                if (sam2["encoder_model"]) {
                    const std::string modelsRoot = yaml["models_root"]
                        ? yaml["models_root"].as<std::string>()
                        : std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/omniedge_models";
                    nodeCfg.encoderOnnxPath = modelsRoot + "/" + sam2["encoder_model"].as<std::string>();
                }
                if (sam2["decoder_model"])
                    nodeCfg.decoderOnnxPath = yaml["models_root"].as<std::string>("") + "/" + sam2["decoder_model"].as<std::string>();
                if (sam2["shm_input"])
                    nodeCfg.inputShmName = sam2["shm_input"].as<std::string>();
                if (sam2["shm_output"])
                    nodeCfg.outputShmName = sam2["shm_output"].as<std::string>();
                if (sam2["jpeg_quality"])
                    nodeCfg.jpegQuality = sam2["jpeg_quality"].as<int>();
                if (sam2["enabled"])
                    nodeCfg.enabledAtStartup = sam2["enabled"].as<bool>();
                if (sam2["zmq_pub_port"])
                    nodeCfg.pubPort = sam2["zmq_pub_port"].as<int>();
            }
        }
    } catch (const YAML::Exception& e) {
        spdlog::warn("[SAM2] YAML parse error: {} — using defaults", e.what());
    }

    // Validate config
    auto validated = Sam2Node::Config::validate(nodeCfg);
    if (!validated.has_value()) {
        spdlog::error("[SAM2] Config validation failed: {}", validated.error());
        return EXIT_FAILURE;
    }

    // Create node and inferencer
    Sam2Node node(validated.value());
    node.setInferencer(createOnnxSam2Inferencer());

    // Initialize (loads model, sets up ZMQ/SHM)
    try {
        node.initialize();
    } catch (const std::exception& e) {
        spdlog::error("[SAM2] Initialization failed: {}", e.what());
        return EXIT_FAILURE;
    }

    spdlog::info("[SAM2] Initialized successfully, entering run loop...");

    // Run (blocks until stop() is called)
    node.run();

    spdlog::info("[SAM2] Shutdown complete");
    return EXIT_SUCCESS;
}
