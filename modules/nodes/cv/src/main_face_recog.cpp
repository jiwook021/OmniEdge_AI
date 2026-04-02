#include "cv/face_recog_node.hpp"
#include "cv/onnx_face_recog_inferencer.hpp"

#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
#include "common/model_path.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// main_face_recog.cpp — entry point for omniedge_face_recog binary
//
// Supports selectable model variants via YAML config:
//   scrfd_adaface_101  — SCRFD-10G + AdaFace IR-101 (default, max accuracy)
//   scrfd_adaface_50   — SCRFD-10G + AdaFace IR-50  (balanced)
//   scrfd_mobilefacenet — SCRFD-2.5G + MobileFaceNet (min VRAM)
//   inspireface        — legacy InspireFace SDK (requires OE_HAS_INSPIREFACE)
//
// CLI:
//   omniedge_face_recog --config <path/to/omniedge_config.yaml>
// ---------------------------------------------------------------------------

#if defined(OE_HAS_INSPIREFACE)
    [[nodiscard]] std::unique_ptr<FaceRecogInferencer> createInspireFaceInferencer();
#endif

int main(int argc, char* argv[])
{
    std::string configPath;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string_view{argv[i]} == "--config") {
            configPath = argv[i + 1];
        }
    }
    if (configPath.empty()) {
        std::cerr << "Usage: omniedge_face_recog --config <path>\n";
        return 1;
    }

    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(configPath);
    } catch (const YAML::Exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    FaceRecognitionNode::Config cfg;

    const auto modelsRoot = yaml["models_root"].as<std::string>("");
    const auto resolve = makeModelPathResolver(modelsRoot);

    std::string variantStr = "scrfd_adaface_101";  // default

    const auto& fr = yaml["face_recognition"];
    if (fr) {
        variantStr                = fr["model_variant"].as<std::string>(variantStr);
        cfg.modelPackPath         = resolve(fr["model_pack_path"].as<std::string>(cfg.modelPackPath));
        cfg.pubPort               = fr["zmq_pub_port"].as<int>(cfg.pubPort);
        cfg.recognitionThreshold  = fr["recognition_threshold"].as<float>(cfg.recognitionThreshold);
        cfg.knownFacesDb          = fr["faces_db"].as<std::string>(cfg.knownFacesDb);
        cfg.frameSubsample        = fr["frame_subsample"].as<uint32_t>(cfg.frameSubsample);
        cfg.inputShmName          = fr["shm_input"].as<std::string>(cfg.inputShmName);
    }
    const auto& vi = yaml["video_ingest"];
    if (vi) {
        cfg.videoSubPort = vi["pub_port"].as<int>(cfg.videoSubPort);
    }
    const auto& ws = yaml["websocket_bridge"];
    if (ws) {
        cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
    }
    cfg.moduleName = "face_recognition";

    // Initialize logger with module name and file output
    OeLogger::instance().setModule("omniedge_face_recog");
    OeLogger::instance().initFile(
        OeLogger::resolveLogDir());
    OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "face_recognition");

    // Validate config
    if (auto v = FaceRecognitionNode::Config::validate(cfg); !v) {
        spdlog::error("Config validation failed: {}", v.error());
        return 1;
    }

    FaceRecognitionNode node(cfg);

    // --- Select inferencer based on model_variant ---
    std::unique_ptr<FaceRecogInferencer> inferencer;

    if (variantStr == "inspireface") {
#if defined(OE_HAS_INSPIREFACE)
        spdlog::info("face_recog_variant: inspireface (legacy)");
        inferencer = createInspireFaceInferencer();
#else
        spdlog::error("model_variant=inspireface but OE_HAS_INSPIREFACE not compiled. "
                      "Rebuild with InspireFace SDK or choose an ONNX variant.");
        return 1;
#endif
    } else {
        const auto variant = parseFaceRecogVariant(variantStr);
        spdlog::info("face_recog_variant: {} (ONNX Runtime)",
                     faceRecogVariantName(variant));
        inferencer = createOnnxFaceRecogInferencer(variant);
    }

    node.setInferencer(std::move(inferencer));

    try {
        node.initialize();
    } catch (const std::exception& e) {
        std::cerr << "[omniedge_face_recog] Initialize failed: " << e.what() << "\n";
        return 1;
    }

    // Shutdown watcher (detached thread)
    launchShutdownWatcher([&node]() { node.stop(); });

    node.run();

    return 0;
}
