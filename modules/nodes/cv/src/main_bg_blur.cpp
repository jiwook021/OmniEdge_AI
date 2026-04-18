#include "cv/background_blur_node.hpp"
#include "common/pipeline_types.hpp"

#include <iostream>
#include <string>

#include <CLI/CLI.hpp>

#include "common/ini_config.hpp"
#include "common/oe_module_main.hpp"
#include "common/cv_config_loader.hpp"
#include "common/signal_shutdown.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_bg_blur entry point
//
// Has custom CLI options for pipeline chaining (--shm-input, --input-port,
// --input-topic, --output-format) so cannot use oeNodeMain<> directly.
// ---------------------------------------------------------------------------

// Resolved at link time: MediaPipe Selfie Segmentation (GPU) or CPU stub.
[[nodiscard]] std::unique_ptr<BlurInferencer> createBlurInferencer();

int main(int argc, char* argv[])
{
    // 1. Parse CLI (custom args for pipeline chaining)
    std::string configPath;
    std::string cliShmInput;
    std::string cliInputTopic;
    std::string cliOutputFormat;
    int         cliInputPort = 0;
    CLI::App app{"omniedge_bg_blur"};
    app.add_option("--config", configPath, "Path to INI config file")->required();
    app.add_option("--shm-input", cliShmInput, "Override input SHM name (pipeline chaining)");
    app.add_option("--input-port", cliInputPort, "Override input ZMQ SUB port");
    app.add_option("--input-topic", cliInputTopic, "Override input ZMQ topic");
    app.add_option("--output-format", cliOutputFormat, "Output format: jpeg (default) or bgr24");
    CLI11_PARSE(app, argc, argv);

    // 2. Load config from INI
    IniConfig ini;
    if (!ini.loadFromFile(configPath)) {
        std::cerr << "[omniedge_bg_blur] Failed to load INI: " << configPath << "\n";
        return 1;
    }

    BackgroundBlurNode::Config cfg;
    cfg.moduleName = "background_blur";

    const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
    cfg.videoSubPort    = common.videoSubPort;
    cfg.wsBridgeSubPort = common.wsBridgeSubPort;

    const auto& bg = ini.backgroundBlur();
    cfg.selfieSegModelPath = common.resolve(bg.enginePath);
    cfg.pubPort          = ini.port("background_blur", cfg.pubPort);
    cfg.jpegQuality      = bg.jpegQuality;
    cfg.blurKernelSize   = bg.blurKernelSize;
    cfg.blurSigma        = bg.blurSigma;
    cfg.outputShmName    = bg.shmOutput;
    cfg.inputShmName     = bg.shmInput;
    cfg.inputTopic       = bg.inputTopic;
    cfg.outputBgrShmName = bg.shmOutputBgr;
    cfg.outputFormat     = parseOutputFormat(bg.outputFormat);

    // CLI overrides (pipeline chaining — daemon passes these)
    if (!cliShmInput.empty())     cfg.inputShmName  = cliShmInput;
    if (cliInputPort > 0)         cfg.videoSubPort  = cliInputPort;
    if (!cliInputTopic.empty())   cfg.inputTopic    = cliInputTopic;
    if (!cliOutputFormat.empty()) cfg.outputFormat   = parseOutputFormat(cliOutputFormat);

    // 3. Logger + validation
    oe::main::initLogger("omniedge_bg_blur", "background_blur");
    if (auto v = BackgroundBlurNode::Config::validate(cfg); !v) {
        spdlog::error("Config validation failed: {}", v.error());
        return 1;
    }

    // 4. Create, initialize, run
    BackgroundBlurNode node(cfg);
    node.setInferencer(createBlurInferencer());
    try {
        node.initialize();
    } catch (const std::exception& e) {
        std::cerr << "[omniedge_bg_blur] Initialize failed: " << e.what() << "\n";
        // Exit 78 (EX_CONFIG) signals "dependency unavailable" — the daemon
        // marks the module degraded and does NOT restart. Used when the host
        // OpenCV lacks CUDA support; retrying would just burn spawn cycles.
        return 78;
    }

    launchShutdownWatcher([&node]() { node.stop(); });
    node.run();
    return 0;
}
