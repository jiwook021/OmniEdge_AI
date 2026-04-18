#include "cv/security_vlm_node.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_security_vlm entry point — LP agent loop on top of security_camera.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    return oeNodeMain<SecurityVlmNode>(argc, argv,
        "omniedge_security_vlm", "security_vlm",

        [](const YAML::Node& /*yaml*/, const IniConfig& ini) -> SecurityVlmNode::Config {
            SecurityVlmNode::Config cfg;
            cfg.moduleName = "security_vlm";

            const auto common = loadCvCommonFromIni(
                ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
            cfg.videoSubPort    = common.videoSubPort;
            cfg.wsBridgeSubPort = common.wsBridgeSubPort;

            const auto& vlm = ini.securityVlm();
            cfg.pubPort              = ini.port("security_vlm", cfg.pubPort);
            cfg.securityEventSubPort = ini.port("security_camera", cfg.securityEventSubPort);
            cfg.modelDir             = common.resolve(vlm.modelDir);
            cfg.scriptPathOverride   = vlm.scriptPathOverride;
            cfg.prerollSec           = vlm.prerollSec;
            cfg.postrollSec          = vlm.postrollSec;
            cfg.frameSamples         = vlm.frameSamples;
            cfg.ingestFps            = vlm.ingestFps;
            cfg.jpegDownscale        = vlm.jpegDownscale;
            cfg.jpegQuality          = vlm.jpegQuality;
            cfg.startupTimeoutSec    = vlm.startupTimeoutSec;
            cfg.requestTimeoutMs     = vlm.requestTimeoutMs;
            cfg.idleUnloadSec        = vlm.idleUnloadSec;
            cfg.maxPendingEvents     = vlm.maxPendingEvents;
            cfg.logDir               = vlm.logDir;
            if (!vlm.promptTemplate.empty()) {
                cfg.promptTemplate = vlm.promptTemplate;
            }

            const auto& vi = ini.videoIngest();
            cfg.inputWidth  = static_cast<uint32_t>(vi.frameWidth);
            cfg.inputHeight = static_cast<uint32_t>(vi.frameHeight);
            return cfg;
        }
    );
}
