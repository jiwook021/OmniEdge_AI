#include "cv/security_camera_node.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_security_camera entry point — YOLO detection + NVENC recording
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<SecurityCameraNode>(argc, argv,
		"omniedge_security_camera", "security_camera",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> SecurityCameraNode::Config {
			SecurityCameraNode::Config cfg;
			cfg.moduleName = "security_camera";

			const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
			cfg.videoSubPort    = common.videoSubPort;
			cfg.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& sec = ini.securityCamera();
			cfg.yoloEnginePath       = common.resolve(sec.enginePath);
			cfg.pubPort              = ini.port("security_camera", cfg.pubPort);
			cfg.jpegQuality          = sec.jpegQuality;
			cfg.confidenceThreshold  = sec.confidenceThreshold;
			cfg.detectionIntervalMs  = sec.detectionIntervalMs;
			cfg.eventCooldownMs      = sec.eventCooldownMs;
			cfg.recordingDir         = sec.recordingDir;
			cfg.segmentDurationMin   = sec.segmentDurationMin;
			cfg.recordingFps         = sec.recordingFps;
			cfg.recordingBitrate     = sec.recordingBitrate;
			cfg.outputShmName        = sec.shmOutput;
			cfg.inputShmName         = sec.shmInput;
			cfg.targetClasses        = sec.targetClasses;

			const auto& vi = ini.videoIngest();
			cfg.inputWidth  = static_cast<uint32_t>(vi.frameWidth);
			cfg.inputHeight = static_cast<uint32_t>(vi.frameHeight);
			return cfg;
		}
	);
}
