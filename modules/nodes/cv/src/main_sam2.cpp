#include "cv/sam2_node.hpp"
#include "cv/onnx_sam2_inferencer.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_sam2 entry point — SAM2 interactive segmentation
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<Sam2Node>(argc, argv,
		"omniedge_sam2", "sam2",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> Sam2Node::Config {
			Sam2Node::Config cfg;
			cfg.moduleName = "sam2";

			const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
			cfg.videoSubPort    = common.videoSubPort;
			cfg.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& s = ini.sam2();
			cfg.encoderOnnxPath  = common.resolve(s.encoderModel);
			cfg.decoderOnnxPath  = common.resolve(s.decoderModel);
			cfg.inputShmName     = s.shmInput;
			cfg.outputShmName    = s.shmOutput;
			cfg.jpegQuality      = s.jpegQuality;
			cfg.enabledAtStartup = s.enabled;
			cfg.pubPort          = ini.port("sam2", cfg.pubPort);
			return cfg;
		},

		[](Sam2Node& node) {
			node.setInferencer(createOnnxSam2Inferencer());
		}
	);
}
