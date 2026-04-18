#include "cv/beauty_node.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// Resolved at link time: stub or ONNX implementation.
[[nodiscard]] std::unique_ptr<BeautyInferencer> createBeautyInferencer();

// ---------------------------------------------------------------------------
// omniedge_beauty entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<BeautyNode>(argc, argv,
		"omniedge_beauty", "beauty",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> BeautyNode::Config {
			BeautyNode::Config cfg;
			cfg.moduleName = "beauty";

			const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
			cfg.videoSubPort    = common.videoSubPort;
			cfg.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& b = ini.beauty();
			cfg.faceMeshOnnxPath = common.resolve(b.modelPath);
			cfg.pubPort          = ini.port("beauty", cfg.pubPort);
			cfg.jpegQuality      = b.jpegQuality;
			cfg.outputShmName    = b.shmOutput;
			cfg.inputShmName     = b.shmInput;
			cfg.enabledAtStartup = b.enabled;
			return cfg;
		},

		[](BeautyNode& node) {
			node.setInferencer(createBeautyInferencer());
		}
	);
}
