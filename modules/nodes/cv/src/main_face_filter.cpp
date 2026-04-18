#include "cv/face_filter_node.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// Resolved at link time: stub or ONNX implementation.
[[nodiscard]] std::unique_ptr<FaceFilterInferencer> createStubFaceFilterInferencer();

// ---------------------------------------------------------------------------
// omniedge_face_filter entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<FaceFilterNode>(argc, argv,
		"omniedge_face_filter", "face_filter",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> FaceFilterNode::Config {
			FaceFilterNode::Config cfg;
			cfg.moduleName = "face_filter";

			const auto common = loadCvCommonFromIni(ini, cfg.videoSubPort, cfg.wsBridgeSubPort);
			cfg.videoSubPort    = common.videoSubPort;
			cfg.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& ff = ini.faceFilter();
			cfg.faceMeshOnnxPath   = common.resolve(ff.modelPath);
			cfg.filterManifestPath = ff.filterManifest;
			cfg.pubPort            = ini.port("face_filter", cfg.pubPort);
			cfg.jpegQuality        = ff.jpegQuality;
			cfg.outputShmName      = ff.shmOutput;
			cfg.inputShmName       = ff.shmInput;
			cfg.activeFilterId     = ff.activeFilter;
			cfg.enabledAtStartup   = ff.enabled;
			return cfg;
		},

		[](FaceFilterNode& node) {
			node.setInferencer(createStubFaceFilterInferencer());
		}
	);
}
