#include "cv/video_denoise_node.hpp"

#include "common/cv_config_loader.hpp"
#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_video_denoise entry point — BasicVSR++ temporal video enhancement
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<VideoDenoiseNode>(argc, argv,
		"omniedge_video_denoise", "video_denoise",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> VideoDenoiseNode::Config {
			VideoDenoiseNode::Config config;

			const auto common = loadCvCommonFromIni(ini, config.videoSubPort, config.wsBridgeSubPort);
			config.videoSubPort    = common.videoSubPort;
			config.wsBridgeSubPort = common.wsBridgeSubPort;

			const auto& vd = ini.videoDenoise();
			config.onnxModelPath      = common.resolve(vd.onnxModel);
			config.inputShmName       = vd.shmInput;
			config.outputShmName      = vd.shmOutput;
			config.pubPort            = ini.port("video_denoise", config.pubPort);
			config.temporalWindowSize = static_cast<uint32_t>(vd.temporalWindow);
			config.cudaStreamPriority = vd.cudaStreamPriority;
			config.moduleName         = "video_denoise";
			return config;
		},

		[](VideoDenoiseNode::Config cfg) {
			return VideoDenoiseNode(std::move(cfg), createOnnxBasicVsrppInferencer());
		}
	);
}
