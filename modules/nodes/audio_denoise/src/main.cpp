#include "audio_denoise/audio_denoise_node.hpp"

#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_audio_denoise entry point
// ---------------------------------------------------------------------------

// Forward-declared factory (defined in onnx_dtln_inferencer.cpp)
[[nodiscard]] std::unique_ptr<AudioDenoiseInferencer> createOnnxDtlnInferencer();

int main(int argc, char* argv[])
{
	return oeNodeMain<AudioDenoiseNode>(argc, argv,
		"omniedge_audio_denoise", "audio_denoise",

		// Config loader
		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> AudioDenoiseNode::Config {
			AudioDenoiseNode::Config config;
			const auto  resolve = makeModelPathResolver(ini.paths().modelsRoot);
			const auto& dn      = ini.audioDenoise();

			config.model1Path         = resolve(dn.modelStage1);
			config.model2Path         = resolve(dn.modelStage2);
			config.shmInput           = dn.shmInput;
			config.shmOutput          = dn.shmOutput;
			config.cudaStreamPriority = dn.cudaStreamPriority;

			config.pubPort         = ini.port("audio_denoise",    config.pubPort);
			config.subAudioPort    = ini.port("audio_ingest",     config.subAudioPort);
			config.subWsBridgePort = ini.port("websocket_bridge", config.subWsBridgePort);
			return config;
		},

		// Node factory (inject ONNX inferencer)
		[](AudioDenoiseNode::Config cfg) {
			return AudioDenoiseNode(std::move(cfg), createOnnxDtlnInferencer());
		}
	);
}
