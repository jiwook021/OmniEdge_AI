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
		[](const YAML::Node& yaml) -> AudioDenoiseNode::Config {
			const YAML::Node& dnCfg = yaml["audio_denoise"];
			if (!dnCfg) throw std::runtime_error("missing [audio_denoise] section");

			AudioDenoiseNode::Config config;
			const auto modelsRoot = yaml["models_root"].as<std::string>("");
			const auto resolve    = makeModelPathResolver(modelsRoot);

			config.model1Path = resolve(dnCfg["model_stage1"].as<std::string>(config.model1Path));
			config.model2Path = resolve(dnCfg["model_stage2"].as<std::string>(config.model2Path));
			config.shmInput   = dnCfg["shm_input"].as<std::string>(config.shmInput);
			config.shmOutput  = dnCfg["shm_output"].as<std::string>(config.shmOutput);
			config.pubPort    = dnCfg["zmq_pub_port"].as<int>(config.pubPort);

			config.subAudioPort    = yaml["audio_ingest"]["pub_port"].as<int>(config.subAudioPort);
			config.subWsBridgePort = yaml["websocket_bridge"]["zmq_pub_port"].as<int>(config.subWsBridgePort);
			config.cudaStreamPriority = dnCfg["cuda_stream_priority"].as<int>(config.cudaStreamPriority);

			return config;
		},

		// Node factory (inject ONNX inferencer)
		[](AudioDenoiseNode::Config cfg) {
			return AudioDenoiseNode(std::move(cfg), createOnnxDtlnInferencer());
		}
	);
}
