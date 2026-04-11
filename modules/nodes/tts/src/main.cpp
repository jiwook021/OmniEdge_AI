#include "tts/tts_node.hpp"

#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_tts entry point
// ---------------------------------------------------------------------------

// Forward-declared factory (defined in onnx_kokoro_inferencer.cpp)
[[nodiscard]] std::unique_ptr<TTSInferencer> createOnnxKokoroInferencer();

int main(int argc, char* argv[])
{
	return oeNodeMain<TTSNode>(argc, argv,
		"omniedge_tts", "tts",

		[](const YAML::Node& yaml) -> TTSNode::Config {
			const YAML::Node& ttsYaml = yaml["tts"] ? yaml["tts"] : yaml["kokoro_tts"];
			if (!ttsYaml) throw std::runtime_error("missing [tts] section");

			TTSNode::Config config;
			const auto modelsRoot = yaml["models_root"].as<std::string>("");
			const auto resolve    = makeModelPathResolver(modelsRoot);

			config.onnxModelPath = resolve(ttsYaml["onnx_model"].as<std::string>(config.onnxModelPath));
			config.voiceDir      = resolve(ttsYaml["voice_dir"].as<std::string>(config.voiceDir));
			config.defaultVoice  = ttsYaml["default_voice"].as<std::string>(config.defaultVoice);
			config.speed         = ttsYaml["speed"].as<float>(config.speed);
			config.shmOutput     = ttsYaml["shm_output"].as<std::string>(config.shmOutput);
			config.pubPort       = ttsYaml["zmq_pub_port"].as<int>(config.pubPort);

			const YAML::Node& llmCfg = yaml["llm"] ? yaml["llm"] : yaml["qwen_llm"];
			if (llmCfg) {
				config.subLlmPort = llmCfg["zmq_pub_port"].as<int>(config.subLlmPort);
			}
			config.subDaemonPort      = yaml["daemon"]["zmq_pub_port"].as<int>(config.subDaemonPort);
			config.cudaStreamPriority = ttsYaml["cuda_stream_priority"].as<int>(config.cudaStreamPriority);
			return config;
		},

		[](TTSNode::Config cfg) {
			return TTSNode(std::move(cfg), createOnnxKokoroInferencer());
		}
	);
}
