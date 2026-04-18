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

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> TTSNode::Config {
			TTSNode::Config config;
			const auto resolve = makeModelPathResolver(ini.paths().modelsRoot);
			const auto& tts    = ini.tts();

			config.onnxModelPath      = resolve(tts.onnxModel);
			config.voiceDir           = resolve(tts.voiceDir);
			config.defaultVoice       = tts.defaultVoice;
			config.speed              = tts.speed;
			config.shmOutput          = tts.shmOutput;
			config.cudaStreamPriority = tts.cudaStreamPriority;

			config.pubPort       = ini.port("tts",    config.pubPort);
			config.subLlmPort    = ini.port("conversation_model", config.subLlmPort);
			config.subDaemonPort = ini.port("daemon", config.subDaemonPort);
			return config;
		},

		[](TTSNode::Config cfg) {
			return TTSNode(std::move(cfg), createOnnxKokoroInferencer());
		}
	);
}
