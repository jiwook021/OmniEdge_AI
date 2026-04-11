#include "ingest/audio_ingest_node.hpp"

#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_audio_ingest entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<AudioIngestNode>(argc, argv,
		"omniedge_audio_ingest", "audio_ingest",

		[](const YAML::Node& yaml) -> AudioIngestNode::Config {
			AudioIngestNode::Config cfg;
			if (const auto& ai = yaml["audio_ingest"]) {
				cfg.windowsHostIp        = ai["windows_host_ip"].as<std::string>(cfg.windowsHostIp);
				cfg.audioTcpPort         = ai["tcp_port"].as<int>(cfg.audioTcpPort);
				cfg.sampleRateHz         = ai["sample_rate_hz"].as<uint32_t>(cfg.sampleRateHz);
				cfg.chunkSamples         = ai["chunk_samples"].as<uint32_t>(cfg.chunkSamples);
				cfg.zmqSendHighWaterMark = ai["zmq_sndhwm"].as<int>(cfg.zmqSendHighWaterMark);
				cfg.pubPort              = ai["pub_port"].as<int>(cfg.pubPort);
			}
			const auto modelsRoot = yaml["models_root"].as<std::string>("");
			const auto resolve    = makeModelPathResolver(modelsRoot);
			if (const auto& vad = yaml["audio_ingest"]["vad"]) {
				cfg.vadModelPath       = resolve(vad["model_path"].as<std::string>(cfg.vadModelPath));
				cfg.vadSpeechThreshold = vad["speech_threshold"].as<float>(cfg.vadSpeechThreshold);
				cfg.silenceDurationMs  = vad["silence_duration_ms"].as<uint32_t>(cfg.silenceDurationMs);
			}
			if (const auto& ws = yaml["websocket_bridge"]) {
				cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
			}
			cfg.moduleName = "audio_ingest";
			return cfg;
		}
	);
}
