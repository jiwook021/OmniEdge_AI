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

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> AudioIngestNode::Config {
			AudioIngestNode::Config cfg;
			const auto  resolve = makeModelPathResolver(ini.paths().modelsRoot);
			const auto& ai      = ini.audioIngest();

			cfg.audioSource          = ai.audioSource;
			cfg.windowsHostIp        = ai.windowsHostIp;
			cfg.audioTcpPort         = ai.tcpPort;
			cfg.sampleRateHz         = static_cast<uint32_t>(ai.sampleRateHz);
			cfg.chunkSamples         = static_cast<uint32_t>(ai.chunkSamples);
			cfg.zmqSendHighWaterMark = ai.zmqSendHighWaterMark;
			cfg.vadModelPath         = resolve(ai.vadModelPath);
			cfg.vadSpeechThreshold   = ai.vadSpeechThreshold;
			cfg.silenceDurationMs    = static_cast<uint32_t>(ai.vadSilenceDurationMs);

			cfg.pubPort         = ini.port("audio_ingest",     cfg.pubPort);
			cfg.wsBridgeSubPort = ini.port("websocket_bridge", cfg.wsBridgeSubPort);
			cfg.moduleName      = "audio_ingest";
			return cfg;
		}
	);
}
