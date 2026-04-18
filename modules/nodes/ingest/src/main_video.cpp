#include "ingest/video_ingest_node.hpp"

#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_video_ingest entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	return oeNodeMain<VideoIngestNode>(argc, argv,
		"omniedge_video_ingest", "video_ingest",

		[](const YAML::Node& /*yaml*/, const IniConfig& ini) -> VideoIngestNode::Config {
			VideoIngestNode::Config cfg;
			const auto& vi = ini.videoIngest();

			cfg.v4l2Device           = vi.v4l2Device;
			cfg.frameWidth           = static_cast<uint32_t>(vi.frameWidth);
			cfg.frameHeight          = static_cast<uint32_t>(vi.frameHeight);
			cfg.zmqSendHighWaterMark = vi.zmqSendHighWaterMark;

			cfg.pubPort         = ini.port("video_ingest",     cfg.pubPort);
			cfg.wsBridgeSubPort = ini.port("websocket_bridge", cfg.wsBridgeSubPort);
			cfg.moduleName      = "video_ingest";
			return cfg;
		}
	);
}
