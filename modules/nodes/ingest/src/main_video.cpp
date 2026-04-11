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

		[](const YAML::Node& yaml) -> VideoIngestNode::Config {
			VideoIngestNode::Config cfg;
			if (const auto& vi = yaml["video_ingest"]) {
				cfg.v4l2Device     = vi["v4l2_device"].as<std::string>(cfg.v4l2Device);
				cfg.frameWidth     = vi["frame_width"].as<uint32_t>(cfg.frameWidth);
				cfg.frameHeight    = vi["frame_height"].as<uint32_t>(cfg.frameHeight);
				cfg.pubPort        = vi["pub_port"].as<int>(cfg.pubPort);
				cfg.zmqSendHighWaterMark = vi["zmq_sndhwm"].as<int>(cfg.zmqSendHighWaterMark);
			}
			if (const auto& ws = yaml["websocket_bridge"]) {
				cfg.wsBridgeSubPort = ws["pub_port"].as<int>(cfg.wsBridgeSubPort);
			}
			cfg.moduleName = "video_ingest";
			return cfg;
		}
	);
}
