#include "ingest/screen_ingest_node.hpp"

#include "common/oe_node_main.hpp"
#include "common/oe_tracy.hpp"

// ---------------------------------------------------------------------------
// omniedge_screen_ingest entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    return oeNodeMain<ScreenIngestNode>(argc, argv,
        "omniedge_screen_ingest", "screen_ingest",

        [](const YAML::Node& /*yaml*/, const IniConfig& ini) -> ScreenIngestNode::Config {
            ScreenIngestNode::Config cfg;
            const auto& si = ini.screenIngest();
            cfg.windowsHostIp    = si.windowsHostIp;
            cfg.tcpPort          = si.tcpPort;
            cfg.healthTimeoutSec = si.healthTimeoutSec;
            cfg.pubPort          = ini.port("screen_ingest",   cfg.pubPort);
            cfg.wsBridgeSubPort  = ini.port("websocket_bridge", cfg.wsBridgeSubPort);
            cfg.moduleName       = "screen_ingest";
            return cfg;
        }
    );
}
