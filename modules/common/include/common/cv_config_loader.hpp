#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — CV common-config loader
//
// Every omniedge_* CV binary reads the same three fields from the INI
// config: models_root, video_ingest.pub_port, websocket_bridge.pub_port.
// ---------------------------------------------------------------------------

#include <functional>
#include <string>

#include "common/ini_config.hpp"
#include "common/model_path.hpp"


struct CvCommonConfig {
	int                                            videoSubPort{0};
	int                                            wsBridgeSubPort{0};
	std::string                                    modelsRoot;
	std::function<std::string(const std::string&)> resolve;
};


/// Reads models_root from [paths] and transport ports from
/// [ports] video_ingest / websocket_bridge.
[[nodiscard]] inline CvCommonConfig
loadCvCommonFromIni(const IniConfig& ini,
                    int              videoDefault,
                    int              wsDefault)
{
	CvCommonConfig cfg;
	cfg.videoSubPort    = ini.port("video_ingest",     videoDefault);
	cfg.wsBridgeSubPort = ini.port("websocket_bridge", wsDefault);
	cfg.modelsRoot      = ini.paths().modelsRoot;
	cfg.resolve         = makeModelPathResolver(cfg.modelsRoot);
	return cfg;
}
