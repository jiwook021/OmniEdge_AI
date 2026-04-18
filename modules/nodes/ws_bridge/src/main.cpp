#include "ws_bridge/websocket_bridge_node.hpp"
#include "ws/ws_server.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>

#include "common/ini_config.hpp"
#include "common/oe_module_main.hpp"
#include "common/signal_shutdown.hpp"

namespace {

class WsServerRelay final : public IWsRelay {
public:
	WsServerRelay(WsServer::Config cfg,
	             std::function<void(nlohmann::json)> onCmd,
	             std::function<void(std::string_view)> onBinaryCmd = nullptr)
		: server_(std::move(cfg), std::move(onCmd), std::move(onBinaryCmd))
	{}

	void run() override { server_.run(); }
	void stop() override { server_.stop(); }
	void broadcastVideo(std::vector<uint8_t> data) override
	{
		server_.broadcastVideo(std::move(data));
	}
	void broadcastAudio(std::vector<uint8_t> data) override
	{
		server_.broadcastAudio(std::move(data));
	}
	void broadcastJson(nlohmann::json msg) override
	{
		server_.broadcastJson(std::move(msg));
	}
	void broadcastDenoisedVideo(std::vector<uint8_t> data) override
	{
		server_.broadcastDenoisedVideo(std::move(data));
	}
	void broadcastDenoisedAudio(std::vector<uint8_t> data) override
	{
		server_.broadcastDenoisedAudio(std::move(data));
	}
	void broadcastSam2Video(std::vector<uint8_t> data) override
	{
		server_.broadcastSam2Video(std::move(data));
	}
	void broadcastScreenVideo(std::vector<uint8_t> data) override
	{
		server_.broadcastScreenVideo(std::move(data));
	}
	void broadcastSecurityVideo(std::vector<uint8_t> data) override
	{
		server_.broadcastSecurityVideo(std::move(data));
	}
	void broadcastBeautyVideo(std::vector<uint8_t> data) override
	{
		server_.broadcastBeautyVideo(std::move(data));
	}

private:
	WsServer server_;
};

} // namespace

int main(int argc, char* argv[])
{
	std::string configPath;
	CLI::App app{"omniedge_ws_bridge"};
	app.add_option("--config", configPath, "Path to INI config file")->required();
	CLI11_PARSE(app, argc, argv);

	IniConfig ini;
	if (!ini.loadFromFile(configPath)) {
		std::cerr << "[omniedge_ws_bridge] Failed to load INI: "
		          << configPath << "\n";
		return 1;
	}

	WebSocketBridgeNode::Config cfg;
	WsServer::Config wsCfg;

	cfg.wsBridgePubPort     = ini.port("websocket_bridge", cfg.wsBridgePubPort);
	wsCfg.wsPort            = ini.websocketBridge().wsPort;
	wsCfg.frontendDir       = ini.websocketBridge().frontendDir;
	wsCfg.recordingsDir     = ini.securityCamera().recordingDir;

	cfg.blurSubPort         = ini.port("background_blur",   cfg.blurSubPort);
	cfg.ttsSubPort          = ini.port("tts",               cfg.ttsSubPort);
	cfg.llmSubPort          = ini.port("conversation_model", cfg.llmSubPort);
	cfg.daemonSubPort       = ini.port("daemon",            cfg.daemonSubPort);
	cfg.faceSubPort         = ini.port("face_recognition",  cfg.faceSubPort);
	cfg.videoDenoiseSubPort = ini.port("video_denoise",     cfg.videoDenoiseSubPort);
	cfg.audioDenoiseSubPort = ini.port("audio_denoise",     cfg.audioDenoiseSubPort);
	cfg.sam2SubPort         = ini.port("sam2",              cfg.sam2SubPort);
	cfg.screenIngestSubPort = ini.port("screen_ingest",     cfg.screenIngestSubPort);
	cfg.beautySubPort       = ini.port("beauty",            cfg.beautySubPort);
	// Initialize logger with module name and file output
	oe::main::initLogger("omniedge_ws_bridge", "websocket_bridge");

	// Validate config
	if (auto v = WebSocketBridgeNode::Config::validate(cfg); !v) {
		spdlog::error("Config validation failed: {}", v.error());
		return 1;
	}

	WebSocketBridgeNode node(cfg);
	auto relay = std::make_unique<WsServerRelay>(
		wsCfg,
		[&node](nlohmann::json cmd) {
			node.handleClientCommand(cmd);
		},
		[&node](std::string_view binaryFrame) {
			node.handleBinaryUpload(binaryFrame);
		}
	);
	node.setRelay(std::move(relay));

	try {
		node.initialize();
	} catch (const std::exception& ex) {
		std::cerr << "[omniedge_ws_bridge] initialize failed: " << ex.what() << "\n";
		return 1;
	}

	// Shutdown watcher (detached thread)
	launchShutdownWatcher([&node]() { node.stop(); });

	try {
		node.run();
	} catch (const std::exception& e) {
		spdlog::error("[omniedge_ws_bridge] run() failed: {}", e.what());
		return 1;
	}

	return 0;
}
