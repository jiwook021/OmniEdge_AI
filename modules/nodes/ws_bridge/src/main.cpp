#include "ws_bridge/websocket_bridge_node.hpp"
#include "ws/ws_server.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

#include "common/oe_logger.hpp"
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

private:
	WsServer server_;
};

} // namespace

int main(int argc, char* argv[])
{
	std::string configPath;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--config") == 0 && (i + 1) < argc) {
			configPath = argv[++i];
		}
	}
	if (configPath.empty()) {
		std::cerr << "Usage: omniedge_ws_bridge --config <path>\n";
		return 1;
	}

	YAML::Node yaml;
	try {
		yaml = YAML::LoadFile(configPath);
	} catch (const YAML::Exception& ex) {
		std::cerr << "Failed to parse config: " << ex.what() << "\n";
		return 1;
	}

	WebSocketBridgeNode::Config cfg;
	WsServer::Config wsCfg;

	const auto ws = yaml["websocket_bridge"];
	if (ws) {
		cfg.wsBridgePubPort = ws["zmq_pub_port"].as<int>(cfg.wsBridgePubPort);
		wsCfg.wsPort = ws["ws_port"].as<int>(wsCfg.wsPort);
		wsCfg.frontendDir = ws["frontend_dir"].as<std::string>("./frontend");
	}

	// Subscribe to module PUB ports — key is zmq_pub_port for AI modules,
	// pub_port for ingest modules (video_ingest, audio_ingest).
	if (yaml["background_blur"] && yaml["background_blur"]["zmq_pub_port"]) {
		cfg.blurSubPort = yaml["background_blur"]["zmq_pub_port"].as<int>(cfg.blurSubPort);
	}
	{
		const auto ttsNode = yaml["tts"] ? yaml["tts"] : yaml["kokoro_tts"];
		if (ttsNode && ttsNode["zmq_pub_port"]) {
			cfg.ttsSubPort = ttsNode["zmq_pub_port"].as<int>(cfg.ttsSubPort);
		}
	}
	{
		const auto llmNode = yaml["llm"] ? yaml["llm"] : yaml["qwen_llm"];
		if (llmNode && llmNode["zmq_pub_port"]) {
			cfg.llmSubPort = llmNode["zmq_pub_port"].as<int>(cfg.llmSubPort);
		}
	}
	if (yaml["daemon"] && yaml["daemon"]["zmq_pub_port"]) {
		cfg.daemonSubPort = yaml["daemon"]["zmq_pub_port"].as<int>(cfg.daemonSubPort);
	}
	if (yaml["face_recognition"] && yaml["face_recognition"]["zmq_pub_port"]) {
		cfg.faceSubPort = yaml["face_recognition"]["zmq_pub_port"].as<int>(cfg.faceSubPort);
	}
	if (yaml["video_denoise"] && yaml["video_denoise"]["zmq_pub_port"]) {
		cfg.videoDenoiseSubPort = yaml["video_denoise"]["zmq_pub_port"].as<int>(cfg.videoDenoiseSubPort);
	}
	if (yaml["audio_denoise"] && yaml["audio_denoise"]["zmq_pub_port"]) {
		cfg.audioDenoiseSubPort = yaml["audio_denoise"]["zmq_pub_port"].as<int>(cfg.audioDenoiseSubPort);
	}
	// Initialize logger with module name and file output
	OeLogger::instance().setModule("omniedge_ws_bridge");
	OeLogger::instance().initFile(
		OeLogger::resolveLogDir());
	OeLogger::instance().applyLogLevelFromIni("config/omniedge.ini", "websocket_bridge");

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

	node.run();

	return 0;
}
