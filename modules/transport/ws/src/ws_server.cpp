// ws_server.cpp — WsServer implementation
//
// uWebSockets is single-threaded.  All WebSocket operations (send, close,
// iteration over client lists) MUST run on the event-loop thread.  Methods
// called from other threads (broadcastVideo, broadcastAudio, broadcastJson,
// stop) use loop_->defer() to marshal work onto the event-loop thread.

#include "ws/ws_server.hpp"

#include <algorithm>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// uWebSockets must be included before any other project headers that may
// pull in conflicting POSIX socket definitions.
#include <App.h>

#include "common/file.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Mime-type lookup for static file serving
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] std::string_view mimeType(std::string_view path) noexcept
{
	// Map common web extensions to MIME types.
	if (path.ends_with(".html")) return "text/html; charset=utf-8";
	if (path.ends_with(".js"))   return "application/javascript";
	if (path.ends_with(".css"))  return "text/css";
	if (path.ends_with(".ico"))  return "image/x-icon";
	if (path.ends_with(".png"))  return "image/png";
	if (path.ends_with(".svg"))  return "image/svg+xml";
	if (path.ends_with(".json")) return "application/json";
	return "application/octet-stream";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// WsServer::Impl — holds all uWS types (complete definition in .cpp)
// ---------------------------------------------------------------------------

struct WsServer::Impl {
	using WsHandle = uWS::WebSocket<false, true, PerSocketData>;

	Config                              config;
	std::function<void(nlohmann::json)> onCmd;
	std::function<void(std::string_view)> onBinaryCmd;

	std::unique_ptr<uWS::App> app;
	uWS::Loop*           loop         = nullptr;
	us_listen_socket_t*  listenSocket = nullptr;

	// Client lists — only touched on the event-loop thread.
	std::vector<WsHandle*> videoClients;
	std::vector<WsHandle*> audioClients;
	std::vector<WsHandle*> chatClients;
	std::vector<WsHandle*> denoiseVideoClients;
	std::vector<WsHandle*> denoiseAudioClients;
	std::vector<WsHandle*> filterVideoClients;

	std::atomic<bool> running{false};
};

// ---------------------------------------------------------------------------
// WsServer — public API
// ---------------------------------------------------------------------------

WsServer::WsServer(Config                              config,
                   std::function<void(nlohmann::json)>  onCmd,
                   std::function<void(std::string_view)> onBinaryCmd)
	: impl_(std::make_unique<Impl>())
{
	impl_->config     = std::move(config);
	impl_->onCmd      = std::move(onCmd);
	impl_->onBinaryCmd = std::move(onBinaryCmd);

	// Capture the current thread's event loop before run() blocks it.
	// Both the constructor and run() must be called on the same thread.
	impl_->loop = uWS::Loop::get();

	impl_->app = std::make_unique<uWS::App>();

	// ── /video endpoint: binary JPEG/PNG frames, server → client only ──────
	impl_->app->ws<PerSocketData>("/video", {
		.compression = uWS::DISABLED,  // frames are already compressed (JPEG/PNG)
		.maxPayloadLength = 0,          // no inbound limit (server sends only)
		.idleTimeout = 0,               // no idle timeout for streaming clients
		.open = [this](Impl::WsHandle* ws) {
			impl_->videoClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=video, clients={}",
				static_cast<int>(impl_->videoClients.size()));
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->videoClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=video, code={}", code);
		}
	});

	// ── /audio endpoint: binary F32 PCM, server → client only ───────────────
	impl_->app->ws<PerSocketData>("/audio", {
		.compression = uWS::DISABLED,  // raw PCM; compression adds latency
		.maxPayloadLength = 0,
		.idleTimeout = 0,
		.open = [this](Impl::WsHandle* ws) {
			impl_->audioClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=audio, clients={}",
				static_cast<int>(impl_->audioClients.size()));
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->audioClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=audio, code={}", code);
		}
	});

	// ── /denoise_video endpoint: denoised JPEG frames (BasicVSR++), server → client
	impl_->app->ws<PerSocketData>("/denoise_video", {
		.compression = uWS::DISABLED,
		.maxPayloadLength = 0,
		.idleTimeout = 0,
		.open = [this](Impl::WsHandle* ws) {
			impl_->denoiseVideoClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=denoise_video, clients={}",
				static_cast<int>(impl_->denoiseVideoClients.size()));
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->denoiseVideoClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=denoise_video, code={}", code);
		}
	});

	// ── /denoise_audio endpoint: denoised PCM (DTLN), server → client ────
	impl_->app->ws<PerSocketData>("/denoise_audio", {
		.compression = uWS::DISABLED,
		.maxPayloadLength = 0,
		.idleTimeout = 0,
		.open = [this](Impl::WsHandle* ws) {
			impl_->denoiseAudioClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=denoise_audio, clients={}",
				static_cast<int>(impl_->denoiseAudioClients.size()));
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->denoiseAudioClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=denoise_audio, code={}", code);
		}
	});

	// ── /filter_video endpoint: filtered JPEG (FaceFilterNode), server → client
	impl_->app->ws<PerSocketData>("/filter_video", {
		.compression = uWS::DISABLED,
		.maxPayloadLength = 0,
		.idleTimeout = 0,
		.open = [this](Impl::WsHandle* ws) {
			impl_->filterVideoClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=filter_video, clients={}",
				static_cast<int>(impl_->filterVideoClients.size()));
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->filterVideoClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=filter_video, code={}", code);
		}
	});

	// ── /chat endpoint: bidirectional JSON + binary image upload ────────────
	impl_->app->ws<PerSocketData>("/chat", {
		.compression = uWS::SHARED_COMPRESSOR,  // text payloads compress well
		.maxPayloadLength = 11u * 1024u * 1024u, // 11 MiB — supports image uploads up to 10 MiB + header
		.idleTimeout = 0,
		.open = [this](Impl::WsHandle* ws) {
			impl_->chatClients.push_back(ws);
			OE_LOG_INFO("ws_client_connected: channel=chat, clients={}",
				static_cast<int>(impl_->chatClients.size()));
		},
		.message = [this](Impl::WsHandle* ws, std::string_view rawMsg, uWS::OpCode opCode) {
			if (opCode == uWS::OpCode::TEXT) {
				// JSON ui_command
				try {
					nlohmann::json cmd = nlohmann::json::parse(rawMsg);
					impl_->onCmd(std::move(cmd));
				} catch (const nlohmann::json::exception& ex) {
					OE_LOG_WARN("ws_chat_parse_error: error={}, raw_len={}",
						ex.what(), static_cast<int>(rawMsg.size()));
					nlohmann::json errMsg = {
						{"v",       1},
						{"type",    "error"},
						{"message", std::format("JSON parse error: {}", ex.what())},
					};
					ws->send(errMsg.dump(), uWS::OpCode::TEXT);
				}
			} else if (opCode == uWS::OpCode::BINARY) {
				// Binary image upload: [4B json_len LE][JSON header][image bytes]
				if (impl_->onBinaryCmd) {
					impl_->onBinaryCmd(rawMsg);
				}
			}
		},
		.close = [this](Impl::WsHandle* ws, int code, std::string_view /*msg*/) {
			auto& v = impl_->chatClients;
			v.erase(std::remove(v.begin(), v.end(), ws), v.end());
			OE_LOG_INFO("ws_client_disconnected: channel=chat, code={}", code);
		}
	});

	// ── HTTP GET /status — JSON health document ──────────────────────────────
	impl_->app->get("/status", [this](uWS::HttpResponse<false>* res,
	                                   uWS::HttpRequest* /*req*/) {
		nlohmann::json doc = {
			{"v",             1},
			{"ws_port",       impl_->config.wsPort},
			{"video_clients",          static_cast<int>(impl_->videoClients.size())},
			{"audio_clients",          static_cast<int>(impl_->audioClients.size())},
			{"chat_clients",           static_cast<int>(impl_->chatClients.size())},
			{"denoise_video_clients",  static_cast<int>(impl_->denoiseVideoClients.size())},
			{"denoise_audio_clients",  static_cast<int>(impl_->denoiseAudioClients.size())},
			{"filter_video_clients",   static_cast<int>(impl_->filterVideoClients.size())},
		};
		res->writeHeader("Content-Type", "application/json")
		   ->writeHeader("Cache-Control", "no-store")
		   ->end(doc.dump());
	});

	// ── HTTP GET /* — static file serving from frontendDir ──────────────────
	impl_->app->get("/*", [this](uWS::HttpResponse<false>* res,
	                               uWS::HttpRequest* req) {
		std::string_view urlPath = req->getUrl();
		// Map "/" to "/index.html".
		std::string relPath = (urlPath == "/") ? "/index.html" : std::string(urlPath);

		// ── Path traversal protection ──────────────────────────────────────
		// Reject any path containing ".." to prevent directory traversal
		// attacks (e.g., GET /../../../etc/passwd).
		if (relPath.find("..") != std::string::npos) {
			res->writeStatus("403 Forbidden")
			   ->writeHeader("Content-Type", "text/plain")
			   ->end("403 Forbidden");
			return;
		}

		std::string filePath = impl_->config.frontendDir + relPath;

		auto fileResult = readText(filePath);
		if (!fileResult) {
			res->writeStatus("404 Not Found")
			   ->writeHeader("Content-Type", "text/plain")
			   ->end("404 Not Found");
			return;
		}

		res->writeHeader("Content-Type", mimeType(filePath))
		   ->writeHeader("Cache-Control", "no-store")
		   ->end(*fileResult);
	});
}

WsServer::~WsServer()
{
	// Ensure the event loop is stopped before the Impl is destroyed.
	stop();
	// impl_ unique_ptr destructs here; unique_ptr<uWS::App> handles cleanup.
}

void WsServer::run()
{
	impl_->running.store(true, std::memory_order_release);

	impl_->app->listen(impl_->config.wsPort,
		[this](us_listen_socket_t* socket) {
			impl_->listenSocket = socket;
			if (socket) {
				OE_LOG_INFO("ws_listening: port={}", impl_->config.wsPort);
			} else {
				OE_LOG_ERROR("ws_listen_failed: port={}", impl_->config.wsPort);
			}
		}
	).run();  // blocks until the listen socket is closed

	impl_->running.store(false, std::memory_order_release);
}

void WsServer::stop()
{
	if (!impl_ || !impl_->loop) return;
	if (!impl_->running.load(std::memory_order_acquire)) return;

	// Defer the close onto the event-loop thread — safe from any thread.
	impl_->loop->defer([this]() {
		if (impl_->listenSocket) {
			us_listen_socket_close(0, impl_->listenSocket);
			impl_->listenSocket = nullptr;
		}
	});
}

void WsServer::broadcastVideo(std::vector<uint8_t> data)
{
	if (!impl_->loop) return;
	OE_LOG_DEBUG("ws_broadcast_video: bytes={}, clients={}",
	           data.size(), static_cast<int>(impl_->videoClients.size()));
	impl_->loop->defer([this, payload = std::move(data)]() {
		if (impl_->videoClients.empty()) return;
		std::string_view view(reinterpret_cast<const char*>(payload.data()),
		                      payload.size());
		for (auto* ws : impl_->videoClients) {
			ws->send(view, uWS::OpCode::BINARY, false);
		}
	});
}

void WsServer::broadcastAudio(std::vector<uint8_t> data)
{
	if (!impl_->loop) return;
	OE_LOG_DEBUG("ws_broadcast_audio: bytes={}, clients={}",
	           data.size(), static_cast<int>(impl_->audioClients.size()));
	impl_->loop->defer([this, payload = std::move(data)]() {
		if (impl_->audioClients.empty()) return;
		std::string_view view(reinterpret_cast<const char*>(payload.data()),
		                      payload.size());
		for (auto* ws : impl_->audioClients) {
			ws->send(view, uWS::OpCode::BINARY, false);
		}
	});
}

void WsServer::broadcastJson(nlohmann::json msg)
{
	if (!impl_->loop) return;
	OE_LOG_DEBUG("ws_broadcast_json: type={}, clients={}",
	           msg.value("type", "unknown"),
	           static_cast<int>(impl_->chatClients.size()));
	// Serialise on the calling thread to avoid holding a JSON object in
	// the deferred closure (JSON copy is more expensive than string copy).
	std::string serialised = msg.dump();
	impl_->loop->defer([this, text = std::move(serialised)]() {
		if (impl_->chatClients.empty()) return;
		std::string_view view(text);
		for (auto* ws : impl_->chatClients) {
			ws->send(view, uWS::OpCode::TEXT);
		}
	});
}

void WsServer::broadcastDenoisedVideo(std::vector<uint8_t> data)
{
	if (!impl_->loop) return;
	impl_->loop->defer([this, payload = std::move(data)]() {
		if (impl_->denoiseVideoClients.empty()) return;
		std::string_view view(reinterpret_cast<const char*>(payload.data()),
		                      payload.size());
		for (auto* ws : impl_->denoiseVideoClients) {
			ws->send(view, uWS::OpCode::BINARY, false);
		}
	});
}

void WsServer::broadcastDenoisedAudio(std::vector<uint8_t> data)
{
	if (!impl_->loop) return;
	impl_->loop->defer([this, payload = std::move(data)]() {
		if (impl_->denoiseAudioClients.empty()) return;
		std::string_view view(reinterpret_cast<const char*>(payload.data()),
		                      payload.size());
		for (auto* ws : impl_->denoiseAudioClients) {
			ws->send(view, uWS::OpCode::BINARY, false);
		}
	});
}

void WsServer::broadcastFilteredVideo(std::vector<uint8_t> data)
{
	if (!impl_->loop) return;
	impl_->loop->defer([this, payload = std::move(data)]() {
		if (impl_->filterVideoClients.empty()) return;
		std::string_view view(reinterpret_cast<const char*>(payload.data()),
		                      payload.size());
		for (auto* ws : impl_->filterVideoClients) {
			ws->send(view, uWS::OpCode::BINARY, false);
		}
	});
}

