#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — WsServer
//
// RAII wrapper around a uWebSockets uWS::App.  Manages six WebSocket
// endpoints and two HTTP endpoints on a single port:
//
//   WS  /video          — binary ArrayBuffer (JPEG frames from BackgroundBlur).
//                         Server → browser only.
//   WS  /audio          — binary ArrayBuffer (F32 24 kHz PCM from KokoroTTS).
//                         Server → browser only.
//   WS  /chat           — JSON text messages.  Bidirectional.
//                         Inbound:  browser ui_command JSON → onClientCommand callback.
//                         Outbound: llm_response, module_status, etc.
//   WS  /denoise_video  — binary ArrayBuffer (denoised JPEG from BasicVSR++).
//                         Server → browser only.  Active when video denoise toggled on.
//   WS  /denoise_audio  — binary ArrayBuffer (denoised PCM from DTLN).
//                         Server → browser only.  Active when audio denoise toggled on.
//   WS  /filter_video   — binary ArrayBuffer (filtered JPEG from FaceFilterNode).
//                         Server → browser only.  Active when face filter toggled on.
//
//   HTTP GET /status — JSON health document (client counts).
//   HTTP GET /*      — static files served from config_.frontendDir.
//
// Threading model:
//   WsServer must be constructed and run() must be called from the SAME
//   thread (the main thread).  The uWebSockets event loop runs on that thread
//   and is not thread-safe.
//
//   The ZMQ relay thread (WebSocketBridgeNode::zmqRelayLoop_) calls
//   broadcastVideo(), broadcastAudio(), broadcastJson() from a different
//   thread.  These methods are safe: they use loop_->defer() to marshal
//   the send onto the event-loop thread.
//
// Teardown:
//   stop() is thread-safe.  It uses loop_->defer() to close the listen
//   socket, which drains the event loop and causes run() to return.
// ---------------------------------------------------------------------------

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/runtime_defaults.hpp"


/**
 * @brief Per-connection state stored inside each WebSocket handle.
 *
 * Currently stateless — routing is entirely determined by which endpoint
 * the client connected to (/video, /audio, /chat).
 */
struct PerSocketData {
};

/**
 * @brief uWebSockets server managing six WebSocket channels + HTTP.
 *
 * Thread safety:
 *   - run()              — blocks the calling thread; call from main thread.
 *   - stop()             — thread-safe; safe from signal handler via relay thread.
 *   - broadcastVideo()   — thread-safe (uses loop_->defer()).
 *   - broadcastAudio()   — thread-safe (uses loop_->defer()).
 *   - broadcastJson()    — thread-safe (uses loop_->defer()).
 */
class WsServer {
public:
	/** @brief Runtime configuration for the WebSocket server. */
	struct Config {
		int         wsPort      = kWsHttpPort;  ///< WebSocket + HTTP listen port
		std::string frontendDir = "./frontend";    ///< Root for static file serving
	};

	/**
	 * @brief Construct and configure the server.
	 *
	 * Saves the event loop pointer for cross-thread broadcast.
	 * Does NOT bind the port yet — that happens in run().
	 *
	 * @param config       Server configuration.
	 * @param onCmd        Callback invoked on the event-loop thread when a /chat
	 *                     client sends a valid JSON ui_command message.
	 * @param onBinaryCmd  Callback invoked on the event-loop thread when a /chat
	 *                     client sends a binary frame (image upload).
	 */
	explicit WsServer(Config                              config,
	                  std::function<void(nlohmann::json)>  onCmd,
	                  std::function<void(std::string_view)> onBinaryCmd = nullptr);

	/** @brief Closes the server if still running. */
	~WsServer();

	WsServer(const WsServer&)            = delete;
	WsServer& operator=(const WsServer&) = delete;

	/**
	 * @brief Bind to the configured port and block on the event loop.
	 *
	 * Returns only after stop() is called (or listen fails).
	 * @throws std::runtime_error if the port cannot be bound.
	 */
	void run();

	/**
	 * @brief Signal the event loop to stop.  Thread-safe.
	 *
	 * Schedules a listen-socket close via loop_->defer(); run() returns
	 * after the current event batch drains.
	 */
	void stop();

	/**
	 * @brief Broadcast a binary frame to all /video clients.
	 *
	 * Thread-safe.  Ownership of data is moved into the deferred lambda.
	 * @param data  JPEG or PNG bytes.
	 */
	void broadcastVideo(std::vector<uint8_t> data);

	/**
	 * @brief Broadcast a binary PCM chunk to all /audio clients.
	 *
	 * Thread-safe.  Ownership of data is moved into the deferred lambda.
	 * @param data  F32 little-endian PCM samples at 24 kHz.
	 */
	void broadcastAudio(std::vector<uint8_t> data);

	/**
	 * @brief Broadcast a JSON text message to all /chat clients.
	 *
	 * Thread-safe.  The JSON is serialised to a string before deferral.
	 * @param msg  JSON object to send (will be serialised internally).
	 */
	void broadcastJson(nlohmann::json msg);

	/**
	 * @brief Broadcast a denoised JPEG frame to all /denoise_video clients.
	 *
	 * Thread-safe. Used by BasicVSR++ video denoising comparison.
	 * @param data  JPEG bytes.
	 */
	void broadcastDenoisedVideo(std::vector<uint8_t> data);

	/**
	 * @brief Broadcast denoised PCM audio to all /denoise_audio clients.
	 *
	 * Thread-safe. Used by DTLN audio denoising comparison.
	 * @param data  F32 little-endian PCM samples at 16 kHz.
	 */
	void broadcastDenoisedAudio(std::vector<uint8_t> data);

	/**
	 * @brief Broadcast a filtered JPEG frame to all /filter_video clients.
	 *
	 * Thread-safe. Used by FaceFilterNode AR face filters.
	 * @param data  JPEG bytes.
	 */
	void broadcastFilteredVideo(std::vector<uint8_t> data);

private:
	struct Impl;  ///< Opaque PIMPL — holds uWS types; defined in ws_server.cpp
	std::unique_ptr<Impl> impl_;
};

