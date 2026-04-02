#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "zmq/message_router.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "common/validated_config.hpp"


/**
 * @brief Abstract relay surface used by WebSocketBridgeNode.
 *
 * Production implementation wraps WsServer (uWebSockets). Tests inject a mock.
 */
class IWsRelay {
public:
	virtual ~IWsRelay() = default;

	/** @brief Start the relay event loop. Blocks until stop() is called. */
	virtual void run() = 0;

	/** @brief Request event-loop shutdown. Thread-safe. */
	virtual void stop() = 0;

	/** @brief Broadcast binary JPEG/PNG payload to /video clients. */
	virtual void broadcastVideo(std::vector<uint8_t> data) = 0;

	/** @brief Broadcast binary PCM payload to /audio clients. */
	virtual void broadcastAudio(std::vector<uint8_t> data) = 0;

	/** @brief Broadcast JSON text message to /chat clients. */
	virtual void broadcastJson(nlohmann::json msg) = 0;

	/** @brief Broadcast denoised JPEG to /denoise_video clients. */
	virtual void broadcastDenoisedVideo(std::vector<uint8_t> data) = 0;

	/** @brief Broadcast denoised PCM to /denoise_audio clients. */
	virtual void broadcastDenoisedAudio(std::vector<uint8_t> data) = 0;
};

/**
 * @brief Bridge node: ZMQ topics <-> five-channel WebSocket relay.
 *
 * Responsibilities:
 *   - Subscribes to upstream module topics and forwards to the appropriate
 *     WebSocket channel (/video, /audio, /chat).
 *   - Validates inbound browser ui_command JSON and publishes to ZMQ topic
 *     ui_command on port 5570.
 *   - Tracks UI mode switches and adjusts active relay topics.
 *
 * Threading model:
 *   - Main thread runs the uWebSockets event loop (relay_->run()).
 *   - A dedicated relay thread runs messageRouter_.run(), which polls all ZMQ SUB
 *     sockets. Handlers registered via messageRouter_.subscribe() execute on this
 *     thread and use relay_->broadcastVideo/Audio/Json to push data into
 *     the uWS event loop.
 *   - handleClientCommand() is called from the uWS main thread and calls
 *     messageRouter_.publish() on the PUB socket. This is safe because the PUB
 *     socket is only touched from the main thread (never from run()).
 */
class WebSocketBridgeNode {
public:
	struct Config {
		int wsBridgePubPort = kWsBridge;
		int blurSubPort     = kBgBlur;
		int videoIngestSubPort = kVideoIngest;
		int ttsSubPort      = kTts;
		int llmSubPort      = kLlm;
		int daemonSubPort   = kDaemon;
		int faceSubPort     = kFaceRecog;
		int videoDenoiseSubPort = kVideoDenoise;
		int audioDenoiseSubPort = kAudioDenoise;

		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};
		std::string moduleName{"omniedge_ws_bridge"};

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	explicit WebSocketBridgeNode(const Config& config);
	~WebSocketBridgeNode();

	WebSocketBridgeNode(const WebSocketBridgeNode&)            = delete;
	WebSocketBridgeNode& operator=(const WebSocketBridgeNode&) = delete;

	/** @brief Inject relay implementation before initialize(). */
	void setRelay(std::unique_ptr<IWsRelay> relay);

	/** @brief Register subscriptions and publish module_ready via router. */
	void initialize();

	/** @brief Start relay loop thread + blocking WebSocket loop. */
	void run();

	/** @brief Thread-safe shutdown. */
	void stop() noexcept;

	/// Check if enough time has passed since the last send for frame pacing.
	/// Public for testability — pure function with no side effects.
	[[nodiscard]] static bool shouldPace(
		std::chrono::steady_clock::time_point lastSend,
		int pacingMs) noexcept;

	/** @brief Handle one browser ui_command message from /chat WebSocket. */
	void handleClientCommand(const nlohmann::json& rawCommand);

	/**
	 * @brief Handle a binary image upload from /chat WebSocket.
	 *
	 * Binary frame format:
	 *   [4 bytes: JSON header length, little-endian uint32]
	 *   [N bytes: UTF-8 JSON with action metadata]
	 *   [remaining bytes: raw JPEG/PNG image data]
	 *
	 * Writes the image to a temp file and publishes describe_uploaded_image.
	 */
	void handleBinaryUpload(std::string_view binaryFrame);

	/**
	 * @brief Validate/normalise ui_command JSON.
	 *
	 * Ensures required fields and action-specific payload fields are present.
	 */
	[[nodiscard]] static tl::expected<nlohmann::json, std::string>
	validateUiCommand(const nlohmann::json& rawCommand);

private:
	void relayTopicPayload_(std::string_view topic, const nlohmann::json& payload);
	void setMode_(std::string_view mode);

	[[nodiscard]] std::vector<uint8_t> readBlurredFrameFromShm_(const nlohmann::json& payload);
	[[nodiscard]] std::vector<uint8_t> readRawVideoFrameFromShm_(const nlohmann::json& payload);
	[[nodiscard]] std::vector<uint8_t> readTtsAudioFromShm_(const nlohmann::json& payload);
	[[nodiscard]] std::vector<uint8_t> readDenoisedFrameFromShm_(const nlohmann::json& payload);
	[[nodiscard]] std::vector<uint8_t> readDenoisedAudioFromShm_(const nlohmann::json& payload);

	Config config_;
	std::atomic<bool> running_{false};

	std::unique_ptr<IWsRelay> relay_;

	MessageRouter messageRouter_;

	// activeTopics_ is written by setMode_() (WebSocket event-loop thread) and
	// read by messageRouter_.run() handlers (relayThread). Guard with activeMutex_.
	std::mutex                      activeMutex_;
	std::unordered_set<std::string> activeTopics_;

	std::unordered_map<std::string, std::unique_ptr<class ShmMapping>> shmCache_;

	std::string currentMode_{"voice"};

	/// Monotonic counter for unique upload temp filenames.
	/// Atomic: incremented on uWS event-loop thread only, but atomic for safety.
	std::atomic<uint64_t> uploadCounter_{0};

	// Tracks the last time a blurred_frame was relayed from bg_blur.
	// Used by relayTopicPayload_() to suppress raw video_frame fallback
	// when bg_blur is actively producing processed frames.
	// Written and read only from the messageRouter_.run() thread — no mutex needed.
	std::chrono::steady_clock::time_point lastBlurFrameTime_{};

	// --- Frame pacing state (relay thread only — no mutex) -----------------

	/// Last time we sent a frame on the /video channel.
	std::chrono::steady_clock::time_point lastVideoSendTime_{};
	/// Cached last blur frame for re-send when no new frame arrives.
	std::vector<uint8_t> lastBlurFrame_;

	/// Last time we sent a frame on the /denoise_video channel.
	std::chrono::steady_clock::time_point lastDenoiseSendTime_{};
	/// Cached last denoise frame for re-send.
	std::vector<uint8_t> lastDenoiseFrame_;

};

