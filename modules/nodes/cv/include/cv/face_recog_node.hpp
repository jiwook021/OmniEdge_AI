#pragma once

#include "face_recog_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/runtime_defaults.hpp"
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include <tl/expected.hpp>
#include "zmq/message_router.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// FaceRecognitionNode — reads BGR frames from SHM, runs face detection +
// embedding (SCRFD + AdaFace by default), publishes identity +
// face_registered on ZMQ port 5566.
//
// Two independent ZMQ SUB sockets:
//   zmqSubFrames_ — "video_frame" from port 5555 (ZMQ_CONFLATE=1)
//   zmqSubCmds_   — "ui_command"  from port 5570 (no conflation, must not drop)
//
// Thread safety: initialize() and stop() NOT thread-safe with run().
// stop() is safe from a signal handler.
// ---------------------------------------------------------------------------


class FaceRecognitionNode : public ModuleNodeBase<FaceRecognitionNode> {
public:
	friend class ModuleNodeBase<FaceRecognitionNode>;

	struct Config {
		// ZMQ
		int  pubPort         = kFaceRecog;
		int  videoSubPort    = kVideoIngest;
		int  wsBridgeSubPort = kWsBridge;
		int  zmqSendHighWaterMark       = kPublisherDataHighWaterMark;
		int  zmqHeartbeatIvlMs  = kHeartbeatIntervalMs;
		int  zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
		int  zmqHeartbeatTimeoutMs  = kHeartbeatTimeoutMs;
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Model
		std::string modelPackPath;

		// SHM
		std::string inputShmName = "/oe.vid.ingest";

		// Database
		std::string knownFacesDb = "./data/known_faces.sqlite";

		// Tuning
		float    recognitionThreshold = kFaceRecognitionThreshold;
		uint32_t frameSubsample       = kFaceDetectionFrameSubsample;

		// Module identity
		std::string moduleName = "face_recognition";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	explicit FaceRecognitionNode(const Config& config);

	/** Inject the inference inferencer. Must be called before initialize(). */
	void setInferencer(std::unique_ptr<FaceRecogInferencer> inferencer) noexcept {
		inferencer_ = std::move(inferencer);
	}

	FaceRecognitionNode(const FaceRecognitionNode&) = delete;
	FaceRecognitionNode& operator=(const FaceRecognitionNode&) = delete;

	~FaceRecognitionNode();

	// -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
	[[nodiscard]] tl::expected<void, std::string> onConfigure();
	[[nodiscard]] tl::expected<void, std::string> onLoadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

	/** Register a named face from a BGR24 image.
	 *  Runs detect → embed → INSERT into SQLite + cache.
	 *  Returns true on success.
	 */
	bool registerFace(const std::string& name,
	                  const uint8_t*     bgrData,
	                  uint32_t           width,
	                  uint32_t           height);

	// -------------------------------------------------------------------------
	// Static cosine similarity utility — exposed for unit tests.
	// Returns a value in [-1, 1]; embeddings must be L2-normalised.
	// -------------------------------------------------------------------------
	[[nodiscard]] static float cosineSimilarity(
		const std::vector<float>& a,
		const std::vector<float>& b) noexcept;

private:
	Config                config_;
	MessageRouter         messageRouter_;
	uint32_t              frameCounter_{0};

	// SHM (consumer)
	std::unique_ptr<ShmMapping> shmIn_;

	// Inferencer
	std::unique_ptr<FaceRecogInferencer> inferencer_;

	// In-memory face database: name → L2-normalised embedding
	std::unordered_map<std::string, std::vector<float>> knownFaces_;

	// Internal helpers
	void processVideoFrame(const nlohmann::json& meta);
	void handleUiCommand(const nlohmann::json& cmd);

	void publishIdentity(const std::string& name,
	                     float              confidence,
	                     const FaceBBox&    bbox,
	                     const FaceLandmarks& landmarks);

	void publishFaceRegistered(const std::string& name, bool success,
	                           const std::string& errorMsg = {});

	std::pair<std::string, float> identify(const std::vector<float>& embedding);
	void loadFacesFromDb();
	void insertFaceToDb(const std::string& name, const std::vector<float>& embedding);
};

