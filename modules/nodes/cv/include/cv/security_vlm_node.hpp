#pragma once

#include "cv/vision_worker_client.hpp"
#include "shm/shm_mapping.hpp"
#include "common/module_node_base.hpp"
#include "common/constants/security_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/port_settings.hpp"
#include "zmq/zmq_constants.hpp"
#include "zmq/message_router.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// SecurityVlmNode — LP agent loop (YOLO event → clip → Gemma VLM → incident)
//
// Pure downstream consumer of the security_camera node:
//
//   /oe.vid.ingest (SHM BGR24)  ──► SecurityVlmNode
//                                       │   in-proc JPEG ring (preroll_sec × fps)
//                                       │
//   ZMQ 5578 "security_event"   ──────► │   enqueue pending incident
//                                       │
//                                       │  after postroll_sec:
//                                       │    sample N frames → VisionWorkerClient.analyze()
//                                       ▼
//   ZMQ 5578 "security_vlm_analysis"  ◄─┤
//   ${logDir}/security_incidents.jsonl  ┘
//
// Alert latency: unchanged. The security_camera node fires security_event
// instantly; this node processes it off-path. Reasoning is seconds-latency.
// ---------------------------------------------------------------------------

class SecurityVlmNode : public ModuleNodeBase<SecurityVlmNode> {
public:
    friend class ModuleNodeBase<SecurityVlmNode>;

    struct Config {
        // ZMQ — shares port 5578 with security_camera.
        int pubPort                  = kSecurityCamera;
        int videoSubPort             = kVideoIngest;
        int securityEventSubPort     = kSecurityCamera;
        int wsBridgeSubPort          = kWsBridge;
        int zmqSendHighWaterMark     = kPublisherDataHighWaterMark;
        int zmqHeartbeatIvlMs        = kHeartbeatIntervalMs;
        int zmqHeartbeatTimeToLiveMs = kHeartbeatTtlMs;
        int zmqHeartbeatTimeoutMs    = kHeartbeatTimeoutMs;
        std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

        // Frame geometry — must match /oe.vid.ingest producer.
        uint32_t inputWidth  = kMaxInputWidth;
        uint32_t inputHeight = kMaxInputHeight;

        // Clip windowing
        int prerollSec    = 3;
        int postrollSec   = 2;
        int frameSamples  = 6;
        int ingestFps     = 5;
        int jpegDownscale = 336;
        int jpegQuality   = 80;

        // Vision worker
        std::string modelDir;
        std::string scriptPathOverride;
        int startupTimeoutSec = 180;
        int requestTimeoutMs  = 60000;
        int idleUnloadSec     = 120;
        int maxPendingEvents  = 8;

        // SHM + logs
        std::string inputShmName      = "/oe.vid.ingest";
        std::string inputTopic        = "video_frame";
        std::string logDir            = "logs/";
        std::string incidentsFileName = "security_incidents.jsonl";

        // Prompt
        std::string promptTemplate =
            "You are a Loss Prevention analyst reviewing sequential frames of a retail scene. "
            "Describe the subject's behavior in one short sentence and assign a suspicion_level "
            "on a 0-3 scale: 0=benign, 1=mild, 2=suspicious, 3=theft-likely. "
            "Reply ONLY with valid JSON: "
            "{\"behavior\":\"...\",\"suspicion_level\":0-3,\"reasoning\":\"...\","
            "\"objects_observed\":[\"...\"]}";

        std::string moduleName = "security_vlm";

        [[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
    };

    explicit SecurityVlmNode(const Config& config);
    ~SecurityVlmNode();

    SecurityVlmNode(const SecurityVlmNode&) = delete;
    SecurityVlmNode& operator=(const SecurityVlmNode&) = delete;

    void initialize();
    void run();
    void stop() noexcept;

private:
    // CRTP hooks
    tl::expected<void, std::string> configureTransport();
    tl::expected<void, std::string> loadInferencer();
    MessageRouter& router() noexcept { return messageRouter_; }
    std::string_view moduleName() const noexcept { return config_.moduleName; }
    void onAfterStop();

    // Hot-path handlers
    void onVideoFrame();
    void onSecurityEvent(const nlohmann::json& msg);

    // Per-tick work (called after each handler — cheap when nothing pending)
    void drainPendingIncidents();
    void maybeUnloadIdleVision();

    // Helpers
    [[nodiscard]] tl::expected<void, std::string> ensureVisionLoaded();
    [[nodiscard]] std::vector<std::vector<std::uint8_t>>
    assembleClip(std::uint64_t triggerSeq) const;
    void publishIncident(const std::string& eventId,
                         const nlohmann::json& event,
                         const nlohmann::json& analysis,
                         int imageCount);
    void appendIncidentJsonl(const nlohmann::json& incident);
    static std::string makeEventId();
    static std::string nowIso8601();
    static std::vector<std::uint8_t>
    encodeJpeg(const std::uint8_t* bgr24, std::uint32_t width,
               std::uint32_t height, std::uint32_t maxDim, int quality);

    struct RingFrame {
        std::uint64_t frameSeq{};
        std::chrono::steady_clock::time_point captureTime;
        std::vector<std::uint8_t> jpeg;
    };

    struct PendingIncident {
        std::string    eventId;
        nlohmann::json source;                            ///< original security_event JSON
        std::uint64_t  triggerFrameSeq{};
        std::chrono::steady_clock::time_point arrival;
        std::chrono::steady_clock::time_point dispatchAt;
    };

    Config         config_;
    MessageRouter  messageRouter_;

    std::unique_ptr<ShmMapping>                         shmIn_;
    std::unique_ptr<omniedge::cv::VisionWorkerClient>   vision_;

    std::deque<RingFrame>        ring_;
    std::size_t                  ringCapacity_ = 0;
    std::uint64_t                ingestFrameSeq_ = 0;

    std::deque<PendingIncident>  pending_;
    std::vector<std::uint8_t>    frameBuf_;

    std::ofstream                incidentsFile_;
    std::chrono::steady_clock::time_point lastVisionUse_{};
    bool                         visionLoaded_ = false;

    std::atomic<bool>            running_{false};
    std::uint64_t                incidentCount_ = 0;
};
