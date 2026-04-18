#pragma once

#include "cv/onnx_security_inferencer.hpp"
#include "cv/security_detection_annotator.hpp"
#include "cv/security_event_logger.hpp"
#include "cv/security_recorder.hpp"
#include "shm/shm_mapping.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/pipeline_types.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "common/constants/security_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/port_settings.hpp"
#include "zmq/zmq_constants.hpp"
#include "zmq/message_router.hpp"
#include "gpu/cuda_priority.hpp"
#include "gpu/cuda_stream.hpp"
#include "gpu/cuda_fence.hpp"
#include "vram/vram_thresholds.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// SecurityCameraNode — YOLOX-Nano detection + NVENC recording + event logging
//
// Data flow (JPEG mode — default, for display):
//   /oe.vid.ingest  ->  [read BGR24 via pinned buffer]
//   -> [YOLOX-Nano bounding-box detection at ~5 fps, COCO-80]
//   -> [detection filtering by class + confidence + cooldown]
//   -> [SecurityEventLogger: append to JSONL]
//   -> [SecurityDetectionAnnotator: draw bboxes -> JPEG]
//   -> /oe.cv.security.jpeg (SHM output)
//   -> ZMQ PUB "security_detection" + "security_event" on port 5578
//   -> [SecurityRecorder: feed BGR24 -> NVENC H.264 -> segmented MP4]
//
// Data flow (BGR24 mode — for pipeline chaining):
//   <configurable input SHM>  ->  [read BGR24]
//   -> [YOLOX-Nano detect + annotate bboxes]
//   -> /oe.cv.security.bgr (circular-buffer SHM, BGR24)
//   -> ZMQ PUB "security_bgr_frame" on port 5578
//
// Thread safety: initialize() and stop() are NOT thread-safe with run().
// stop() is safe to call from a signal handler (sets atomic flag only).
// ---------------------------------------------------------------------------

class SecurityCameraNode : public ModuleNodeBase<SecurityCameraNode> {
public:
    friend class ModuleNodeBase<SecurityCameraNode>;

    struct Config {
        // ZMQ
        int pubPort          = kSecurityCamera;
        int videoSubPort     = kVideoIngest;
        int wsBridgeSubPort  = kWsBridge;
        int zmqSendHighWaterMark      = kPublisherDataHighWaterMark;
        int zmqHeartbeatIvlMs         = kHeartbeatIntervalMs;
        int zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
        int zmqHeartbeatTimeoutMs     = kHeartbeatTimeoutMs;
        std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

        // Model — path to yolox_nano.onnx (auto-fetched from HF on first use).
        std::string yoloEnginePath;

        // Frame geometry
        uint32_t inputWidth  = kMaxInputWidth;
        uint32_t inputHeight = kMaxInputHeight;

        // Detection
        int   detectionIntervalMs    = kSecurityDetectionIntervalMs;
        float confidenceThreshold    = kSecurityConfidenceThreshold;
        int   eventCooldownMs        = kSecurityEventCooldownMs;
        std::vector<std::string> targetClasses = {"person", "backpack", "suitcase"};

        // Recording
        std::string recordingDir       = "recordings/";
        int         segmentDurationMin = kSecuritySegmentDurationMin;
        int         recordingFps       = kSecurityRecordingFps;
        int         recordingBitrate   = kSecurityRecordingBitrate;

        // Annotated JPEG
        int jpegQuality = kSecurityJpegQuality;

        // SHM — input
        std::string inputShmName  = "/oe.vid.ingest";
        std::string inputTopic    = "video_frame";   ///< ZMQ topic that triggers processFrame

        // SHM — output
        std::string outputShmName = "/oe.cv.security.jpeg";   ///< JPEG mode default
        std::string outputBgrShmName = "/oe.cv.security.bgr"; ///< BGR24 mode default
        OutputFormat outputFormat = OutputFormat::kJpeg;
        uint32_t outputSlotCount = kCircularBufferSlotCount; ///< BGR24 circular buffer slots

        // Event log
        std::string logDir = "logs/";

        // Module identity
        std::string moduleName = "security_camera";
        std::string bgrTopic = "security_bgr_frame";

        [[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
    };

    explicit SecurityCameraNode(const Config& config);
    ~SecurityCameraNode();

    SecurityCameraNode(const SecurityCameraNode&) = delete;
    SecurityCameraNode& operator=(const SecurityCameraNode&) = delete;

    /// Allocate GPU buffers, open SHM, bind ZMQ, load YOLO engine.
    void initialize();

    /// Blocking event loop — returns on stop() or SIGTERM.
    void run();

    /// Signal the event loop to exit (thread-safe, signal-handler-safe).
    void stop() noexcept;

private:
    // ── CRTP hooks ──────────────────────────────────────────────────────────
    tl::expected<void, std::string> configureTransport();
    tl::expected<void, std::string> loadInferencer();
    MessageRouter& router() noexcept { return messageRouter_; }
    std::string_view moduleName() const noexcept { return config_.moduleName; }

    // ── Hot path ────────────────────────────────────────────────────────────
    void processFrame();

    // ── UI command handlers ─────────────────────────────────────────────────
    void handleToggleDetection(const nlohmann::json& msg);
    void handleListRecordings(const nlohmann::json& msg);
    void handleListEvents(const nlohmann::json& msg);
    void handleUpdateClasses(const nlohmann::json& msg);
    void handleSetStyle(const nlohmann::json& msg);
    void handleSetRoi(const nlohmann::json& msg);

    // ── Utilities ───────────────────────────────────────────────────────────
    /// Generate an ISO-8601 UTC timestamp string for the current moment.
    static std::string nowIso8601();

    /// Check if a YOLO class index is in the target classes list.
    bool isTargetClass(int classId) const;

    /// Map COCO class ID to name.
    static std::string cocoClassName(int classId);

    // ── Members ─────────────────────────────────────────────────────────────
    Config config_;
    MessageRouter messageRouter_;

    // SHM
    std::unique_ptr<ShmMapping> shmIn_;
    std::unique_ptr<ShmMapping> shmOut_;  ///< JPEG mode only

    // BGR24 output SHM (only in kBgr24 mode — pipeline chaining)
    std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shmOutBgr_;

    // YOLOX-Nano detection inference (ONNX Runtime).
    std::unique_ptr<OnnxSecurityInferencer> inferencer_;

    // Security subsystems
    std::unique_ptr<oe::security::SecurityEventLogger>      eventLogger_;
    std::unique_ptr<oe::security::SecurityRecorder>          recorder_;
    std::unique_ptr<oe::security::SecurityDetectionAnnotator> annotator_;

    // State
    std::atomic<bool> running_{false};
    bool              detectionActive_ = true;
    bool              recordingActive_ = false;

    // Per-class cooldown tracking (class name → last event time).
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> lastEventTime_;

    // Frame timing
    std::chrono::steady_clock::time_point lastDetectionTime_;
    std::chrono::steady_clock::time_point lastPurgeTime_;
    uint64_t frameSeq_ = 0;

    // Pre-allocated frame buffer for SHM read.
    std::vector<uint8_t> frameBuf_;

    // Publish "security_bgr_frame" ZMQ notification (BGR24 mode)
};
