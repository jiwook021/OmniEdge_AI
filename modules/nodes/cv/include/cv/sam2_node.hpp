#pragma once

#include <atomic>
#include <mutex>

#include "sam2_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "common/runtime_defaults.hpp"
#include "common/pipeline_types.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "zmq/message_router.hpp"
#include "common/constants/video_constants.hpp"
#include "common/constants/cv_constants.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Sam2Node — SHM consumer + SAM2 interactive segmentation + SHM/ZMQ producer
//
// Data flow (JPEG mode — default, for display):
//   /oe.vid.ingest  ->  [read BGRFrame via ShmCircularBuffer]
//   -> [SAM2 image encoder -> embeddings cache]
//   -> (user prompt arrives via ui_command: point, box, or mask)
//   -> [SAM2 mask decoder -> binary segmentation mask]
//   -> [Alpha-composite mask overlay onto original frame]
//   -> [JPEG encode]
//   -> /oe.cv.sam2.mask (SHM circular buffer)
//   -> ZMQ PUB "segmentation_mask" on port 5576
//
// Data flow (BGR24 mode — for pipeline chaining):
//   <configurable input SHM>  ->  [read BGRFrame]
//   -> [SAM2 encode + segment + composite on GPU]
//   -> /oe.cv.sam2.bgr (circular-buffer SHM, BGR24)
//   -> ZMQ PUB "sam2_bgr_frame" on port 5576
//
// Commands handled (via ui_command topic from WebSocketBridge port 5570):
//   sam2_segment_point  — segment at a point (x, y, label)
//   sam2_segment_box    — segment within a bounding box (x1, y1, x2, y2)
//   sam2_segment_mask   — refine an existing mask
//   toggle_sam2         — enable/disable the SAM2 module
//
// Thread safety: initialize() and stop() are NOT thread-safe with run().
// stop() is safe to call from a signal handler (sets atomic flag only).
// ---------------------------------------------------------------------------


class Sam2Node : public ModuleNodeBase<Sam2Node> {
public:
    friend class ModuleNodeBase<Sam2Node>;

    struct Config {
        // ZMQ
        int  pubPort         = kSam2;
        int  videoSubPort    = kVideoIngest;
        int  wsBridgeSubPort = kWsBridge;
        int  daemonSubPort   = kDaemon;
        int  zmqSendHighWaterMark       = kPublisherDataHighWaterMark;
        int  zmqHeartbeatIvlMs  = kHeartbeatIntervalMs;
        int  zmqHeartbeatTimeToLiveMs  = kHeartbeatTtlMs;
        int  zmqHeartbeatTimeoutMs  = kHeartbeatTimeoutMs;
        std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

        // Model paths
        std::string encoderOnnxPath;
        std::string decoderOnnxPath;

        // Frame geometry
        uint32_t inputWidth  = kMaxInputWidth;
        uint32_t inputHeight = kMaxInputHeight;

        // Quality
        int   jpegQuality = kSam2JpegQuality;

        // SHM — input
        std::string inputShmName  = "/oe.vid.ingest";
        std::string inputTopic    = "video_frame";   ///< ZMQ topic that triggers processSegmentation

        // SHM — output
        std::string outputShmName = "/oe.cv.sam2.mask";   ///< JPEG mode default
        std::string outputBgrShmName = "/oe.cv.sam2.bgr"; ///< BGR24 mode default
        OutputFormat outputFormat = OutputFormat::kJpeg;
        uint32_t outputSlotCount = kCircularBufferSlotCount; ///< BGR24 circular buffer slots

        // Module identity
        std::string moduleName = "sam2";
        std::string bgrTopic = "sam2_bgr_frame";

        // Whether SAM2 is enabled at startup
        bool enabledAtStartup = false;

        [[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
    };

    explicit Sam2Node(const Config& config);

    /** Inject the inference inferencer. Must be called before initialize().
     *  Enables stub injection in tests without requiring a GPU.
     */
    void setInferencer(std::unique_ptr<Sam2Inferencer> inferencer) noexcept {
        inferencer_ = std::move(inferencer);
    }

    Sam2Node(const Sam2Node&) = delete;
    Sam2Node& operator=(const Sam2Node&) = delete;

    ~Sam2Node();

    // -- CRTP lifecycle hooks (called by ModuleNodeBase) -----
    [[nodiscard]] tl::expected<void, std::string> configureTransport();
    [[nodiscard]] tl::expected<void, std::string> loadInferencer();
    [[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
    [[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

    /** Query whether SAM2 is currently enabled. */
    [[nodiscard]] bool isSam2Enabled() const noexcept { return sam2Enabled_; }

    /** Get the last prompt type processed. */
    [[nodiscard]] Sam2PromptType lastPromptType() const noexcept {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        return lastPromptType_;
    }

    /** Get the current background compositing mode. */
    [[nodiscard]] Sam2BackgroundMode backgroundMode() const noexcept {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        return bgMode_;
    }

    /** Query whether continuous tracking mode is active. */
    [[nodiscard]] bool isTracking() const noexcept {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        return isTracking_;
    }

private:
    Config                        config_;
    MessageRouter                 messageRouter_;
    uint32_t                      frameSeq_{0};

    // SHM
    std::unique_ptr<ShmMapping> shmIn_;   ///< /oe.vid.ingest     (consumer)
    std::unique_ptr<ShmMapping> shmOut_;  ///< JPEG output (producer, only in kJpeg mode)

    // BGR24 output SHM (only in kBgr24 mode — pipeline chaining)
    std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shmOutBgr_;

    // Inference inferencer
    std::unique_ptr<Sam2Inferencer> inferencer_;

    // Module state (atomics for safe access from public API / signal handler)
    std::atomic<bool> sam2Enabled_{false};

    // Mutex protects all prompt and UI-configurable state that is shared
    // between processSegmentation() and handleUiCommand() callbacks.
    mutable std::mutex promptMutex_;
    Sam2PromptType    lastPromptType_{Sam2PromptType::kPoint};
    Sam2Prompt        pendingPrompt_;
    bool              hasPendingPrompt_{false};

    // Continuous tracking mode: after initial prompt, keep re-segmenting each frame
    bool isTracking_{false};

    // Frame pacing for continuous tracking (33ms = ~30fps)
    std::chrono::steady_clock::time_point lastFrameTime_{};

    // Background compositing mode (configurable via UI)
    Sam2BackgroundMode bgMode_{Sam2BackgroundMode::kBlur};
    uint8_t bgColorR_{kSam2DefaultBgColorR};
    uint8_t bgColorG_{kSam2DefaultBgColorG};
    uint8_t bgColorB_{kSam2DefaultBgColorB};

    // Process a segmentation request
    void processSegmentation(const nlohmann::json& frameMetadata);

    // Handle a ui_command JSON
    void handleUiCommand(const nlohmann::json& cmd);

    // Publish "segmentation_mask" ZMQ notification (JPEG mode)
    void publishSegmentationMask(std::size_t jpegSize, const Sam2Result& result);

    // Publish "sam2_bgr_frame" ZMQ notification (BGR24 mode)
};

