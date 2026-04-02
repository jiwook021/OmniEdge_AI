#pragma once

#include "sam2_inferencer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "zmq/message_router.hpp"
#include "common/constants/video_constants.hpp"
#include "gpu/cuda_fence.hpp"
#include "common/constants/cv_constants.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Sam2Node — SHM consumer + SAM2 interactive segmentation + SHM/ZMQ producer
//
// Data flow:
//   /oe.vid.ingest  ->  [read BGRFrame via ShmCircularBuffer]
//   -> [SAM2 image encoder → embeddings cache]
//   -> (user prompt arrives via ui_command: point, box, or mask)
//   -> [SAM2 mask decoder → binary segmentation mask]
//   -> [Alpha-composite mask overlay onto original frame]
//   -> [JPEG encode]
//   -> /oe.cv.sam2.mask (SHM circular buffer)
//   -> ZMQ PUB "segmentation_mask" on port 5576
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

        // SHM
        std::string inputShmName  = "/oe.vid.ingest";
        std::string outputShmName = "/oe.cv.sam2.mask";

        // Module identity
        std::string moduleName = "sam2";

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
    [[nodiscard]] tl::expected<void, std::string> onConfigure();
    [[nodiscard]] tl::expected<void, std::string> onLoadInferencer();
    [[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
    [[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

    /** Query whether SAM2 is currently enabled. */
    [[nodiscard]] bool isSam2Enabled() const noexcept { return sam2Enabled_; }

    /** Get the last prompt type processed. */
    [[nodiscard]] Sam2PromptType lastPromptType() const noexcept { return lastPromptType_; }

private:
    Config                        config_;
    MessageRouter                 messageRouter_;
    uint32_t                      frameSeq_{0};

    // SHM
    std::unique_ptr<ShmMapping> shmIn_;   ///< /oe.vid.ingest     (consumer)
    std::unique_ptr<ShmMapping> shmOut_;  ///< /oe.cv.sam2.mask   (producer)

    // Inference inferencer
    std::unique_ptr<Sam2Inferencer> inferencer_;

    // GPU->host synchronization fence
    CudaFence gpuFence_;

    // Module state
    bool            sam2Enabled_{false};
    Sam2PromptType  lastPromptType_{Sam2PromptType::kPoint};
    Sam2Prompt      pendingPrompt_;
    bool            hasPendingPrompt_{false};

    // Process a segmentation request
    void processSegmentation(const nlohmann::json& frameMetadata);

    // Handle a ui_command JSON
    void handleUiCommand(const nlohmann::json& cmd);

    // Publish "segmentation_mask" ZMQ notification
    void publishSegmentationMask(std::size_t jpegSize, const Sam2Result& result);
};

