#include "cv/sam2_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_frame_reader.hpp"

#include <cstring>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>


// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------
tl::expected<Sam2Node::Config, std::string>
Sam2Node::Config::validate(const Config& raw)
{
    ConfigValidator v;
    v.requirePort("pubPort", raw.pubPort);
    v.requirePort("videoSubPort", raw.videoSubPort);
    v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);
    v.requireRange("jpegQuality", raw.jpegQuality, 1, 100);

    if (auto err = v.finish(); !err.empty()) {
        return tl::unexpected(err);
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
Sam2Node::Sam2Node(const Config& config)
    : config_(config),
      messageRouter_(MessageRouter::Config{
          config.moduleName,
          config.pubPort,
          config.zmqSendHighWaterMark,
          config.pollTimeout}),
      sam2Enabled_(config.enabledAtStartup)
{
}

Sam2Node::~Sam2Node()
{
    stop();
}

// ---------------------------------------------------------------------------
// CRTP lifecycle: configureTransport
// ---------------------------------------------------------------------------
tl::expected<void, std::string> Sam2Node::configureTransport()
{
    OE_ZONE_SCOPED;
    OeLogger::instance().setModule(config_.moduleName);

    // Input SHM: /oe.vid.ingest (consumer — retry up to 10 times waiting for producer)
    const std::size_t inShmSize =
        ShmCircularBuffer<ShmVideoHeader>::segmentSize(
            kCircularBufferSlotCount, kMaxBgr24FrameBytes);
    auto shmResult = oe::shm::openConsumerWithRetry(
        config_.inputShmName, inShmSize, "sam2");
    if (!shmResult) return tl::unexpected(shmResult.error());
    shmIn_ = std::move(*shmResult);

    // Output SHM: /oe.cv.sam2.mask (producer — will unlink on close)
    shmOut_ = oe::shm::createProducer(config_.outputShmName, kJpegShmSegmentByteSize);

    // ZMQ subscriptions via MessageRouter
    messageRouter_.subscribe(config_.videoSubPort, "video_frame", /*conflate=*/true,
        [this](const nlohmann::json& msg) { processSegmentation(msg); });
    messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
        [this](const nlohmann::json& msg) { handleUiCommand(msg); });
    messageRouter_.subscribe(config_.daemonSubPort, "sam2_prompt", /*conflate=*/false,
        [this](const nlohmann::json& msg) { handleUiCommand(msg); });

    OE_LOG_INFO("sam2_configured: pub={}, video_sub={}, ws_sub={}, shm_in={}, shm_out={}, enabled={}",
        config_.pubPort, config_.videoSubPort, config_.wsBridgeSubPort,
        config_.inputShmName, config_.outputShmName, sam2Enabled_);

    return {};
}

// ---------------------------------------------------------------------------
// CRTP lifecycle: loadInferencer
// ---------------------------------------------------------------------------
tl::expected<void, std::string> Sam2Node::loadInferencer()
{
    OE_ZONE_SCOPED;

    if (!inferencer_) {
        return tl::unexpected(std::string(
            "No inferencer — call setInferencer() before initialize()"));
    }

    try {
        inferencer_->loadModel(config_.encoderOnnxPath, config_.decoderOnnxPath);
    } catch (const std::exception& e) {
        return tl::unexpected(std::format(
            "SAM2 model load failed: {}", e.what()));
    }

    OE_LOG_INFO("sam2_model_loaded: encoder={}, decoder={}, vram_mib={}",
        config_.encoderOnnxPath, config_.decoderOnnxPath,
        inferencer_->currentVramUsageBytes() / (1024 * 1024));

    return {};
}

// ---------------------------------------------------------------------------
// processSegmentation — run on each video_frame when a prompt is pending
// ---------------------------------------------------------------------------
void Sam2Node::processSegmentation(const nlohmann::json& /*frameMetadata*/)
{
    OE_ZONE_SCOPED;
    if (!sam2Enabled_ || !hasPendingPrompt_ || !inferencer_) {
        return;
    }

    OE_LOG_DEBUG("sam2_process: seq={}, prompt_type={}",
                 frameSeq_, static_cast<int>(pendingPrompt_.type));

    // 1. Read latest BGR24 frame from input SHM
    const auto frame = readLatestBgrFrame(*shmIn_);
    if (!frame.data) return;

    // 2. Pre-flight VRAM check
    auto vramCheck = ensureVramAvailableMiB(kSam2InferenceHeadroomMiB);
    if (!vramCheck.has_value()) {
        OE_LOG_WARN("sam2_vram_insufficient: {}", vramCheck.error());
        return;
    }

    // 3. Pick the inactive output slot (double-buffer flip)
    auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
    const uint32_t writeSlot =
        1u - ctrl->writeIndex.load(std::memory_order_relaxed);
    uint8_t* outSlot = shmOut_->bytes()
                       + sizeof(ShmJpegControl)
                       + writeSlot * kMaxJpegBytesPerSlot;

    // 4. Run full pipeline: encode + segment + overlay + JPEG
    auto result = inferencer_->processFrame(
        frame.data, frame.width, frame.height,
        pendingPrompt_,
        outSlot, kMaxJpegBytesPerSlot);

    if (!result.has_value()) {
        OE_LOG_ERROR("sam2_process_failed: {}", result.error());
        return;
    }

    const std::size_t jpegSize = result.value();

    // 5. GPU fence — block until the GPU has finished writing the JPEG
    //    to the output slot. Without this, the consumer (WS Bridge) could
    //    read a partially-written frame.
    gpuFence_.record(nullptr);  // default CUDA stream
    gpuFence_.synchronize();

    // 6. Stale-read guard — discard if input was overwritten during inference
    if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

    // 7. Commit JPEG to output SHM and publish
    ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
    ctrl->seqNumber[writeSlot] = ++frameSeq_;
    ctrl->writeIndex.store(writeSlot, std::memory_order_release);

    // Retrieve segmentation metadata (IOU score, stability) from the inferencer.
    // processFrame() already ran encodeImage+segmentWithPrompt internally,
    // so this re-call is a cache hit on most implementations.
    Sam2Result segMeta;
    segMeta.maskWidth  = frame.width;
    segMeta.maskHeight = frame.height;
    auto segResult = inferencer_->segmentWithPrompt(pendingPrompt_);
    if (segResult.has_value()) {
        segMeta.iouScore  = segResult->iouScore;
        segMeta.stability = segResult->stability;
    }
    publishSegmentationMask(jpegSize, segMeta);

    // Clear the pending prompt (one-shot per prompt)
    hasPendingPrompt_ = false;
}

// ---------------------------------------------------------------------------
// handleUiCommand — parse SAM2-related commands from the UI
// ---------------------------------------------------------------------------
void Sam2Node::handleUiCommand(const nlohmann::json& cmd)
{
    const std::string action = cmd.value("action", std::string{});

    if (action == "toggle_sam2") {
        sam2Enabled_ = !sam2Enabled_;
        OE_LOG_INFO("sam2_toggled: enabled={}", sam2Enabled_);
        return;
    }

    if (action == "sam2_segment_point") {
        pendingPrompt_.type = Sam2PromptType::kPoint;
        pendingPrompt_.points.clear();
        Sam2PointPrompt pt;
        pt.x     = cmd.value("x", 0.5f);
        pt.y     = cmd.value("y", 0.5f);
        pt.label = cmd.value("label", 1);
        pendingPrompt_.points.push_back(pt);
        hasPendingPrompt_ = true;
        lastPromptType_   = Sam2PromptType::kPoint;
        OE_LOG_INFO("sam2_point_prompt: ({:.3f}, {:.3f}) label={}",
                    pt.x, pt.y, pt.label);
        return;
    }

    if (action == "sam2_segment_box") {
        pendingPrompt_.type = Sam2PromptType::kBox;
        pendingPrompt_.box.x1 = cmd.value("x1", 0.0f);
        pendingPrompt_.box.y1 = cmd.value("y1", 0.0f);
        pendingPrompt_.box.x2 = cmd.value("x2", 1.0f);
        pendingPrompt_.box.y2 = cmd.value("y2", 1.0f);
        hasPendingPrompt_ = true;
        lastPromptType_   = Sam2PromptType::kBox;
        OE_LOG_INFO("sam2_box_prompt: ({:.3f},{:.3f})->({:.3f},{:.3f})",
                    pendingPrompt_.box.x1, pendingPrompt_.box.y1,
                    pendingPrompt_.box.x2, pendingPrompt_.box.y2);
        return;
    }

    if (action == "sam2_segment_mask") {
        pendingPrompt_.type = Sam2PromptType::kMask;
        hasPendingPrompt_ = true;
        lastPromptType_   = Sam2PromptType::kMask;
        OE_LOG_INFO("sam2_mask_prompt: refinement");
        return;
    }
}

// ---------------------------------------------------------------------------
// publishSegmentationMask — publish result on ZMQ
// ---------------------------------------------------------------------------
void Sam2Node::publishSegmentationMask(std::size_t jpegSize,
                                        const Sam2Result& result)
{
    nlohmann::json msg;
    msg["v"]           = kSchemaVersion;
    msg["type"]        = "segmentation_mask";
    msg["seq"]         = frameSeq_;
    msg["prompt_type"] = static_cast<int>(lastPromptType_);
    msg["iou_score"]   = result.iouScore;
    msg["stability"]   = result.stability;
    msg["mask_width"]  = result.maskWidth;
    msg["mask_height"] = result.maskHeight;
    msg["jpeg_size"]   = jpegSize;
    msg["shm"]         = config_.outputShmName;
    msg["module"]      = config_.moduleName;

    const std::string topic(kZmqTopicSegmentationMask);
    messageRouter_.publish(topic, msg);

    OE_LOG_DEBUG("sam2_published: seq={}, iou={:.2f}, size={}",
                 frameSeq_, result.iouScore, jpegSize);
}

