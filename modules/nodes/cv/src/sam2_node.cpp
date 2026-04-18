// ---------------------------------------------------------------------------
// Sam2Node — SAM 2 interactive segmentation pipeline stage
//
// Input:  BGR24 via ShmCircularBuffer from /oe.vid.ingest (or upstream node)
// GPU:    SAM 2 encoder + decoder with user-provided point/box prompts
// Output: JPEG to /oe.cv.sam2.jpeg (WebSocketBridge display), or
//         BGR24 to /oe.cv.sam2.bgr (downstream chaining)
// ---------------------------------------------------------------------------
#include "cv/sam2_node.hpp"

#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"


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
    auto transport = oe::cv::configureCvTransport(
        config_.moduleName, config_.inputShmName, config_.outputFormat,
        config_.outputBgrShmName, config_.outputSlotCount, config_.outputShmName);
    if (!transport) return tl::unexpected(transport.error());
    shmIn_     = std::move(transport->shmIn);
    shmOut_    = std::move(transport->shmOutJpeg);
    shmOutBgr_ = std::move(transport->shmOutBgr);

    // ZMQ subscriptions via MessageRouter (configurable input topic)
    messageRouter_.subscribe(config_.videoSubPort, config_.inputTopic, /*conflate=*/true,
        [this](const nlohmann::json& msg) { processSegmentation(msg); });
    messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
        [this](const nlohmann::json& msg) { handleUiCommand(msg); });
    messageRouter_.subscribe(config_.daemonSubPort, "sam2_prompt", /*conflate=*/false,
        [this](const nlohmann::json& msg) { handleUiCommand(msg); });

    OE_LOG_INFO("sam2_configured: pub={}, video_sub={}, ws_sub={}, shm_in={}, shm_out={}, enabled={}, output={}, input_topic={}",
        config_.pubPort, config_.videoSubPort, config_.wsBridgeSubPort,
        config_.inputShmName, config_.outputShmName, sam2Enabled_,
        outputFormatName(config_.outputFormat), config_.inputTopic);

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

    // Apply initial background mode to inferencer
    inferencer_->setBackgroundMode(bgMode_);
    inferencer_->setBackgroundColor(bgColorR_, bgColorG_, bgColorB_);

    OE_LOG_INFO("sam2_model_loaded: encoder={}, decoder={}, vram_mib={}, bg_mode={}",
        config_.encoderOnnxPath, config_.decoderOnnxPath,
        inferencer_->currentVramUsageBytes() / (1024 * 1024),
        sam2BgModeName(bgMode_));

    return {};
}

// ---------------------------------------------------------------------------
// processSegmentation — run on each video_frame when a prompt is pending
//                       or continuous tracking is active
// ---------------------------------------------------------------------------
void Sam2Node::processSegmentation(const nlohmann::json& /*frameMetadata*/)
{
    OE_ZONE_SCOPED;

    if (!sam2Enabled_ || !inferencer_) return;

    // Snapshot all shared state under a single lock — avoids TOCTOU races
    // between processSegmentation and handleUiCommand.
    Sam2Prompt localPrompt;
    bool localHasPending;
    bool localIsTracking;
    {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        localHasPending = hasPendingPrompt_;
        localIsTracking = isTracking_;
        localPrompt     = pendingPrompt_;
    }

    // Gate: must have a new prompt or be in tracking mode
    if (!localHasPending && !localIsTracking) return;

    // Frame pacing for continuous tracking — skip frames to maintain target fps.
    // In one-shot mode (hasPendingPrompt && !isTracking), process immediately.
    if (localIsTracking && !localHasPending) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastFrameTime_);
        if (elapsed.count() < kSam2FramePacingMs) {
            return;  // too soon — skip this frame
        }
    }

    OE_LOG_DEBUG("sam2_process: seq={}, prompt_type={}, tracking={}",
                 frameSeq_, static_cast<int>(localPrompt.type), localIsTracking);

    const auto frameStartTime = std::chrono::steady_clock::now();

    // 1. Read latest BGR24 frame from input SHM
    const auto frame = readLatestBgrFrame(*shmIn_);
    if (!frame.data) return;

    // 2. Pre-flight VRAM check
    auto vramCheck = ensureVramAvailableMiB(kSam2InferenceHeadroomMiB);
    if (!vramCheck.has_value()) {
        OE_LOG_WARN("sam2_vram_insufficient: {}", vramCheck.error());
        return;
    }

    // 3. Run inference — output path depends on outputFormat
    if (config_.outputFormat == OutputFormat::kBgr24) {
        // --- BGR24 pipeline chaining path ---
        auto [slotPtr, slotIdx] = shmOutBgr_->acquireWriteSlot();

        auto result = inferencer_->processFrameGetBgr(
            frame.data, frame.width, frame.height,
            localPrompt,
            slotPtr, kMaxBgr24FrameBytes);

        if (!result.has_value()) {
            OE_LOG_ERROR("sam2_process_bgr_failed: {}", result.error());
            return;
        }

        // Stale-read guard
        if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

        // Update header and commit
        auto* hdr = shmOutBgr_->header();
        hdr->width        = frame.width;
        hdr->height       = frame.height;
        hdr->bytesPerPixel = kBgr24BytesPerPixel;
        hdr->seqNumber    = ++frameSeq_;
        hdr->timestampNs  = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        shmOutBgr_->commitWrite();

        OE_LOG_DEBUG("sam2_bgr_frame_done: slot={}, seq={}, {}x{}",
                     slotIdx, frameSeq_, frame.width, frame.height);
        oe::cv::publishBgrFrame(messageRouter_, config_.bgrTopic,
            config_.outputBgrShmName, frameSeq_, frame.width, frame.height);
    } else {
        // --- JPEG display path (default) ---
        auto* ctrl = reinterpret_cast<ShmJpegControl*>(shmOut_->bytes());
        const uint32_t writeSlot =
            1u - ctrl->writeIndex.load(std::memory_order_relaxed);
        uint8_t* outSlot = shmOut_->bytes()
                           + sizeof(ShmJpegControl)
                           + writeSlot * kMaxJpegBytesPerSlot;

        // Run full pipeline: encode + segment + overlay + JPEG
        //    No lock held during inference — prompt was copied above.
        auto result = inferencer_->processFrame(
            frame.data, frame.width, frame.height,
            localPrompt,
            outSlot, kMaxJpegBytesPerSlot);

        if (!result.has_value()) {
            OE_LOG_ERROR("sam2_process_failed: {}", result.error());
            return;
        }

        const std::size_t jpegSize = result.value();

        // Deadline check — if this frame took too long in continuous mode,
        //    log a warning but still output it (don't drop completed work).
        const auto frameEndTime = std::chrono::steady_clock::now();
        const auto frameMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEndTime - frameStartTime).count();
        if (localIsTracking && frameMs > kSam2ContinuousDeadlineMs) {
            OE_LOG_WARN("sam2_deadline_exceeded: frame_ms={}, deadline_ms={}",
                        frameMs, kSam2ContinuousDeadlineMs);
        }

        // processFrame() synchronises internally before writing JPEG to outBuf
        //    (host memory), so no additional GPU fence is needed here.

        // Stale-read guard — discard if input was overwritten during inference
        if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) return;

        // Commit JPEG to output SHM and publish
        ctrl->jpegSize[writeSlot]  = static_cast<uint32_t>(jpegSize);
        ctrl->seqNumber[writeSlot] = ++frameSeq_;
        ctrl->writeIndex.store(writeSlot, std::memory_order_release);

        // Retrieve segmentation metadata from the inferencer's cached result.
        // processFrame() already ran the full pipeline — lastSegmentResult()
        // returns the stored scores without re-running inference.
        const auto& segMeta = inferencer_->lastSegmentResult();
        publishSegmentationMask(jpegSize, segMeta);
    }

    lastFrameTime_ = std::chrono::steady_clock::now();

    // Transition state under lock: first prompt triggers continuous tracking,
    // subsequent prompts update the tracked prompt.
    {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        if (hasPendingPrompt_ && !isTracking_) {
            // First prompt triggers continuous tracking
            isTracking_ = true;
            hasPendingPrompt_ = false;
            OE_LOG_INFO("sam2_tracking_started: prompt_type={}",
                        static_cast<int>(lastPromptType_));
        } else if (hasPendingPrompt_) {
            // New prompt arrived while already tracking — apply it and continue
            hasPendingPrompt_ = false;
            OE_LOG_INFO("sam2_prompt_updated: prompt_type={}",
                        static_cast<int>(lastPromptType_));
        }
    }
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
        Sam2PointPrompt pt;
        pt.x     = std::clamp(cmd.value("x", 0.5f), 0.0f, 1.0f);
        pt.y     = std::clamp(cmd.value("y", 0.5f), 0.0f, 1.0f);
        pt.label = cmd.value("label", 1);
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            pendingPrompt_.type = Sam2PromptType::kPoint;
            pendingPrompt_.points.clear();
            pendingPrompt_.points.push_back(pt);
            hasPendingPrompt_ = true;
            lastPromptType_   = Sam2PromptType::kPoint;
        }
        OE_LOG_INFO("sam2_point_prompt: ({:.3f}, {:.3f}) label={}",
                    pt.x, pt.y, pt.label);
        return;
    }

    if (action == "sam2_segment_box") {
        const float x1 = std::clamp(cmd.value("x1", 0.0f), 0.0f, 1.0f);
        const float y1 = std::clamp(cmd.value("y1", 0.0f), 0.0f, 1.0f);
        const float x2 = std::clamp(cmd.value("x2", 1.0f), 0.0f, 1.0f);
        const float y2 = std::clamp(cmd.value("y2", 1.0f), 0.0f, 1.0f);
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            pendingPrompt_.type = Sam2PromptType::kBox;
            pendingPrompt_.box.x1 = x1;
            pendingPrompt_.box.y1 = y1;
            pendingPrompt_.box.x2 = x2;
            pendingPrompt_.box.y2 = y2;
            hasPendingPrompt_ = true;
            lastPromptType_   = Sam2PromptType::kBox;
        }
        OE_LOG_INFO("sam2_box_prompt: ({:.3f},{:.3f})->({:.3f},{:.3f})",
                    x1, y1, x2, y2);
        return;
    }

    if (action == "sam2_segment_mask") {
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            pendingPrompt_.type = Sam2PromptType::kMask;
            hasPendingPrompt_ = true;
            lastPromptType_   = Sam2PromptType::kMask;
        }
        OE_LOG_INFO("sam2_mask_prompt: refinement");
        return;
    }

    if (action == "set_sam2_bg_mode") {
        const std::string mode = cmd.value("mode", std::string{"overlay"});
        const auto parsed = parseSam2BgMode(mode);
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            bgMode_ = parsed;
        }
        if (inferencer_) {
            inferencer_->setBackgroundMode(parsed);
        }
        OE_LOG_INFO("sam2_bg_mode: {}", sam2BgModeName(parsed));
        return;
    }

    if (action == "set_sam2_bg_color") {
        const auto r = static_cast<uint8_t>(
            std::clamp(cmd.value("r", static_cast<int>(kSam2DefaultBgColorR)), 0, 255));
        const auto g = static_cast<uint8_t>(
            std::clamp(cmd.value("g", static_cast<int>(kSam2DefaultBgColorG)), 0, 255));
        const auto b = static_cast<uint8_t>(
            std::clamp(cmd.value("b", static_cast<int>(kSam2DefaultBgColorB)), 0, 255));
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            bgColorR_ = r;
            bgColorG_ = g;
            bgColorB_ = b;
        }
        if (inferencer_) {
            inferencer_->setBackgroundColor(r, g, b);
        }
        OE_LOG_INFO("sam2_bg_color: ({}, {}, {})", r, g, b);
        return;
    }

    if (action == "set_sam2_bg_image") {
        // Background image uploaded via binary protocol — image_path points to temp file.
        // WS bridge validates JPEG/PNG format upstream before writing the temp file.
        const std::string imagePath = cmd.value("image_path", std::string{});
        if (imagePath.empty()) {
            OE_LOG_WARN("sam2_bg_image: missing image_path");
            return;
        }

        // Read image file into memory
        std::ifstream ifs(imagePath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            OE_LOG_ERROR("sam2_bg_image: cannot open {}", imagePath);
            return;
        }
        const auto fileSize = static_cast<std::size_t>(ifs.tellg());
        if (fileSize == 0 || fileSize > kSam2MaxBgImageBytes) {
            OE_LOG_WARN("sam2_bg_image: invalid size {} bytes (max {})",
                        fileSize, kSam2MaxBgImageBytes);
            return;
        }
        ifs.seekg(0);
        std::vector<uint8_t> imageData(fileSize);
        ifs.read(reinterpret_cast<char*>(imageData.data()),
                 static_cast<std::streamsize>(fileSize));
        if (!ifs || ifs.gcount() != static_cast<std::streamsize>(fileSize)) {
            OE_LOG_ERROR("sam2_bg_image: read failed for {} (got {}/{})",
                         imagePath, ifs.gcount(), fileSize);
            return;
        }

        // Auto-switch to image background mode
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            bgMode_ = Sam2BackgroundMode::kImage;
        }
        if (inferencer_) {
            inferencer_->setBackgroundMode(Sam2BackgroundMode::kImage);
            inferencer_->setBackgroundImage(imageData, 0, 0);
        }
        OE_LOG_INFO("sam2_bg_image: loaded {} bytes from {}", fileSize, imagePath);
        return;
    }

    if (action == "sam2_stop_tracking") {
        {
            const std::lock_guard<std::mutex> lock(promptMutex_);
            isTracking_ = false;
            hasPendingPrompt_ = false;
        }
        OE_LOG_INFO("sam2_tracking_stopped");
        return;
    }
}

// ---------------------------------------------------------------------------
// publishSegmentationMask — publish result on ZMQ
// ---------------------------------------------------------------------------
void Sam2Node::publishSegmentationMask(std::size_t jpegSize,
                                        const Sam2Result& result)
{
    // Snapshot shared state under lock for the JSON message.
    Sam2PromptType localPromptType;
    Sam2BackgroundMode localBgMode;
    bool localIsTracking;
    {
        const std::lock_guard<std::mutex> lock(promptMutex_);
        localPromptType = lastPromptType_;
        localBgMode     = bgMode_;
        localIsTracking = isTracking_;
    }

    nlohmann::json msg;
    msg["v"]           = kSchemaVersion;
    msg["type"]        = "segmentation_mask";
    msg["seq"]         = frameSeq_;
    msg["prompt_type"] = static_cast<int>(localPromptType);
    msg["iou_score"]   = result.iouScore;
    msg["stability"]   = result.stability;
    msg["mask_width"]  = result.maskWidth;
    msg["mask_height"] = result.maskHeight;
    msg["jpeg_size"]   = jpegSize;
    msg["shm"]         = config_.outputShmName;
    msg["module"]      = config_.moduleName;
    msg["bg_mode"]     = std::string(sam2BgModeName(localBgMode));
    msg["tracking"]    = localIsTracking;

    const std::string topic(kZmqTopicSegmentationMask);
    messageRouter_.publish(topic, msg);

    OE_LOG_DEBUG("sam2_published: seq={}, iou={:.2f}, size={}",
                 frameSeq_, result.iouScore, jpegSize);
}


