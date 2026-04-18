#include "cv/security_camera_node.hpp"

#include <chrono>
#include <ctime>
#include <thread>

#include <spdlog/spdlog.h>

#include "common/oe_tracy.hpp"
#include "cv/cv_transport_helpers.hpp"
#include "common/oe_shm_helpers.hpp"
#include "common/zmq_messages.hpp"
#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_frame_writer.hpp"

namespace {

/// Map a subset of COCO class IDs to human-readable names.
constexpr std::array<std::pair<int, const char*>, 80> kCocoClassNames = {{
    {0, "person"}, {1, "bicycle"}, {2, "car"}, {3, "motorcycle"},
    {4, "airplane"}, {5, "bus"}, {6, "train"}, {7, "truck"},
    {8, "boat"}, {9, "traffic light"}, {10, "fire hydrant"},
    {11, "stop sign"}, {12, "parking meter"}, {13, "bench"},
    {14, "bird"}, {15, "cat"}, {16, "dog"}, {17, "horse"},
    {18, "sheep"}, {19, "cow"}, {20, "elephant"}, {21, "bear"},
    {22, "zebra"}, {23, "giraffe"}, {24, "backpack"}, {25, "umbrella"},
    {26, "handbag"}, {27, "tie"}, {28, "suitcase"}, {29, "frisbee"},
    {30, "skis"}, {31, "snowboard"}, {32, "sports ball"}, {33, "kite"},
    {34, "baseball bat"}, {35, "baseball glove"}, {36, "skateboard"},
    {37, "surfboard"}, {38, "tennis racket"}, {39, "bottle"},
    {40, "wine glass"}, {41, "cup"}, {42, "fork"}, {43, "knife"},
    {44, "spoon"}, {45, "bowl"}, {46, "banana"}, {47, "apple"},
    {48, "sandwich"}, {49, "orange"}, {50, "broccoli"}, {51, "carrot"},
    {52, "hot dog"}, {53, "pizza"}, {54, "donut"}, {55, "cake"},
    {56, "chair"}, {57, "couch"}, {58, "potted plant"}, {59, "bed"},
    {60, "dining table"}, {61, "toilet"}, {62, "tv"}, {63, "laptop"},
    {64, "mouse"}, {65, "remote"}, {66, "keyboard"}, {67, "cell phone"},
    {68, "microwave"}, {69, "oven"}, {70, "toaster"}, {71, "sink"},
    {72, "refrigerator"}, {73, "book"}, {74, "clock"}, {75, "vase"},
    {76, "scissors"}, {77, "teddy bear"}, {78, "hair drier"}, {79, "toothbrush"},
}};

}  // anonymous namespace

// ── Config Validation ───────────────────────────────────────────────────────

tl::expected<SecurityCameraNode::Config, std::string>
SecurityCameraNode::Config::validate(const Config& raw) {
    if (raw.yoloEnginePath.empty())
        return tl::unexpected(std::string("yoloEnginePath must not be empty"));
    if (raw.confidenceThreshold < 0.0f || raw.confidenceThreshold > 1.0f)
        return tl::unexpected(std::string("confidenceThreshold must be in [0.0, 1.0]"));
    if (raw.detectionIntervalMs < 10)
        return tl::unexpected(std::string("detectionIntervalMs must be >= 10"));
    if (raw.recordingDir.empty())
        return tl::unexpected(std::string("recordingDir must not be empty"));
    return raw;
}

// ── Construction / Destruction ──────────────────────────────────────────────

SecurityCameraNode::SecurityCameraNode(const Config& config)
    : config_(config)
    , messageRouter_(MessageRouter::Config{
          config.moduleName,
          config.pubPort,
          config.zmqSendHighWaterMark,
          config.pollTimeout})
{
    SPDLOG_INFO("[SecurityCameraNode] constructed — pubPort={}, detectionInterval={}ms, "
                "confidence={:.2f}, recording={}",
                config_.pubPort, config_.detectionIntervalMs,
                config_.confidenceThreshold, config_.recordingDir);
}

SecurityCameraNode::~SecurityCameraNode() {
    stop();
}

// ── Public API ──────────────────────────────────────────────────────────────

void SecurityCameraNode::initialize() {
    OE_ZONE_SCOPED;
    SPDLOG_INFO("[SecurityCameraNode] initializing...");

    // Open input SHM (video ingest circular buffer). Security mode starts
    // shortly after video_ingest, so allow a longer retry window to avoid a
    // fatal initialize failure when camera startup is slow.
    const std::size_t inputShmBytes =
        ShmCircularBuffer<ShmVideoHeader>::segmentSize(
            kCircularBufferSlotCount, kMaxBgr24FrameBytes);
    auto shmResult = oe::shm::openConsumerWithRetry(
        config_.inputShmName, inputShmBytes, config_.moduleName,
        /*maxAttempts=*/120);
    if (!shmResult) {
        throw std::runtime_error(
            "Failed to open input SHM: " + shmResult.error());
    }
    shmIn_ = std::move(*shmResult);
    SPDLOG_INFO("[SecurityCameraNode] opened input SHM: {} ({} bytes)",
                config_.inputShmName, inputShmBytes);

    // Output SHM — mode-dependent
    if (config_.outputFormat == OutputFormat::kBgr24) {
        // BGR24 mode: ShmCircularBuffer for downstream chaining
        shmOutBgr_ = std::make_unique<ShmCircularBuffer<ShmVideoHeader>>(
            config_.outputBgrShmName,
            config_.outputSlotCount,
            kMaxBgr24FrameBytes,
            /*create=*/true);
        SPDLOG_INFO("[SecurityCameraNode] created output BGR24 SHM: {}, slots={}",
            config_.outputBgrShmName, config_.outputSlotCount);
    } else {
        // JPEG mode (default): [uint32 jpegSize][JPEG bytes] in a 4 MiB segment.
        shmOut_ = std::make_unique<ShmMapping>(
            config_.outputShmName, kSecurityJpegShmSize, /*create=*/true);
        SPDLOG_INFO("[SecurityCameraNode] created output SHM: {}", config_.outputShmName);
    }

    // Pre-allocate frame buffer to max supported resolution.
    frameBuf_.resize(kMaxBgr24FrameBytes);
    SPDLOG_INFO("[SecurityCameraNode] frame buffer: {} bytes (max)", frameBuf_.size());

    // Initialise YOLOX-Nano detection inferencer (ONNX Runtime + CUDA EP).
    inferencer_ = std::make_unique<OnnxSecurityInferencer>();
    auto loadResult = inferencer_->loadEngine(config_.yoloEnginePath);
    if (!loadResult) {
        throw std::runtime_error(
            "Failed to load YOLOX-Nano inferencer: " + loadResult.error());
    }
    SPDLOG_INFO("[SecurityCameraNode] YOLOX-Nano loaded: {}", config_.yoloEnginePath);

    // Initialise subsystems.
    eventLogger_ = std::make_unique<oe::security::SecurityEventLogger>(
        config_.logDir, std::string(kSecurityEventLogFileName));

    recorder_ = std::make_unique<oe::security::SecurityRecorder>();

    oe::security::SecurityDetectionAnnotator::Config annotatorCfg;
    annotatorCfg.jpegQuality = config_.jpegQuality;
    annotator_ = std::make_unique<oe::security::SecurityDetectionAnnotator>(annotatorCfg);

    // Subscribe to video frame notifications (configurable input topic).
    messageRouter_.subscribe(config_.videoSubPort, config_.inputTopic, true,
        [this](const nlohmann::json& /*msg*/) {
            processFrame();
        });

    // Security commands are replayed by the daemon after security mode is
    // active so the freshly spawned node does not miss the user's first click.
    messageRouter_.subscribe(config_.daemonSubPort, "security_command", false,
        [this](const nlohmann::json& msg) {
            const auto action = msg.value("action", "");
            if (action == "toggle_security_mode") {
                handleToggleDetection(msg);
            } else if (action == "security_list_recordings") {
                handleListRecordings(msg);
            } else if (action == "security_list_events") {
                handleListEvents(msg);
            } else if (action == "security_update_classes") {
                handleUpdateClasses(msg);
            } else if (action == "security_set_style") {
                handleSetStyle(msg);
            } else if (action == "security_set_roi") {
                handleSetRoi(msg);
            }
        });

    // Purge recordings older than the retention period on startup.
    int purged = oe::security::SecurityRecorder::purgeOldRecordings(
        config_.recordingDir, kSecurityRecordingRetentionDays);
    if (purged > 0) {
        SPDLOG_INFO("[SecurityCameraNode] purged {} old recording(s) on startup", purged);
    }
    lastPurgeTime_ = std::chrono::steady_clock::now();

    SPDLOG_INFO("[SecurityCameraNode] initialization complete");
}

void SecurityCameraNode::run() {
    OE_ZONE_SCOPED;
    running_.store(true, std::memory_order_release);
    SPDLOG_INFO("[SecurityCameraNode] entering event loop");
    messageRouter_.run();
    SPDLOG_INFO("[SecurityCameraNode] event loop exited");
}

void SecurityCameraNode::stop() noexcept {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    SPDLOG_INFO("[SecurityCameraNode] stopping...");
    messageRouter_.stop();
    if (recorder_ && recorder_->isRecording()) {
        recorder_->stop();
    }
}

// ── Hot Path ────────────────────────────────────────────────────────────────

void SecurityCameraNode::processFrame() {
    OE_ZONE_SCOPED;

    if (!detectionActive_) return;

    // Rate-limit YOLO to detectionIntervalMs.
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastDetectionTime_).count();
    if (elapsed < config_.detectionIntervalMs) return;
    lastDetectionTime_ = now;

    // Periodic recording purge — every 6 hours.
    auto sincePurge = std::chrono::duration_cast<std::chrono::hours>(
        now - lastPurgeTime_).count();
    if (sincePurge >= 6) {
        lastPurgeTime_ = now;
        int purged = oe::security::SecurityRecorder::purgeOldRecordings(
            config_.recordingDir, kSecurityRecordingRetentionDays);
        if (purged > 0) {
            SPDLOG_INFO("[SecurityCameraNode] periodic purge removed {} recording(s)", purged);
        }
    }

    // Read latest frame from SHM circular buffer.
    if (!shmIn_) return;
    const auto frame = readLatestBgrFrame(*shmIn_);
    if (!frame.data || frame.width == 0 || frame.height == 0) return;

    const std::size_t frameBytes = static_cast<std::size_t>(frame.width) *
                                   frame.height * kBgr24BytesPerPixel;
    if (frameBytes > frameBuf_.size()) {
        SPDLOG_WARN("[SecurityCameraNode] frame too large: {} bytes (max {})",
                    frameBytes, frameBuf_.size());
        return;
    }
    std::memcpy(frameBuf_.data(), frame.data, frameBytes);
    lastFrameWidth_ = frame.width;
    lastFrameHeight_ = frame.height;
    ++frameSeq_;

    // Filter detections by target classes and confidence.
    std::vector<oe::security::DetectionBox> filteredDetections;
    nlohmann::json detectionsJson = nlohmann::json::array();
    try {
        auto inferResult = inferencer_->infer(frameBuf_.data(), frame.width, frame.height);
        if (!inferResult) {
            SPDLOG_WARN("[SecurityCameraNode] YOLOX inference failed: {}", inferResult.error());
        } else {
            for (const auto& det : inferResult->detections) {
                if (det.confidence < config_.confidenceThreshold) continue;
                if (!isTargetClass(det.classId)) continue;

                // Compute normalised bbox center for ROI filtering.
                float centerX = (det.bbox.x + det.bbox.w * 0.5f) / static_cast<float>(frame.width);
                float centerY = (det.bbox.y + det.bbox.h * 0.5f) / static_cast<float>(frame.height);
                if (annotator_ && !annotator_->isInsideRoi(centerX, centerY)) continue;

                std::string className = cocoClassName(det.classId);

                oe::security::DetectionBox box;
                box.x          = det.bbox.x / static_cast<float>(frame.width);
                box.y          = det.bbox.y / static_cast<float>(frame.height);
                box.w          = det.bbox.w / static_cast<float>(frame.width);
                box.h          = det.bbox.h / static_cast<float>(frame.height);
                box.confidence = det.confidence;
                box.className  = className;
                filteredDetections.push_back(box);

                detectionsJson.push_back({
                    {"class",      className},
                    {"confidence", det.confidence},
                    {"bbox",       {{"x", box.x}, {"y", box.y}, {"w", box.w}, {"h", box.h}}},
                });

                // Event cooldown — publish high-level events with per-class dedup.
                auto& lastTime = lastEventTime_[className];
                auto cooldownMs = std::chrono::milliseconds(config_.eventCooldownMs);
                if (now - lastTime >= cooldownMs) {
                    lastTime = now;

                    // Log event.
                    oe::security::SecurityEvent event;
                    event.timestamp     = nowIso8601();
                    event.detectedClass = className;
                    event.confidence    = det.confidence;
                    event.bboxX         = box.x;
                    event.bboxY         = box.y;
                    event.bboxW         = box.w;
                    event.bboxH         = box.h;
                    event.recordingFile = recorder_ ? recorder_->currentFile() : "";
                    event.frameSeq      = frameSeq_;

                    if (auto appendResult = eventLogger_->appendEvent(event); !appendResult) {
                        SPDLOG_WARN("[SecurityCameraNode] event log failed: {}",
                                    appendResult.error());
                    }

                    // Publish high-level security_event.
                    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
                        {"v", 1},
                        {"type", "security_event"},
                        {"event", className + "_detected"},
                        {"confidence", det.confidence},
                        {"timestamp", event.timestamp},
                        {"recording", event.recordingFile},
                    });

                    SPDLOG_INFO("[SecurityCameraNode] DETECTION: {} ({:.0f}%) at frame #{}",
                                className, det.confidence * 100.0f, frameSeq_);
                }
            }
        }
    } catch (const std::exception& e) {
        SPDLOG_WARN("[SecurityCameraNode] YOLOX inference exception: {}", e.what());
    }

    // Publish every processed frame, even when the scene is empty. The bridge
    // keys /security_video relay off this topic, so skipping empty payloads
    // leaves the security canvas blank until the first detection arrives.
    messageRouter_.publish(std::string(kZmqTopicSecurityDetection), nlohmann::json{
        {"v", 1},
        {"type", "security_detection"},
        {"detections", detectionsJson},
        {"timestamp", nowIso8601()},
        {"frame_seq", frameSeq_},
    });

    // Annotate frame and write to output SHM — mode-dependent.
    if (config_.outputFormat == OutputFormat::kBgr24) {
        // BGR24 pipeline chaining: write the unannotated BGR24 frame to the
        // circular buffer. Downstream nodes do their own composition/overlay;
        // drawing bboxes here would baked-in annotations into a shared video
        // stream other pipelines don't want.
        auto [slotPtr, slotIdx] = shmOutBgr_->acquireWriteSlot();
        std::memcpy(slotPtr, frameBuf_.data(), frameBytes);

        auto* hdr = shmOutBgr_->header();
        hdr->width        = frame.width;
        hdr->height       = frame.height;
        hdr->bytesPerPixel = kBgr24BytesPerPixel;
        hdr->seqNumber    = static_cast<uint32_t>(frameSeq_);
        hdr->timestampNs  = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
        shmOutBgr_->commitWrite();

        oe::cv::publishBgrFrame(messageRouter_, config_.bgrTopic,
            config_.outputBgrShmName, frameSeq_, frame.width, frame.height);
    } else {
        // JPEG mode (default): annotate + encode JPEG to SHM
        auto jpegResult = annotator_->annotate(frameBuf_.data(),
                                                static_cast<int>(frame.width),
                                                static_cast<int>(frame.height),
                                                filteredDetections);
        if (jpegResult) {
            if (shmOut_ && shmOut_->bytes()) {
                auto& jpeg = *jpegResult;
                // Write JPEG size (4 bytes) + JPEG data to SHM.
                uint32_t jpegSize = static_cast<uint32_t>(jpeg.size());
                auto* base = shmOut_->bytes();
                std::memcpy(base, &jpegSize, sizeof(jpegSize));
                std::memcpy(base + sizeof(jpegSize), jpeg.data(), jpeg.size());
            }
        }
    }

    // Feed raw frame to recorder.
    if (recorder_ && recorder_->isRecording()) {
        auto tsNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
        if (auto pushResult = recorder_->pushFrame(frameBuf_.data(), frameBytes, tsNs);
            !pushResult) {
            SPDLOG_WARN("[SecurityCameraNode] recorder pushFrame failed: {}",
                        pushResult.error());
        }
    }

    OE_FRAME_MARK;
}

// ── UI Command Handlers ─────────────────────────────────────────────────────

void SecurityCameraNode::handleToggleDetection(const nlohmann::json& msg) {
    bool enabled = msg.value("enabled", true);
    detectionActive_ = enabled;

    if (enabled && !recordingActive_) {
        // Start recording.
        oe::security::SecurityRecorder::Config recCfg;
        recCfg.recordingDir       = config_.recordingDir;
        recCfg.fps                = config_.recordingFps;
        recCfg.width              = static_cast<int>(lastFrameWidth_ ? lastFrameWidth_ : config_.inputWidth);
        recCfg.height             = static_cast<int>(lastFrameHeight_ ? lastFrameHeight_ : config_.inputHeight);
        recCfg.bitrate            = config_.recordingBitrate;
        recCfg.segmentDurationMin = config_.segmentDurationMin;

        if (auto r = recorder_->start(recCfg); r) {
            recordingActive_ = true;
            SPDLOG_INFO("[SecurityCameraNode] recording started");
        } else {
            SPDLOG_ERROR("[SecurityCameraNode] recording start failed: {}", r.error());
        }
    } else if (!enabled && recordingActive_) {
        recorder_->stop();
        recordingActive_ = false;
        SPDLOG_INFO("[SecurityCameraNode] recording stopped");
    }

    // Publish status.
    messageRouter_.publish(std::string(kZmqTopicSecurityRecordingStatus), nlohmann::json{
        {"v", 1},
        {"type", "security_recording_status"},
        {"detection_active", detectionActive_},
        {"recording", recordingActive_},
        {"file", recorder_ ? recorder_->currentFile() : ""},
    });

    SPDLOG_INFO("[SecurityCameraNode] detection={}, recording={}", detectionActive_, recordingActive_);
}

void SecurityCameraNode::handleListRecordings(const nlohmann::json& /*msg*/) {
    auto recordings = oe::security::SecurityRecorder::listRecordings(config_.recordingDir);

    nlohmann::json list = nlohmann::json::array();
    for (const auto& rec : recordings) {
        list.push_back({
            {"file",       rec.file},
            {"start",      rec.startTime},
            {"end",        rec.endTime},
            {"size_mb",    rec.sizeMB},
            {"events",     rec.eventCount},
        });
    }

    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
        {"v", 1},
        {"type", "security_recordings_list"},
        {"recordings", list},
    });
}

void SecurityCameraNode::handleListEvents(const nlohmann::json& msg) {
    std::string from = msg.value("from", "");
    std::string to   = msg.value("to", "");

    auto result = eventLogger_->queryEvents(from, to);
    if (!result) {
        SPDLOG_WARN("[SecurityCameraNode] event query failed: {}", result.error());
        return;
    }

    nlohmann::json list = nlohmann::json::array();
    for (const auto& ev : *result) {
        list.push_back({
            {"timestamp",  ev.timestamp},
            {"class",      ev.detectedClass},
            {"confidence", ev.confidence},
            {"recording",  ev.recordingFile},
        });
    }

    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
        {"v", 1},
        {"type", "security_events_list"},
        {"events", list},
    });
}

void SecurityCameraNode::handleUpdateClasses(const nlohmann::json& msg) {
    if (!msg.contains("classes") || !msg["classes"].is_array()) {
        SPDLOG_WARN("[SecurityCameraNode] security_update_classes: missing 'classes' array");
        return;
    }

    std::vector<std::string> newClasses;
    for (const auto& cls : msg["classes"]) {
        if (cls.is_string()) {
            newClasses.push_back(cls.get<std::string>());
        }
    }

    config_.targetClasses = std::move(newClasses);

    SPDLOG_INFO("[SecurityCameraNode] target classes updated: [{}]",
                [&]() {
                    std::string s;
                    for (std::size_t i = 0; i < config_.targetClasses.size(); ++i) {
                        if (i > 0) s += ", ";
                        s += config_.targetClasses[i];
                    }
                    return s;
                }());

    // Acknowledge to frontend with the active class list.
    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
        {"v", 1},
        {"type", "security_classes_updated"},
        {"classes", config_.targetClasses},
    });
}

void SecurityCameraNode::handleSetStyle(const nlohmann::json& msg) {
    std::string styleName = msg.value("style", "corner_bracket");

    oe::security::AnnotationStyle style = oe::security::AnnotationStyle::kCornerBracket;
    if (styleName == "rectangle") {
        style = oe::security::AnnotationStyle::kRectangle;
    } else if (styleName == "crosshair") {
        style = oe::security::AnnotationStyle::kCrosshair;
    }

    if (annotator_) {
        annotator_->setStyle(style);
    }

    SPDLOG_INFO("[SecurityCameraNode] annotation style set to: {}", styleName);

    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
        {"v", 1},
        {"type", "security_style_updated"},
        {"style", styleName},
    });
}

void SecurityCameraNode::handleSetRoi(const nlohmann::json& msg) {
    std::vector<std::pair<float, float>> points;

    if (msg.contains("points") && msg["points"].is_array()) {
        for (const auto& pt : msg["points"]) {
            float x = pt.value("x", 0.0f);
            float y = pt.value("y", 0.0f);
            points.emplace_back(x, y);
        }
    }

    if (annotator_) {
        annotator_->setRoi(std::move(points));
    }

    SPDLOG_INFO("[SecurityCameraNode] ROI set with {} points",
                annotator_ ? annotator_->roi().size() : 0u);

    messageRouter_.publish(std::string(kZmqTopicSecurityEvent), nlohmann::json{
        {"v", 1},
        {"type", "security_roi_updated"},
        {"point_count", annotator_ ? static_cast<int>(annotator_->roi().size()) : 0},
    });
}

// ── Utilities ───────────────────────────────────────────────────────────────

std::string SecurityCameraNode::nowIso8601() {
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);

    char result[40];
    std::snprintf(result, sizeof(result), "%s.%03dZ", buf,
                  static_cast<int>(ms.count()));
    return result;
}

bool SecurityCameraNode::isTargetClass(int classId) const {
    std::string name = cocoClassName(classId);
    for (const auto& target : config_.targetClasses) {
        if (target == name) return true;
    }
    return false;
}

std::string SecurityCameraNode::cocoClassName(int classId) {
    for (const auto& [id, name] : kCocoClassNames) {
        if (id == classId) return name;
    }
    return "class_" + std::to_string(classId);
}
