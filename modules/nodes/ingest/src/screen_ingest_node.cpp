// screen_ingest_node.cpp -- ScreenIngestNode full implementation

#include "ingest/screen_ingest_node.hpp"

#include "ingest/tcp_frame_receiver.hpp"

#include <chrono>
#include <cstring>
#include <format>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/constants/ingest_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/time_utils.hpp"
#include "common/zmq_messages.hpp"


namespace {

constexpr uint32_t kScreenSlotCount = kCircularBufferSlotCount;  // 4

/// Maximum sane resolution (8K) — reject frames beyond this.
constexpr uint32_t kMaxResolution = 7680;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<ScreenIngestNode::Config, std::string>
ScreenIngestNode::Config::validate(const Config& raw)
{
    ConfigValidator v;
    v.requirePort("pubPort", raw.pubPort);
    v.requireNonEmpty("windowsHostIp", raw.windowsHostIp);
    v.requirePositive("tcpPort", raw.tcpPort);
    v.requirePositive("healthTimeoutSec", raw.healthTimeoutSec);

    if (auto err = v.finish(); !err.empty()) {
        return tl::unexpected(err);
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ScreenIngestNode::ScreenIngestNode(const Config& config)
    : config_(config)
    , messageRouter_(MessageRouter::Config{
        .moduleName  = config.moduleName,
        .pubPort     = config.pubPort,
        .pubHwm      = kPublisherDataHighWaterMark,
        .pollTimeout = config.pollTimeout,
    })
{
}

ScreenIngestNode::~ScreenIngestNode()
{
    stop();
}

// ---------------------------------------------------------------------------
// configureTransport
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ScreenIngestNode::configureTransport()
{
    OE_ZONE_SCOPED;
    OeLogger::instance().setModule(config_.moduleName);
    OE_LOG_INFO("screen_ingest_configuring: host={}, port={}",
               config_.windowsHostIp, config_.tcpPort);

    // SHM is NOT created here — it's created lazily in processPendingFrame()
    // when the first frame arrives and we know the resolution.

    // Register per-iteration callback: drain staged JPEG + health check.
    messageRouter_.setOnPollCallback([this]() -> bool {
        if (pendingFrame_.load(std::memory_order_acquire)) {
            processPendingFrame();
        }
        checkHealthAndPublish();
        return true;
    });

    OE_LOG_INFO("screen_ingest_configured: pub_port={}", config_.pubPort);
    return {};
}

// ---------------------------------------------------------------------------
// loadInferencer — create TcpFrameReceiver (no GPU, no model)
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ScreenIngestNode::loadInferencer()
{
    OE_ZONE_SCOPED;

    tcpReceiver_ = std::make_unique<TcpFrameReceiver>(
        TcpFrameReceiver::Config{
            .host = config_.windowsHostIp,
            .port = config_.tcpPort,
        });

    OE_LOG_INFO("screen_ingest_tcp_receiver_created: {}:{}",
               config_.windowsHostIp, config_.tcpPort);
    return {};
}

// ---------------------------------------------------------------------------
// onBeforeRun — start TCP receiver thread
// ---------------------------------------------------------------------------

void ScreenIngestNode::onBeforeRun()
{
    tcpReceiver_->start(
        [this](std::span<const uint8_t> jpegData, uint32_t w, uint32_t h) {
            onTcpFrame(jpegData, w, h);
        });
    OE_LOG_INFO("screen_ingest_tcp_receiver_started");
}

// ---------------------------------------------------------------------------
// onAfterStop — stop TCP receiver thread
// ---------------------------------------------------------------------------

void ScreenIngestNode::onAfterStop()
{
    if (tcpReceiver_) {
        tcpReceiver_->stop();
    }
    OE_LOG_INFO("screen_ingest_stopped");
}

// ---------------------------------------------------------------------------
// onTcpFrame — called on TCP recv thread, stages JPEG under mutex
// ---------------------------------------------------------------------------

void ScreenIngestNode::onTcpFrame(std::span<const uint8_t> jpegData,
                                  uint32_t width, uint32_t height)
{
    OE_ZONE_SCOPED;

    {
        std::lock_guard lock(jpegMutex_);
        pendingJpeg_.assign(jpegData.begin(), jpegData.end());
        pendingWidth_  = width;
        pendingHeight_ = height;
    }

    pendingFrame_.store(true, std::memory_order_release);
    lastFrameReceivedTs_.store(
        static_cast<uint64_t>(steadyClockNanoseconds()),
        std::memory_order_release);
}

// ---------------------------------------------------------------------------
// processPendingFrame — called on main thread, decodes + writes SHM
// ---------------------------------------------------------------------------

void ScreenIngestNode::processPendingFrame()
{
    OE_ZONE_SCOPED;
    pendingFrame_.store(false, std::memory_order_relaxed);

    // Swap staging buffer under lock (fast, no decode under lock).
    std::vector<uint8_t> localJpeg;
    uint32_t width, height;
    {
        std::lock_guard lock(jpegMutex_);
        localJpeg.swap(pendingJpeg_);
        width  = pendingWidth_;
        height = pendingHeight_;
    }

    if (localJpeg.empty() || width == 0 || height == 0) return;

    // Validate dimensions.
    if (width > kMaxResolution || height > kMaxResolution) {
        OE_LOG_WARN("screen_frame_oversized: {}x{}, skipping", width, height);
        return;
    }

    // Ensure SHM is sized for this resolution.
    ensureShmCapacity(width, height);
    if (!shm_) {
        OE_LOG_WARN("screen_shm_unavailable: cannot write frame");
        return;
    }

    // Decode JPEG → BGR24 using OpenCV (1 FPS, not hot path).
    cv::Mat jpegMat(1, static_cast<int>(localJpeg.size()), CV_8UC1, localJpeg.data());
    cv::Mat bgrMat = cv::imdecode(jpegMat, cv::IMREAD_COLOR);
    if (bgrMat.empty()) {
        OE_LOG_WARN("screen_jpeg_decode_failed: {} bytes", localJpeg.size());
        return;
    }

    // Verify decoded dimensions match the header.
    if (static_cast<uint32_t>(bgrMat.cols) != width ||
        static_cast<uint32_t>(bgrMat.rows) != height) {
        OE_LOG_WARN("screen_resolution_mismatch: header={}x{}, decoded={}x{}",
                   width, height, bgrMat.cols, bgrMat.rows);
        return;
    }

    const std::size_t frameBytes = static_cast<std::size_t>(width) * height * 3;

    // Ensure the decoded mat is contiguous BGR24.
    if (!bgrMat.isContinuous()) {
        bgrMat = bgrMat.clone();
    }

    // Write to SHM circular buffer.
    auto [dst, slotIdx] = shm_->acquireWriteSlot();
    std::memcpy(dst, bgrMat.data, frameBytes);

    auto* videoHeader = shm_->header();
    const uint64_t seq = frameSeq_++;
    const uint64_t ts  = static_cast<uint64_t>(steadyClockNanoseconds());

    videoHeader->seqNumber   = seq;
    videoHeader->timestampNs = ts;

    shm_->commitWrite();

    OE_LOG_DEBUG("screen_frame_written: slot={}, seq={}, {}x{}, {} bytes",
               slotIdx, seq, width, height, frameBytes);

    // Publish ZMQ notification.
    static thread_local ScreenFrameMsg msg;
    msg.seq = seq;
    msg.ts  = ts;
    msg.w   = width;
    msg.h   = height;
    messageRouter_.publish(
        std::string{kZmqTopicScreenFrame},
        nlohmann::json(msg));
}

// ---------------------------------------------------------------------------
// ensureShmCapacity
// ---------------------------------------------------------------------------

void ScreenIngestNode::ensureShmCapacity(uint32_t width, uint32_t height)
{
    if (width == currentShmWidth_ && height == currentShmHeight_ && shm_) return;

    // Destroy old SHM if exists.
    shm_.reset();

    const std::size_t slotSize = static_cast<std::size_t>(width) * height * 3;
    shm_ = std::make_unique<ShmCircularBuffer<ShmVideoHeader>>(
        std::string{kScreenShmName}, kScreenSlotCount, slotSize, /*create=*/true);

    auto* hdr = shm_->header();
    hdr->width         = width;
    hdr->height        = height;
    hdr->bytesPerPixel = 3;

    currentShmWidth_  = width;
    currentShmHeight_ = height;

    OE_LOG_INFO("screen_shm_reallocated: {}x{}, slot_size={}, shm={}",
               width, height, slotSize, kScreenShmName);
}

// ---------------------------------------------------------------------------
// checkHealthAndPublish
// ---------------------------------------------------------------------------

void ScreenIngestNode::checkHealthAndPublish()
{
    const auto nowNs     = static_cast<uint64_t>(steadyClockNanoseconds());
    const auto lastFrame = lastFrameReceivedTs_.load(std::memory_order_acquire);
    const bool tcpConnected = tcpReceiver_ && tcpReceiver_->isConnected();
    const bool hasRecentFrame = (lastFrame > 0) &&
        ((nowNs - lastFrame) <
         static_cast<uint64_t>(config_.healthTimeoutSec) * 1'000'000'000ULL);

    const bool healthy = tcpConnected && hasRecentFrame;

    // Only publish on state change.
    if (healthy == agentConnected_) return;
    agentConnected_ = healthy;

    nlohmann::json status;
    status["v"]         = kSchemaVersion;
    status["type"]      = "screen_health";
    status["connected"] = healthy;

    if (!healthy && !tcpConnected) {
        status["error"] = "Screen capture agent not detected \u2014 "
                          "start oe_screen_capture.exe on Windows";
    } else if (!healthy && !hasRecentFrame) {
        status["error"] = "Screen capture agent connected but not sending frames";
    }

    messageRouter_.publish(
        std::string{kZmqTopicScreenHealth}, status);

    OE_LOG_INFO("screen_health_changed: connected={}", healthy);
}
