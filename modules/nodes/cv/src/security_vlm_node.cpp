#include "cv/security_vlm_node.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/constants/security_constants.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
using Clock = std::chrono::steady_clock;
}

// ───────────────────────────────────────────────────────────────────────────
// Config::validate
// ───────────────────────────────────────────────────────────────────────────
tl::expected<SecurityVlmNode::Config, std::string>
SecurityVlmNode::Config::validate(const Config& raw)
{
    if (raw.modelDir.empty()) {
        return tl::unexpected(std::string("modelDir must be set (path to Gemma vision model)"));
    }
    if (raw.inputWidth == 0 || raw.inputHeight == 0) {
        return tl::unexpected(std::string("inputWidth/inputHeight must be > 0"));
    }
    if (raw.prerollSec < 0 || raw.postrollSec < 0) {
        return tl::unexpected(std::string("prerollSec/postrollSec must be >= 0"));
    }
    if (raw.frameSamples < 1 || raw.frameSamples > 32) {
        return tl::unexpected(std::string("frameSamples must be in [1, 32]"));
    }
    if (raw.ingestFps < 1 || raw.ingestFps > 120) {
        return tl::unexpected(std::string("ingestFps must be in [1, 120]"));
    }
    if (raw.maxPendingEvents < 1) {
        return tl::unexpected(std::string("maxPendingEvents must be >= 1"));
    }
    if (raw.jpegDownscale < 64 || raw.jpegDownscale > 2048) {
        return tl::unexpected(std::string("jpegDownscale must be in [64, 2048]"));
    }
    return raw;
}

// ───────────────────────────────────────────────────────────────────────────
// ctor / dtor
// ───────────────────────────────────────────────────────────────────────────
SecurityVlmNode::SecurityVlmNode(const Config& config)
    : config_(config)
    , messageRouter_(MessageRouter::Config{
          config.moduleName,
          config.pubPort,
          config.zmqSendHighWaterMark,
          config.pollTimeout})
{
    SPDLOG_INFO("[SecurityVlmNode] constructed — pubPort={} preroll={}s postroll={}s "
                "samples={} modelDir={}",
                config_.pubPort, config_.prerollSec, config_.postrollSec,
                config_.frameSamples, config_.modelDir);
}

SecurityVlmNode::~SecurityVlmNode()
{
    stop();
    if (vision_) {
        vision_->stop();
    }
    if (incidentsFile_.is_open()) {
        incidentsFile_.close();
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lifecycle
// ───────────────────────────────────────────────────────────────────────────
void SecurityVlmNode::initialize()
{
    ModuleNodeBase<SecurityVlmNode>::initialize();
}

void SecurityVlmNode::run()
{
    ModuleNodeBase<SecurityVlmNode>::run();
}

void SecurityVlmNode::stop() noexcept
{
    ModuleNodeBase<SecurityVlmNode>::stop();
}

void SecurityVlmNode::onAfterStop()
{
    if (vision_) {
        vision_->stop();
        visionLoaded_ = false;
    }
    if (incidentsFile_.is_open()) {
        incidentsFile_.flush();
    }
}

// ───────────────────────────────────────────────────────────────────────────
// CRTP hooks
// ───────────────────────────────────────────────────────────────────────────
tl::expected<void, std::string>
SecurityVlmNode::configureTransport()
{
    OE_ZONE_SCOPED;

    // Input SHM: /oe.vid.ingest — raw BGR24 latest-frame mapping.
    const std::size_t frameBytes =
        static_cast<std::size_t>(config_.inputWidth) *
        config_.inputHeight * kBgr24BytesPerPixel;

    try {
        shmIn_ = std::make_unique<ShmMapping>(
            config_.inputShmName, kMaxBgr24FrameBytes, /*create=*/false);
    } catch (const std::exception& e) {
        return tl::unexpected(std::string("open input SHM failed: ") + e.what());
    }
    frameBuf_.resize(frameBytes);
    SPDLOG_INFO("[SecurityVlmNode] opened input SHM: {} ({} bytes/frame)",
                config_.inputShmName, frameBytes);

    // In-process JPEG ring capacity = (preroll + postroll) × fps — the bracketed window.
    ringCapacity_ = static_cast<std::size_t>(
        std::max(1, (config_.prerollSec + config_.postrollSec) * config_.ingestFps));

    // Prepare incidents JSONL.
    std::error_code ec;
    std::filesystem::create_directories(config_.logDir, ec);
    const auto incidentsPath =
        std::filesystem::path(config_.logDir) / config_.incidentsFileName;
    incidentsFile_.open(incidentsPath, std::ios::out | std::ios::app);
    if (!incidentsFile_) {
        return tl::unexpected("failed to open incidents log: " + incidentsPath.string());
    }
    SPDLOG_INFO("[SecurityVlmNode] incidents log: {}", incidentsPath.string());

    // Subscribe: video frames (conflated — we only ever need the latest).
    messageRouter_.subscribe(config_.videoSubPort, config_.inputTopic, /*conflate=*/true,
        [this](const nlohmann::json& /*msg*/) { onVideoFrame(); });

    // Subscribe: security_event (no conflate — every trigger matters).
    messageRouter_.subscribe(config_.securityEventSubPort,
        std::string(kZmqTopicSecurityEvent), /*conflate=*/false,
        [this](const nlohmann::json& msg) { onSecurityEvent(msg); });

    SPDLOG_INFO("[SecurityVlmNode] subscribed: videoSubPort={} securityEventSubPort={} "
                "ringCapacity={} frames",
                config_.videoSubPort, config_.securityEventSubPort, ringCapacity_);

    return {};
}

tl::expected<void, std::string>
SecurityVlmNode::loadInferencer()
{
    OE_ZONE_SCOPED;

    vision_ = std::make_unique<omniedge::cv::VisionWorkerClient>();

    // Lazy-load: don't spawn the Python worker at node init; wait for first event.
    // Keeps VRAM free until needed, honoring priority 2 / dynamic VRAM policy.
    SPDLOG_INFO("[SecurityVlmNode] VisionWorkerClient constructed (lazy-load on first event)");
    return {};
}

// ───────────────────────────────────────────────────────────────────────────
// Hot-path handlers
// ───────────────────────────────────────────────────────────────────────────
void SecurityVlmNode::onVideoFrame()
{
    OE_ZONE_SCOPED;

    if (!shmIn_) return;
    auto* data = shmIn_->data();
    if (!data) return;

    const std::size_t frameBytes = frameBuf_.size();
    std::memcpy(frameBuf_.data(), data, frameBytes);
    ++ingestFrameSeq_;

    // Encode to JPEG (downscaled) and push into the ring.
    auto jpeg = encodeJpeg(frameBuf_.data(),
                           config_.inputWidth, config_.inputHeight,
                           static_cast<std::uint32_t>(config_.jpegDownscale),
                           config_.jpegQuality);
    if (jpeg.empty()) {
        SPDLOG_WARN("[SecurityVlmNode] JPEG encode returned empty at seq={}", ingestFrameSeq_);
        return;
    }

    ring_.push_back(RingFrame{
        ingestFrameSeq_,
        Clock::now(),
        std::move(jpeg),
    });
    while (ring_.size() > ringCapacity_) {
        ring_.pop_front();
    }

    // Opportunistic: on every frame, try to drain any pending incidents whose
    // postroll window has filled. This keeps reasoning latency bounded to one
    // ingest-tick after the postroll deadline.
    drainPendingIncidents();
    maybeUnloadIdleVision();
}

void SecurityVlmNode::onSecurityEvent(const nlohmann::json& msg)
{
    OE_ZONE_SCOPED;

    // We only care about the real detection trigger — ignore our own
    // republishing, UI command echoes, etc.
    const std::string type = msg.value("type", "");
    if (type != "security_event") {
        return;
    }
    const std::string evt = msg.value("event", "");
    if (evt.empty() || evt.find("_detected") == std::string::npos) {
        return;
    }

    if (static_cast<int>(pending_.size()) >= config_.maxPendingEvents) {
        SPDLOG_WARN("[SecurityVlmNode] incident_dropped — pending queue full ({}), event={}",
                    pending_.size(), evt);
        return;
    }

    PendingIncident inc{};
    inc.eventId         = makeEventId();
    inc.source          = msg;
    inc.triggerFrameSeq = ingestFrameSeq_;
    inc.arrival         = Clock::now();
    inc.dispatchAt      = inc.arrival + std::chrono::seconds(config_.postrollSec);
    pending_.push_back(std::move(inc));

    SPDLOG_INFO("[SecurityVlmNode] event queued: {} (event_id={} trigger_seq={} postroll={}s)",
                evt, pending_.back().eventId, ingestFrameSeq_, config_.postrollSec);
}

// ───────────────────────────────────────────────────────────────────────────
// Pending incident drain
// ───────────────────────────────────────────────────────────────────────────
void SecurityVlmNode::drainPendingIncidents()
{
    const auto now = Clock::now();
    while (!pending_.empty() && pending_.front().dispatchAt <= now) {
        auto inc = std::move(pending_.front());
        pending_.pop_front();

        if (auto r = ensureVisionLoaded(); !r) {
            SPDLOG_ERROR("[SecurityVlmNode] vision load failed: {} — event_id={}",
                         r.error(), inc.eventId);
            // Still persist the dropped incident so the audit trail survives.
            appendIncidentJsonl({
                {"v", 1},
                {"event_id", inc.eventId},
                {"timestamp", nowIso8601()},
                {"source_event", inc.source},
                {"status", "vision_load_failed"},
                {"error", r.error()},
            });
            continue;
        }

        auto clip = assembleClip(inc.triggerFrameSeq);
        if (clip.empty()) {
            SPDLOG_WARN("[SecurityVlmNode] no frames in ring for event_id={} — skipping",
                        inc.eventId);
            appendIncidentJsonl({
                {"v", 1},
                {"event_id", inc.eventId},
                {"timestamp", nowIso8601()},
                {"source_event", inc.source},
                {"status", "empty_clip"},
            });
            continue;
        }

        SPDLOG_INFO("[SecurityVlmNode] analyzing clip: event_id={} frames={}",
                    inc.eventId, clip.size());

        auto t0 = Clock::now();
        auto result = vision_->analyze(
            std::span<const std::vector<std::uint8_t>>(clip.data(), clip.size()),
            config_.promptTemplate, inc.eventId);
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - t0).count();

        if (!result) {
            SPDLOG_ERROR("[SecurityVlmNode] analyze failed: {} (event_id={})",
                         result.error(), inc.eventId);
            appendIncidentJsonl({
                {"v", 1},
                {"event_id", inc.eventId},
                {"timestamp", nowIso8601()},
                {"source_event", inc.source},
                {"status", "vlm_error"},
                {"error", result.error()},
                {"elapsed_ms", elapsedMs},
            });
            continue;
        }

        lastVisionUse_ = Clock::now();
        ++incidentCount_;

        nlohmann::json analysis = result->result;
        if (!result->rawText.empty()) {
            analysis = nlohmann::json{{"raw_text", result->rawText}};
        }

        publishIncident(inc.eventId, inc.source, analysis, static_cast<int>(clip.size()));

        appendIncidentJsonl({
            {"v", 1},
            {"event_id", inc.eventId},
            {"timestamp", nowIso8601()},
            {"source_event", inc.source},
            {"status", "ok"},
            {"analysis", analysis},
            {"image_count", clip.size()},
            {"elapsed_ms", elapsedMs},
        });
        SPDLOG_INFO("[SecurityVlmNode] incident_published event_id={} elapsed_ms={} count={}",
                    inc.eventId, elapsedMs, incidentCount_);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────
tl::expected<void, std::string>
SecurityVlmNode::ensureVisionLoaded()
{
    if (visionLoaded_ && vision_->isLoaded()) {
        return {};
    }
    omniedge::cv::VisionWorkerClient::Options opts;
    opts.scriptPathOverride = config_.scriptPathOverride;
    opts.modelDir           = config_.modelDir;
    opts.startupTimeout     = std::chrono::seconds(config_.startupTimeoutSec);
    opts.requestTimeout     = std::chrono::milliseconds(config_.requestTimeoutMs);

    auto r = vision_->start(opts);
    if (!r) {
        return tl::unexpected(r.error());
    }
    visionLoaded_  = true;
    lastVisionUse_ = Clock::now();
    SPDLOG_INFO("[SecurityVlmNode] vision worker loaded (model={})", config_.modelDir);
    return {};
}

void SecurityVlmNode::maybeUnloadIdleVision()
{
    if (!visionLoaded_ || !pending_.empty()) return;
    const auto idle =
        std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - lastVisionUse_).count();
    if (idle < config_.idleUnloadSec) return;

    SPDLOG_INFO("[SecurityVlmNode] vision idle for {}s — unloading to free VRAM", idle);
    vision_->stop();
    visionLoaded_ = false;
}

std::vector<std::vector<std::uint8_t>>
SecurityVlmNode::assembleClip(std::uint64_t /*triggerSeq*/) const
{
    // Even sampling across the current ring. The ring is already
    // [trigger - preroll, trigger + postroll] because we waited postroll_sec
    // before dispatching, so "all of the ring" == "the bracketed window".
    std::vector<std::vector<std::uint8_t>> out;
    if (ring_.empty()) return out;

    const int samples = std::min<int>(config_.frameSamples, static_cast<int>(ring_.size()));
    if (samples == 1) {
        out.push_back(ring_.back().jpeg);
        return out;
    }

    const double step = static_cast<double>(ring_.size() - 1) / (samples - 1);
    out.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const auto idx = static_cast<std::size_t>(i * step);
        out.push_back(ring_[idx].jpeg);
    }
    return out;
}

void SecurityVlmNode::publishIncident(const std::string& eventId,
                                      const nlohmann::json& event,
                                      const nlohmann::json& analysis,
                                      int imageCount)
{
    messageRouter_.publish(std::string(kZmqTopicSecurityVlmAnalysis), nlohmann::json{
        {"v", 1},
        {"type", std::string(kZmqTopicSecurityVlmAnalysis)},
        {"event_id", eventId},
        {"timestamp", nowIso8601()},
        {"source_event", event},
        {"analysis", analysis},
        {"image_count", imageCount},
    });
}

void SecurityVlmNode::appendIncidentJsonl(const nlohmann::json& incident)
{
    if (!incidentsFile_.is_open()) return;
    incidentsFile_ << incident.dump() << '\n';
    incidentsFile_.flush();
}

std::string SecurityVlmNode::makeEventId()
{
    // Compact random 16-hex-char id — enough entropy for correlation without
    // pulling in a UUID library.
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    const std::uint64_t a = rng();
    const std::uint64_t b = rng();
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << a
       << std::setw(16) << std::setfill('0') << b;
    return os.str();
}

std::string SecurityVlmNode::nowIso8601()
{
    using namespace std::chrono;
    const auto tp = system_clock::now();
    const auto sec = time_point_cast<seconds>(tp);
    const auto ms = duration_cast<milliseconds>(tp - sec).count();
    const auto tt = system_clock::to_time_t(sec);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    std::ostringstream os;
    os << buf << '.' << std::setw(3) << std::setfill('0') << ms << 'Z';
    return os.str();
}

std::vector<std::uint8_t>
SecurityVlmNode::encodeJpeg(const std::uint8_t* bgr24, std::uint32_t width,
                            std::uint32_t height, std::uint32_t maxDim, int quality)
{
    if (!bgr24 || width == 0 || height == 0) return {};
    cv::Mat src(static_cast<int>(height), static_cast<int>(width), CV_8UC3,
                const_cast<std::uint8_t*>(bgr24));
    cv::Mat scaled;
    const std::uint32_t maxSide = std::max(width, height);
    if (maxSide > maxDim) {
        const double s = static_cast<double>(maxDim) / maxSide;
        cv::resize(src, scaled, cv::Size(), s, s, cv::INTER_AREA);
    } else {
        scaled = src;
    }

    std::vector<std::uint8_t> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", scaled, jpeg, params)) {
        return {};
    }
    return jpeg;
}
