#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <tl/expected.hpp>

#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "common/runtime_defaults.hpp"
#include "common/constants/ingest_constants.hpp"
#include "zmq/message_router.hpp"
#include "zmq/port_settings.hpp"
#include "zmq/zmq_constants.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — ScreenIngestNode
//
// Receives JPEG screen frames from the Windows DXGI capture agent over TCP,
// decodes them to BGR24 using turbo-jpeg, and writes into a POSIX SHM
// circular buffer for ConversationNode (and optionally WS Bridge) to consume.
//
// Always-on module — launched at boot alongside video_ingest and audio_ingest.
// The Camera/Screen toggle in the UI only tells ConversationNode which SHM
// to read; it does NOT spawn or kill this process.
//
// IPC contracts:
//   SHM producer : /oe.screen.ingest (dynamic resolution, BGR24 per slot, 4 slots)
//   ZMQ PUB      : ipc:///tmp/omniedge_5577
//   Topics       : "screen_frame"  — frame notification per decoded frame
//                  "screen_health" — health status on state change
//
// Thread model:
//   TCP recv thread (TcpFrameReceiver) stages raw JPEG bytes under a mutex.
//   ZMQ main thread (poll loop) drains staged JPEG, decodes, writes SHM,
//   and publishes ZMQ notifications.  All SHM access is single-threaded.
// ---------------------------------------------------------------------------


class TcpFrameReceiver;

class ScreenIngestNode : public ModuleNodeBase<ScreenIngestNode> {
public:
    friend class ModuleNodeBase<ScreenIngestNode>;

    struct Config {
        std::string moduleName      = "screen_ingest";
        int         pubPort         = kScreenIngest;       // 5577
        int         wsBridgeSubPort = kWsBridge;           // 5570
        std::string windowsHostIp   = "172.17.0.1";       // WSL2 gateway default
        int         tcpPort         = kScreenIngestTcpPort;
        std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

        /// Seconds without a valid frame before publishing screen_health error.
        int healthTimeoutSec = 5;

        [[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
    };

    explicit ScreenIngestNode(const Config& config);
    ~ScreenIngestNode();

    ScreenIngestNode(const ScreenIngestNode&)            = delete;
    ScreenIngestNode& operator=(const ScreenIngestNode&) = delete;

    // ---- CRTP hooks (required) ------------------------------------------------

    [[nodiscard]] tl::expected<void, std::string> configureTransport();
    [[nodiscard]] tl::expected<void, std::string> loadInferencer();

    // ---- CRTP hooks (optional) ------------------------------------------------

    void onBeforeRun();
    void onAfterStop();

    // ---- Accessors ------------------------------------------------------------

    [[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
    [[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
    Config        config_;
    MessageRouter messageRouter_;

    // ── SHM (dynamically sized from first frame) ──────────────────────────
    std::unique_ptr<ShmCircularBuffer<ShmVideoHeader>> shm_;
    uint32_t currentShmWidth_{0};
    uint32_t currentShmHeight_{0};

    void ensureShmCapacity(uint32_t width, uint32_t height);

    // ── TCP frame receiver ───────────��────────────────────────────��───────
    std::unique_ptr<TcpFrameReceiver> tcpReceiver_;

    // ── Thread-safe JPEG handoff (TCP thread → main thread) ──────────────
    //
    // The TCP recv thread stages raw JPEG bytes under jpegMutex_.
    // The main ZMQ poll thread drains the staging buffer, decodes, and
    // writes SHM.  This avoids the race where ensureShmCapacity() on the
    // TCP thread could destroy SHM while the main thread reads it.
    std::mutex           jpegMutex_;
    std::vector<uint8_t> pendingJpeg_;
    uint32_t             pendingWidth_{0};
    uint32_t             pendingHeight_{0};
    std::atomic<bool>    pendingFrame_{false};

    // ── Decode buffer + frame sequence (main-thread only) ─────────────────
    std::vector<uint8_t> decodeBuf_;
    uint64_t             frameSeq_{0};

    // ── Health watchdog ────��──────────────────────────────────────────────
    std::atomic<uint64_t> lastFrameReceivedTs_{0};  // CLOCK_MONOTONIC ns
    bool                  agentConnected_{false};

    /// Called on the TCP recv thread — stages JPEG under mutex.
    void onTcpFrame(std::span<const uint8_t> jpegData, uint32_t width, uint32_t height);

    /// Called on the main thread — decodes JPEG, writes SHM, publishes ZMQ.
    void processPendingFrame();

    /// Runs every poll cycle — publishes screen_health on state change.
    void checkHealthAndPublish();
};
