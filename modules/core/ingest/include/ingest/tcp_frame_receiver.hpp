#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "common/constants/ingest_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — TCP Frame Receiver
//
// Connects to a TCP server (Windows DXGI screen capture agent) and reads
// framed JPEG data with an 8-byte header per frame:
//
//   [uint32_t jpegLen][uint16_t width][uint16_t height][JPEG payload]
//
// All header fields are native little-endian (both sides are x86).
//
// Thread safety:
//   start() spawns a recv thread that invokes onFrame for each
//   decoded frame.  The callback fires on the recv thread — callers must
//   synchronise access to shared state from inside the callback.
//   stop() is safe to call from any thread.
// ---------------------------------------------------------------------------


/**
 * @brief TCP client that receives framed JPEG data from a remote server.
 *
 * Designed for the screen capture pipeline: connects to the Windows DXGI
 * agent, reads 8-byte headers + JPEG payloads, and invokes a callback
 * per frame.  Reconnects automatically on connection loss.
 *
 * Non-copyable, non-movable — owns a socket and thread.
 */
class TcpFrameReceiver {
public:
    struct Config {
        std::string host;
        int         port{kScreenIngestTcpPort};
        std::chrono::seconds reconnectInterval{2};
    };

    explicit TcpFrameReceiver(const Config& config);
    ~TcpFrameReceiver();

    TcpFrameReceiver(const TcpFrameReceiver&)            = delete;
    TcpFrameReceiver& operator=(const TcpFrameReceiver&) = delete;

    /// Start the recv thread.  @p onFrame fires on the recv thread.
    void start(std::function<void(std::span<const uint8_t> jpegData,
                                  uint32_t width, uint32_t height)> onFrame);

    /// Stop the recv thread and close the socket.  Safe from any thread.
    void stop();

    /// True if the TCP socket is currently connected to the remote server.
    [[nodiscard]] bool isConnected() const noexcept { return connected_.load(std::memory_order_acquire); }

private:
    Config config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::thread       recvThread_;
    int               sockFd_{-1};

    /// Receive buffer — reused across frames to avoid reallocation.
    std::vector<uint8_t> recvBuf_;

    void recvLoop(std::function<void(std::span<const uint8_t> jpegData,
                                     uint32_t width, uint32_t height)> onFrame);

    /// Attempt to connect with retry.  Blocks until connected or stopped.
    /// @return true if connected, false if stop() was called.
    [[nodiscard]] bool connectWithRetry();

    /// Read exactly @p len bytes into @p buf.
    /// @return true on success, false on disconnect or stop.
    [[nodiscard]] bool readExact(void* buf, std::size_t len);

    void closeSocket() noexcept;
};
