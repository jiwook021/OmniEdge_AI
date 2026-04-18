// tcp_frame_receiver.cpp -- TCP client for receiving framed JPEG from Windows agent

#include "ingest/tcp_frame_receiver.hpp"

#include <cerrno>
#include <cstring>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <poll.h>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Wire protocol header — native little-endian (both sides x86, no byte-swap).
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t jpegLen;
    uint16_t width;
    uint16_t height;
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 8, "FrameHeader must be exactly 8 bytes");

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

TcpFrameReceiver::TcpFrameReceiver(const Config& config)
    : config_(config)
{
}

TcpFrameReceiver::~TcpFrameReceiver()
{
    stop();
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void TcpFrameReceiver::start(std::function<void(std::span<const uint8_t> jpegData,
                                                uint32_t width, uint32_t height)> onFrame)
{
    if (running_.load()) return;
    running_.store(true, std::memory_order_release);
    recvThread_ = std::thread([this, cb = std::move(onFrame)]() mutable {
        recvLoop(std::move(cb));
    });
}

void TcpFrameReceiver::stop()
{
    running_.store(false, std::memory_order_release);
    closeSocket();
    if (recvThread_.joinable()) {
        recvThread_.join();
    }
}

// ---------------------------------------------------------------------------
// recvLoop
// ---------------------------------------------------------------------------

void TcpFrameReceiver::recvLoop(std::function<void(std::span<const uint8_t> jpegData,
                                                   uint32_t width, uint32_t height)> onFrame)
{
    OE_ZONE_SCOPED;
    OE_LOG_INFO("tcp_recv_loop_started: host={}, port={}", config_.host, config_.port);

    while (running_.load(std::memory_order_acquire)) {
        if (!connectWithRetry()) {
            break;  // stop() was called
        }

        OE_LOG_INFO("tcp_connected: {}:{}", config_.host, config_.port);

        // Read frames until disconnect or stop.
        while (running_.load(std::memory_order_acquire)) {
            FrameHeader hdr{};
            if (!readExact(&hdr, sizeof(hdr))) {
                OE_LOG_WARN("tcp_header_read_failed: disconnected");
                break;
            }

            // Sanity-check header values.
            if (hdr.jpegLen == 0 || hdr.jpegLen > 16 * 1024 * 1024) {
                OE_LOG_WARN("tcp_invalid_header: jpegLen={}, dropping connection", hdr.jpegLen);
                break;
            }
            if (hdr.width == 0 || hdr.height == 0 || hdr.width > 7680 || hdr.height > 4320) {
                OE_LOG_WARN("tcp_invalid_header: {}x{}, dropping connection", hdr.width, hdr.height);
                break;
            }

            // Read JPEG payload.
            recvBuf_.resize(hdr.jpegLen);
            if (!readExact(recvBuf_.data(), hdr.jpegLen)) {
                OE_LOG_WARN("tcp_payload_read_failed: expected {} bytes", hdr.jpegLen);
                break;
            }

            OE_LOG_DEBUG("tcp_frame_received: {}x{}, jpeg_bytes={}", hdr.width, hdr.height, hdr.jpegLen);
            onFrame(std::span<const uint8_t>{recvBuf_}, hdr.width, hdr.height);
        }

        // Disconnected — close socket and retry.
        closeSocket();
    }

    OE_LOG_INFO("tcp_recv_loop_stopped");
}

// ---------------------------------------------------------------------------
// connectWithRetry
// ---------------------------------------------------------------------------

bool TcpFrameReceiver::connectWithRetry()
{
    while (running_.load(std::memory_order_acquire)) {
        sockFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sockFd_ < 0) {
            OE_LOG_WARN("tcp_socket_create_failed: {}", std::strerror(errno));
            std::this_thread::sleep_for(config_.reconnectInterval);
            continue;
        }

        // Disable Nagle — we want whole frames without delay.
        int flag = 1;
        ::setsockopt(sockFd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(config_.port));
        if (::inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) <= 0) {
            OE_LOG_WARN("tcp_invalid_host: {}", config_.host);
            closeSocket();
            std::this_thread::sleep_for(config_.reconnectInterval);
            continue;
        }

        if (::connect(sockFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            connected_.store(true, std::memory_order_release);
            return true;
        }

        OE_LOG_DEBUG("tcp_connect_retry: {}:{} ({})", config_.host, config_.port, std::strerror(errno));
        closeSocket();
        std::this_thread::sleep_for(config_.reconnectInterval);
    }

    return false;  // stop() was called
}

// ---------------------------------------------------------------------------
// readExact
// ---------------------------------------------------------------------------

bool TcpFrameReceiver::readExact(void* buf, std::size_t len)
{
    auto* ptr       = static_cast<uint8_t*>(buf);
    std::size_t remaining = len;

    while (remaining > 0 && running_.load(std::memory_order_acquire)) {
        // Use poll() so we can check running_ periodically.
        pollfd pfd{};
        pfd.fd     = sockFd_;
        pfd.events = POLLIN;
        int pollResult = ::poll(&pfd, 1, 500);  // 500ms timeout

        if (pollResult < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (pollResult == 0) continue;  // timeout — re-check running_

        auto n = ::recv(sockFd_, ptr, remaining, 0);
        if (n <= 0) {
            connected_.store(false, std::memory_order_release);
            return false;
        }
        ptr       += n;
        remaining -= static_cast<std::size_t>(n);
    }

    return remaining == 0;
}

// ---------------------------------------------------------------------------
// closeSocket
// ---------------------------------------------------------------------------

void TcpFrameReceiver::closeSocket() noexcept
{
    connected_.store(false, std::memory_order_release);
    if (sockFd_ >= 0) {
        ::close(sockFd_);
        sockFd_ = -1;
    }
}
