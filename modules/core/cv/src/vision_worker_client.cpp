#include "cv/vision_worker_client.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/subprocess_manager.hpp"

namespace omniedge::cv {

namespace {

constexpr int  kStartupReadChunkMs   = 5000;
constexpr int  kResponsePollMs       = 500;
constexpr int  kPingTimeoutChunkMs   = 250;
constexpr int  kKillGracePeriodS     = 2;
constexpr std::size_t kReadBufSize   = 8192;

/// Minimal standard Base64 encoder — used to marshal JPEG blobs into the
/// JSON-lines request body. Kept local to avoid pulling in a new utility
/// library for a single consumer.
std::string base64Encode(const std::uint8_t* data, std::size_t len)
{
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= len) {
        const std::uint32_t v =
            (static_cast<std::uint32_t>(data[i])     << 16) |
            (static_cast<std::uint32_t>(data[i + 1]) << 8)  |
             static_cast<std::uint32_t>(data[i + 2]);
        out.push_back(kAlphabet[(v >> 18) & 0x3F]);
        out.push_back(kAlphabet[(v >> 12) & 0x3F]);
        out.push_back(kAlphabet[(v >> 6)  & 0x3F]);
        out.push_back(kAlphabet[ v        & 0x3F]);
        i += 3;
    }
    if (i < len) {
        std::uint32_t v = static_cast<std::uint32_t>(data[i]) << 16;
        if (i + 1 < len) v |= static_cast<std::uint32_t>(data[i + 1]) << 8;
        out.push_back(kAlphabet[(v >> 18) & 0x3F]);
        out.push_back(kAlphabet[(v >> 12) & 0x3F]);
        out.push_back(i + 1 < len ? kAlphabet[(v >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

/// Auto-resolve where `vision_generate.py` lives — the source tree, the
/// install dir, or an installed share dir. Ordered by how we actually run.
std::string resolveDefaultScriptPath()
{
    const std::array<std::filesystem::path, 4> candidates = {
        std::filesystem::path(__FILE__).parent_path() / "vision_generate.py",
        std::filesystem::current_path() / "modules" / "core" / "cv" / "src" / "vision_generate.py",
        std::filesystem::path("/opt/omniedge/share/vision_generate.py"),
        std::filesystem::current_path() / "share" / "vision_generate.py",
    };
    for (const auto& p : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) return p.string();
    }
    return {};
}

}  // namespace


struct VisionWorkerClient::Impl {
    pid_t pid{-1};
    int   stdinFd{-1};
    int   stdoutFd{-1};
    bool  ready{false};
    std::string scriptPath;
    std::string modelDir;
    std::string lineBuf;

    bool writeLine(const nlohmann::json& j)
    {
        if (stdinFd < 0) return false;
        const std::string line = j.dump() + "\n";
        const char* ptr = line.data();
        std::size_t rem = line.size();
        while (rem > 0) {
            ssize_t w = ::write(stdinFd, ptr, rem);
            if (w < 0) {
                if (errno == EINTR) continue;
                OE_LOG_WARN("vision_worker_write_error: {}", std::strerror(errno));
                return false;
            }
            ptr += w;
            rem -= static_cast<std::size_t>(w);
        }
        return true;
    }

    nlohmann::json readLine(int timeoutMs)
    {
        if (stdoutFd < 0) return {};
        while (true) {
            const auto nl = lineBuf.find('\n');
            if (nl != std::string::npos) {
                std::string line = lineBuf.substr(0, nl);
                lineBuf.erase(0, nl + 1);
                if (line.empty()) continue;
                try {
                    return nlohmann::json::parse(line);
                } catch (...) {
                    OE_LOG_WARN("vision_worker_parse_error: {}", line);
                    return {};
                }
            }

            struct pollfd pfd{};
            pfd.fd = stdoutFd;
            pfd.events = POLLIN;
            const int r = ::poll(&pfd, 1, timeoutMs);
            if (r < 0) {
                if (errno == EINTR) continue;
                OE_LOG_WARN("vision_worker_poll_error: {}", std::strerror(errno));
                return {};
            }
            if (r == 0) return {};

            std::array<char, kReadBufSize> buf{};
            const ssize_t got = ::read(stdoutFd, buf.data(), buf.size());
            if (got <= 0) {
                if (got < 0 && errno == EINTR) continue;
                return {};
            }
            lineBuf.append(buf.data(), static_cast<std::size_t>(got));
        }
    }

    void killChild() noexcept
    {
        if (pid > 0) {
            SubprocessManager::terminateProcess(
                pid,
                std::chrono::seconds{kKillGracePeriodS},
                /*killProcessGroup=*/false);
            pid = -1;
        }
        if (stdinFd  >= 0) { ::close(stdinFd);  stdinFd  = -1; }
        if (stdoutFd >= 0) { ::close(stdoutFd); stdoutFd = -1; }
        lineBuf.clear();
        ready = false;
    }
};


VisionWorkerClient::VisionWorkerClient()
    : pimpl_(std::make_unique<Impl>())
{}

VisionWorkerClient::~VisionWorkerClient()
{
    stop();
}

bool VisionWorkerClient::isLoaded() const noexcept
{
    return pimpl_ && pimpl_->ready;
}


tl::expected<void, std::string>
VisionWorkerClient::start(const Options& opts)
{
    if (pimpl_->ready) {
        return {};  // idempotent
    }

    pimpl_->scriptPath = opts.scriptPathOverride.empty()
                             ? resolveDefaultScriptPath()
                             : opts.scriptPathOverride;
    if (pimpl_->scriptPath.empty()) {
        return tl::unexpected(std::string("vision_generate.py not found"));
    }

    pimpl_->modelDir = opts.modelDir;
    if (pimpl_->modelDir.empty()) {
        return tl::unexpected(std::string("VisionWorkerClient: modelDir is empty"));
    }

    OE_LOG_INFO("vision_worker_spawn: script={}, model_dir={}",
                pimpl_->scriptPath, pimpl_->modelDir);

    int in_pipe[2]  = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) {
        return tl::unexpected(std::format("pipe() failed: {}", std::strerror(errno)));
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]);  ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        return tl::unexpected(std::format("fork() failed: {}", std::strerror(errno)));
    }

    if (pid == 0) {
        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        ::dup2(in_pipe[0],  STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);

        ::execlp("python3", "python3",
                 pimpl_->scriptPath.c_str(),
                 pimpl_->modelDir.c_str(),
                 nullptr);
        ::_exit(127);
    }

    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    pimpl_->pid      = pid;
    pimpl_->stdinFd  = in_pipe[1];
    pimpl_->stdoutFd = out_pipe[0];

    if (const int flags = ::fcntl(pimpl_->stdoutFd, F_GETFL, 0); flags >= 0) {
        ::fcntl(pimpl_->stdoutFd, F_SETFL, flags | O_NONBLOCK);
    }

    const auto startT = std::chrono::steady_clock::now();
    const auto deadline = startT + opts.startupTimeout;

    while (std::chrono::steady_clock::now() < deadline) {
        const int remainingMs =
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count());
        const auto msg = pimpl_->readLine(std::min(remainingMs, kStartupReadChunkMs));
        if (msg.is_null()) {
            const auto info = SubprocessManager::checkProcess(pimpl_->pid);
            if (info.exited) {
                pimpl_->killChild();
                return tl::unexpected(std::format(
                    "Vision worker exited during startup (status={})", info.exitCode));
            }
            continue;
        }
        if (msg.contains("status") && msg["status"] == "ready") {
            pimpl_->ready = true;
            break;
        }
        if (msg.contains("error")) {
            pimpl_->killChild();
            return tl::unexpected(std::format(
                "Vision worker startup error: {}", msg["error"].get<std::string>()));
        }
    }

    if (!pimpl_->ready) {
        pimpl_->killChild();
        return tl::unexpected(std::string("Vision worker startup timed out"));
    }

    const auto loadMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - startT).count();
    OE_LOG_INFO("vision_worker_ready: load_ms={:.0f}", loadMs);
    return {};
}


bool VisionWorkerClient::ping(std::chrono::milliseconds timeout)
{
    if (!pimpl_->ready) return false;
    if (!pimpl_->writeLine({{"command", "ping"}})) return false;

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const auto msg = pimpl_->readLine(kPingTimeoutChunkMs);
        if (msg.contains("status") && msg["status"] == "pong") return true;
    }
    return false;
}


tl::expected<VisionWorkerClient::AnalysisResult, std::string>
VisionWorkerClient::analyze(std::span<const std::vector<std::uint8_t>> images,
                            std::string_view prompt,
                            std::string_view eventId)
{
    OE_ZONE_SCOPED;

    if (!pimpl_->ready) {
        return tl::unexpected(std::string("Vision worker not started"));
    }
    if (images.empty()) {
        return tl::unexpected(std::string("analyze() requires at least one image"));
    }
    if (prompt.empty()) {
        return tl::unexpected(std::string("analyze() requires a non-empty prompt"));
    }

    nlohmann::json req = {
        {"mode",     "analyze"},
        {"prompt",   std::string(prompt)},
        {"event_id", std::string(eventId)},
    };
    auto& arr = req["images_b64"] = nlohmann::json::array();
    for (const auto& jpeg : images) {
        arr.push_back(base64Encode(jpeg.data(), jpeg.size()));
    }

    if (!pimpl_->writeLine(req)) {
        return tl::unexpected(std::string("Failed to write request to vision worker"));
    }

    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(60000);

    while (std::chrono::steady_clock::now() < deadline) {
        const auto msg = pimpl_->readLine(kResponsePollMs);
        if (msg.is_null()) {
            const auto info = SubprocessManager::checkProcess(pimpl_->pid);
            if (info.exited) {
                pimpl_->ready = false;
                pimpl_->pid = -1;
                return tl::unexpected(
                    std::string("Vision worker died during analyze()"));
            }
            continue;
        }

        // Ignore stray {"status":"pong"} that may show up from an earlier ping().
        if (msg.contains("status")) continue;

        if (msg.contains("error")) {
            return tl::unexpected(std::format(
                "Vision worker error: {}", msg["error"].get<std::string>()));
        }

        if (msg.contains("result")) {
            AnalysisResult out;
            out.result = msg["result"];
            if (out.result.is_object() && out.result.contains("raw_text")) {
                out.rawText = out.result["raw_text"].get<std::string>();
            }
            return out;
        }
    }

    return tl::unexpected(std::string("Vision worker analyze() timed out"));
}


void VisionWorkerClient::stop() noexcept
{
    if (!pimpl_) return;
    if (pimpl_->ready) {
        pimpl_->writeLine({{"command", "unload"}});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    pimpl_->killChild();
}

}  // namespace omniedge::cv
