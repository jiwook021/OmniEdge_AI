// gemma_inferencer.cpp — HF-transformers-backed Gemma-4 inferencer.
//
// Spawns a persistent Python subprocess (model_generate.py) that loads a
// `google/gemma-4-*-it` checkpoint via AutoModelForCausalLM and streams tokens
// back over stdin/stdout using a JSON-lines protocol.
//
// Flow:
//   1. loadModel(): spawns model_generate.py with the model directory
//   2. generate(): writes a JSON prompt to stdin, reads streaming tokens from
//      stdout and forwards them to the caller via the token callback
//   3. cancel(): writes {"command":"cancel"} to the worker's stdin
//   4. unloadModel(): writes {"command":"unload"}, waits for the child to exit

#include "conversation/gemma_inferencer.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/subprocess_manager.hpp"


namespace {

/// Maximum time to wait for the Python worker to emit `{"status":"ready"}`.
constexpr int kWorkerStartupTimeoutSeconds = 120;

/// Per-readLine cap during the startup wait loop (ms).  Keeps the poller
/// responsive even when the worker is slow to respond.
constexpr int kWorkerStartupReadLineChunkMs = 5000;

/// Poll interval during generation — short so we can pump heartbeats
/// between polls. The outer while-loop retries until kTokenOverallTimeoutMs.
constexpr int kTokenPollTimeoutMs    = 1000;

/// Overall timeout before giving up on a single generation turn.
constexpr int kTokenOverallTimeoutMs = 120000;

/// Measured VRAM estimates (RTX PRO 3000 Blackwell, BF16). Calibrate before
/// tightening the budgets in vram_thresholds.hpp.
constexpr std::size_t kVramGemmaE2b = 2500ULL * 1024 * 1024;
constexpr std::size_t kVramGemmaE4b = 3000ULL * 1024 * 1024;

/// Buffer size for reading lines from the child's stdout.
constexpr std::size_t kReadBufferSize = 4096;

/// SIGTERM → SIGKILL grace period when tearing down the Python worker.
constexpr int kKillChildGracePeriodS = 2;

} // namespace


// ---------------------------------------------------------------------------
// Pimpl
// ---------------------------------------------------------------------------

struct GemmaInferencer::Impl {
    pid_t       childPid{-1};
    int         stdinFd{-1};
    int         stdoutFd{-1};
    bool        loaded{false};
    std::size_t vramBytes{0};
    std::string modelDir;
    std::string scriptPath;
    std::string lineBuf;  ///< partial line buffer for child stdout

    /// Write a JSON object to the child's stdin (one line + newline).
    tl::expected<void, std::string> writeLine(const nlohmann::json& j)
    {
        if (stdinFd < 0) {
            return tl::unexpected(std::string("worker stdin closed"));
        }
        std::string line = j.dump() + "\n";
        const char* ptr = line.data();
        std::size_t remaining = line.size();
        while (remaining > 0) {
            ssize_t written = ::write(stdinFd, ptr, remaining);
            if (written < 0) {
                if (errno == EINTR) continue;
                return tl::unexpected(
                    std::format("worker stdin write failed: {}", std::strerror(errno)));
            }
            ptr += written;
            remaining -= static_cast<std::size_t>(written);
        }
        return {};
    }

    /// Read one JSON line from the child's stdout. Returns empty json on
    /// timeout or error. Valid JSON objects always contain at least one key.
    nlohmann::json readLine(int timeoutMs)
    {
        if (stdoutFd < 0) return {};

        while (true) {
            auto nlPos = lineBuf.find('\n');
            if (nlPos != std::string::npos) {
                std::string line = lineBuf.substr(0, nlPos);
                lineBuf.erase(0, nlPos + 1);
                if (line.empty()) continue;
                try {
                    return nlohmann::json::parse(line);
                } catch (...) {
                    OE_LOG_WARN("gemma_worker_parse_error: {}", line);
                    return {};
                }
            }

            struct pollfd pfd{};
            pfd.fd = stdoutFd;
            pfd.events = POLLIN;
            int ret = ::poll(&pfd, 1, timeoutMs);
            if (ret < 0) {
                if (errno == EINTR) continue;
                OE_LOG_WARN("gemma_worker_poll_error: {}", std::strerror(errno));
                return {};
            }
            if (ret == 0) {
                return {};
            }

            std::array<char, kReadBufferSize> buf{};
            ssize_t bytesRead = ::read(stdoutFd, buf.data(), buf.size());
            if (bytesRead <= 0) {
                if (bytesRead < 0 && errno == EINTR) continue;
                OE_LOG_WARN("gemma_worker_read_eof");
                return {};
            }
            lineBuf.append(buf.data(), static_cast<std::size_t>(bytesRead));
        }
    }

    /// Kill the child process if running.
    void killChild() noexcept
    {
        if (childPid > 0) {
            SubprocessManager::terminateProcess(
                childPid,
                std::chrono::seconds{kKillChildGracePeriodS},
                /*killProcessGroup=*/false);
            childPid = -1;
        }
        if (stdinFd >= 0)  { ::close(stdinFd);  stdinFd  = -1; }
        if (stdoutFd >= 0) { ::close(stdoutFd); stdoutFd = -1; }
        lineBuf.clear();
    }
};


// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GemmaInferencer::GemmaInferencer(Variant variant)
    : variant_(variant)
    , pimpl_(std::make_unique<Impl>())
{
}

GemmaInferencer::~GemmaInferencer()
{
    unloadModel();
}

std::string_view GemmaInferencer::variantTag(Variant variant) noexcept
{
    switch (variant) {
        case Variant::kE2B: return "gemma-e2b";
        case Variant::kE4B: return "gemma-e4b";
    }
    return "gemma-unknown";
}

std::string GemmaInferencer::name() const
{
    return std::string(variantTag(variant_));
}


// ---------------------------------------------------------------------------
// loadModel()
// ---------------------------------------------------------------------------

tl::expected<void, std::string>
GemmaInferencer::loadModel(const std::string& modelDir)
{
    OE_LOG_INFO("conversation_load_start: model_dir={}, variant={}",
                modelDir, variantTag(variant_));

    if (pimpl_->loaded) {
        unloadModel();
    }

    if (!std::filesystem::is_directory(modelDir)) {
        return tl::unexpected(
            std::format("Model directory not found: {}", modelDir));
    }

    if (!std::filesystem::exists(std::filesystem::path(modelDir) / "config.json")) {
        return tl::unexpected(
            std::format("No config.json in model directory: {}", modelDir));
    }

    pimpl_->modelDir = modelDir;

    const std::vector<std::string> scriptCandidates = {
        std::string(std::filesystem::path(__FILE__).parent_path() / "model_generate.py"),
        (std::filesystem::current_path() / "modules" / "core" / "conversation" / "src" / "model_generate.py").string(),
        "/opt/omniedge/share/model_generate.py",
        (std::filesystem::current_path() / "share" / "model_generate.py").string(),
    };

    for (const auto& candidate : scriptCandidates) {
        if (std::filesystem::exists(candidate)) {
            pimpl_->scriptPath = candidate;
            break;
        }
    }

    if (pimpl_->scriptPath.empty()) {
        return tl::unexpected(std::string("model_generate.py not found"));
    }

    OE_LOG_INFO("conversation_spawning_worker: script={}, model_dir={}",
                pimpl_->scriptPath, pimpl_->modelDir);

    int stdinPipe[2]  = {-1, -1};  // [0]=read (child), [1]=write (parent)
    int stdoutPipe[2] = {-1, -1};  // [0]=read (parent), [1]=write (child)

    if (::pipe(stdinPipe) != 0 || ::pipe(stdoutPipe) != 0) {
        return tl::unexpected(std::format(
            "pipe() failed for {} worker (script={}): {}",
            variantTag(variant_), pimpl_->scriptPath, std::strerror(errno)));
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(stdinPipe[0]);  ::close(stdinPipe[1]);
        ::close(stdoutPipe[0]); ::close(stdoutPipe[1]);
        return tl::unexpected(std::format(
            "fork() failed for {} worker (script={}): {}",
            variantTag(variant_), pimpl_->scriptPath, std::strerror(errno)));
    }

    if (pid == 0) {
        // ── Child process ──
        ::prctl(PR_SET_PDEATHSIG, SIGTERM);

        ::close(stdinPipe[1]);
        ::close(stdoutPipe[0]);

        ::dup2(stdinPipe[0], STDIN_FILENO);
        ::dup2(stdoutPipe[1], STDOUT_FILENO);

        ::close(stdinPipe[0]);
        ::close(stdoutPipe[1]);

        ::execlp("python3", "python3",
                 pimpl_->scriptPath.c_str(),
                 pimpl_->modelDir.c_str(),
                 nullptr);
        ::_exit(127);
    }

    // ── Parent process ──
    ::close(stdinPipe[0]);
    ::close(stdoutPipe[1]);

    pimpl_->childPid = pid;
    pimpl_->stdinFd  = stdinPipe[1];
    pimpl_->stdoutFd = stdoutPipe[0];

    int flags = ::fcntl(pimpl_->stdoutFd, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(pimpl_->stdoutFd, F_SETFL, flags | O_NONBLOCK);
    }

    OE_LOG_INFO("conversation_waiting_for_worker: timeout={}s", kWorkerStartupTimeoutSeconds);
    auto startTime = std::chrono::steady_clock::now();
    bool ready = false;

    while (!ready) {
        auto elapsed = std::chrono::steady_clock::now() - startTime;
        int remainingMs = kWorkerStartupTimeoutSeconds * 1000
            - static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        if (remainingMs <= 0) break;

        auto msg = pimpl_->readLine(std::min(remainingMs, kWorkerStartupReadLineChunkMs));
        if (msg.is_null()) {
            auto exitInfo = SubprocessManager::checkProcess(pimpl_->childPid);
            if (exitInfo.exited) {
                pimpl_->killChild();
                return tl::unexpected(
                    std::format("Worker exited during startup (status={})", exitInfo.exitCode));
            }
            continue;
        }

        if (msg.contains("status") && msg["status"] == "ready") {
            ready = true;
        } else if (msg.contains("error")) {
            pimpl_->killChild();
            return tl::unexpected(
                std::format("Worker error during startup: {}", msg["error"].get<std::string>()));
        }
    }

    if (!ready) {
        pimpl_->killChild();
        return tl::unexpected(std::string("Worker startup timed out"));
    }

    auto loadTime = std::chrono::steady_clock::now() - startTime;
    double loadMs = std::chrono::duration<double, std::milli>(loadTime).count();

    pimpl_->vramBytes = (variant_ == Variant::kE2B) ? kVramGemmaE2b : kVramGemmaE4b;
    pimpl_->loaded = true;

    OE_LOG_INFO("conversation_load_done: variant={}, load_time_ms={:.0f}, estimated_vram_mb={:.0f}",
                variantTag(variant_), loadMs,
                static_cast<double>(pimpl_->vramBytes) / (1024.0 * 1024.0));

    return {};
}


// ---------------------------------------------------------------------------
// generate()
// ---------------------------------------------------------------------------

tl::expected<void, std::string>
GemmaInferencer::generate(
    const std::string& prompt,
    const GenerationParams& params,
    const std::function<void(std::string_view, bool, bool)>& callback)
{
    OE_ZONE_SCOPED;

    if (!pimpl_->loaded) {
        return tl::unexpected(std::string("Model not loaded — call loadModel() first"));
    }

    cancelRequested_.store(false, std::memory_order_release);

    nlohmann::json request = {
        {"prompt",      prompt},
        {"max_tokens",  params.maxTokens},
        {"temperature", params.temperature},
        {"top_p",       params.topP},
    };

    if (auto w = pimpl_->writeLine(request); !w) {
        return tl::unexpected(std::format(
            "Failed to send prompt to {} worker: {}",
            variantTag(variant_), w.error()));
    }

    bool prevEndedSentence = false;
    auto overallStart = std::chrono::steady_clock::now();

    while (true) {
        if (heartbeatCb_) {
            heartbeatCb_();
        }

        if (cancelRequested_.load(std::memory_order_acquire)) {
            (void)pimpl_->writeLine({{"command", "cancel"}});
            for (int i = 0; i < 100; ++i) {
                auto msg = pimpl_->readLine(1000);
                if (msg.is_null() || (msg.contains("done") && msg["done"].get<bool>())) break;
            }
            callback("", false, true);
            return {};
        }

        auto msg = pimpl_->readLine(kTokenPollTimeoutMs);

        if (msg.is_null()) {
            auto exitInfo = SubprocessManager::checkProcess(pimpl_->childPid);
            if (exitInfo.exited) {
                pimpl_->loaded = false;
                pimpl_->childPid = -1;
                return tl::unexpected(std::string("Worker process died during generation"));
            }
            auto elapsed = std::chrono::steady_clock::now() - overallStart;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                    > kTokenOverallTimeoutMs) {
                callback("", false, true);
                return tl::unexpected(std::string("Generation timed out"));
            }
            continue;
        }

        if (msg.contains("error")) {
            std::string errMsg = msg.value("error", "unknown error");
            callback("", false, true);
            return tl::unexpected(std::format("Worker generation error: {}", errMsg));
        }

        if (msg.contains("token")) {
            std::string token = msg.value("token", "");
            bool done = msg.value("done", false);

            if (!token.empty() || done) {
                callback(token, prevEndedSentence, done);

                if (!token.empty()) {
                    char last = token.back();
                    prevEndedSentence = (last == '.' || last == '?' || last == '!');
                }
            }

            if (done) {
                return {};
            }
        }
    }
}


// ---------------------------------------------------------------------------
// generateWithVideo() — delegates to text-only for now (model_generate.py
// does not yet accept image tensors over the JSON-lines protocol).
// ---------------------------------------------------------------------------

tl::expected<void, std::string>
GemmaInferencer::generateWithVideo(
    const std::string& prompt,
    std::span<const uint8_t> /*videoFrame*/,
    uint32_t /*frameWidth*/,
    uint32_t /*frameHeight*/,
    const GenerationParams& params,
    const std::function<void(std::string_view, bool, bool)>& callback)
{
    return generate(prompt, params, callback);
}


// ---------------------------------------------------------------------------
// transcribe() — native STT is exposed via the Gemma chat turn itself, so
// the C++ layer has no need to call a standalone transcribe endpoint.
// ---------------------------------------------------------------------------

tl::expected<std::string, std::string>
GemmaInferencer::transcribe(
    std::span<const float> /*pcmAudio*/,
    uint32_t /*sampleRateHz*/)
{
    return tl::unexpected(std::string(
        "Gemma-4 native STT is consumed inline by generate() — "
        "call generate() with the audio embedded in the chat turn"));
}


// ---------------------------------------------------------------------------
// cancel()
// ---------------------------------------------------------------------------

void GemmaInferencer::cancel() noexcept
{
    cancelRequested_.store(true, std::memory_order_release);
}


// ---------------------------------------------------------------------------
// unloadModel()
// ---------------------------------------------------------------------------

void GemmaInferencer::unloadModel() noexcept
{
    if (pimpl_ && pimpl_->loaded) {
        (void)pimpl_->writeLine({{"command", "unload"}});
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (pimpl_) {
        pimpl_->killChild();
        pimpl_->loaded = false;
        pimpl_->vramBytes = 0;
    }
}


// ---------------------------------------------------------------------------
// currentVramUsageBytes()
// ---------------------------------------------------------------------------

size_t GemmaInferencer::currentVramUsageBytes() const noexcept
{
    return pimpl_ ? pimpl_->vramBytes : 0;
}


// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<ConversationInferencer>
createGemmaInferencer(GemmaInferencer::Variant variant)
{
    return std::make_unique<GemmaInferencer>(variant);
}
