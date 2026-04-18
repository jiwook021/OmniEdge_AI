#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

namespace omniedge::cv {

/// Long-lived Python vision worker (vision_generate.py) owned by one C++
/// process. Blocking request/response: one JPEG list goes in, one parsed
/// JSON `result` comes out.
///
/// Lives in modules/core/cv/ deliberately — nodes/cv/ uses it without any
/// dependency on modules/core/conversation/.
class VisionWorkerClient {
public:
    struct Options {
        std::string scriptPathOverride;  ///< Empty → auto-resolve vision_generate.py
        std::string modelDir;            ///< Passed as argv[1] to the script
        std::chrono::seconds startupTimeout{180};
        std::chrono::milliseconds requestTimeout{60000};
    };

    struct AnalysisResult {
        nlohmann::json result;   ///< Parsed `result` object from the worker
        std::string   rawText;   ///< Populated when the model returned prose the worker couldn't parse
    };

    VisionWorkerClient();
    ~VisionWorkerClient();

    VisionWorkerClient(const VisionWorkerClient&) = delete;
    VisionWorkerClient& operator=(const VisionWorkerClient&) = delete;

    /// Spawn the Python worker and wait for its `{"status":"ready"}` handshake.
    [[nodiscard]] tl::expected<void, std::string> start(const Options& opts);

    /// `images` — JPEG-encoded byte blobs, one per sampled frame.
    /// Returns the worker's `result` object (already parsed).
    [[nodiscard]] tl::expected<AnalysisResult, std::string>
    analyze(std::span<const std::vector<std::uint8_t>> images,
            std::string_view prompt,
            std::string_view eventId);

    /// Send an {"command":"unload"} and terminate the child.
    void stop() noexcept;

    /// Best-effort liveness check — writes {"command":"ping"} and waits for pong.
    [[nodiscard]] bool ping(std::chrono::milliseconds timeout);

    /// True once start() has completed and the worker has reported ready.
    [[nodiscard]] bool isLoaded() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace omniedge::cv
