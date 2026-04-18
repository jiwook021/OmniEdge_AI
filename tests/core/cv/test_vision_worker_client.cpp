#include <gtest/gtest.h>

#include "cv/vision_worker_client.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// VisionWorkerClient Integration Test
//
// Purpose: Verify the long-lived Python subprocess + JSON-lines IPC that
// backs SecurityVlmNode.  Uses the deterministic mock sidecar
// (tests/mocks/mock_vision_generate.py) so the test exercises the real
// pipe/stdin/stdout protocol without pulling in a multi-gigabyte VLM.
//
// What this catches:
//   - fork/execlp path resolution regressions
//   - JSON request serialisation (base64 encoding, field names, event_id)
//   - Response parsing (AnalysisResult::result population, raw_text fallback)
//   - Ping round-trip
//   - Request counter behaviour across multiple analyze() calls
//   - Clean shutdown via stop()
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;
using omniedge::cv::VisionWorkerClient;

namespace {

fs::path findMockScript()
{
    // Tests run from build/; repo root is one level up.  If the caller set
    // OE_MOCK_VISION_GENERATE, prefer that.
    if (const char* override = std::getenv("OE_MOCK_VISION_GENERATE")) {
        return fs::path(override);
    }

    // Walk up from CWD to find tests/mocks/mock_vision_generate.py.
    fs::path cur = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        const auto candidate = cur / "tests" / "mocks" / "mock_vision_generate.py";
        if (fs::exists(candidate)) {
            return candidate;
        }
        if (!cur.has_parent_path()) break;
        cur = cur.parent_path();
    }

    // Fallback: canonical mount in the dev container.
    return fs::path("/src/tests/mocks/mock_vision_generate.py");
}

}  // namespace

class VisionWorkerClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        scriptPath_ = findMockScript();
        ASSERT_TRUE(fs::exists(scriptPath_))
            << "mock_vision_generate.py not found at " << scriptPath_;
    }

    fs::path scriptPath_;
};

TEST_F(VisionWorkerClientTest, StartStopRoundTrip)
{
    VisionWorkerClient client;

    VisionWorkerClient::Options opts;
    opts.scriptPathOverride = scriptPath_.string();
    opts.modelDir           = "unused/by/mock";  // mock ignores this
    opts.startupTimeout     = std::chrono::seconds(10);
    opts.requestTimeout     = std::chrono::milliseconds(5000);

    auto started = client.start(opts);
    ASSERT_TRUE(started) << "start() failed: "
                         << (started ? "" : started.error());
    EXPECT_TRUE(client.isLoaded());

    EXPECT_TRUE(client.ping(std::chrono::milliseconds(2000)));

    client.stop();
    EXPECT_FALSE(client.isLoaded());
}

TEST_F(VisionWorkerClientTest, AnalyzeReturnsParsedResult)
{
    VisionWorkerClient client;

    VisionWorkerClient::Options opts;
    opts.scriptPathOverride = scriptPath_.string();
    opts.modelDir           = "unused";
    opts.requestTimeout     = std::chrono::milliseconds(5000);

    ASSERT_TRUE(client.start(opts));

    // Two tiny fake JPEG-like byte blobs. The mock doesn't decode — it only
    // counts them — so arbitrary bytes are fine.
    std::vector<std::vector<std::uint8_t>> images = {
        {0xFF, 0xD8, 0xFF, 0xE0, 'A', 'B', 'C', 0xFF, 0xD9},
        {0xFF, 0xD8, 0xFF, 0xE0, 'X', 'Y', 'Z', 0xFF, 0xD9},
    };

    auto r = client.analyze(std::span<const std::vector<std::uint8_t>>(
                                images.data(), images.size()),
                            "describe the scene", "evt-42");
    ASSERT_TRUE(r) << "analyze() failed: " << r.error();

    EXPECT_TRUE(r->rawText.empty()) << "mock reply should parse as structured JSON";
    ASSERT_TRUE(r->result.contains("behavior"));
    EXPECT_EQ(r->result["suspicion_level"].get<int>(), 2);
    EXPECT_EQ(r->result["image_count"].get<int>(), 2);
    EXPECT_EQ(r->result["mock_call_number"].get<int>(), 1);

    // Second call — mock increments the counter, proving the same process
    // handles multiple requests on the same pipe.
    auto r2 = client.analyze(std::span<const std::vector<std::uint8_t>>(
                                 images.data(), images.size()),
                             "describe again", "evt-43");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->result["mock_call_number"].get<int>(), 2);

    client.stop();
}

TEST_F(VisionWorkerClientTest, AnalyzeRejectsEmptyInputs)
{
    VisionWorkerClient client;

    VisionWorkerClient::Options opts;
    opts.scriptPathOverride = scriptPath_.string();
    opts.modelDir           = "unused";
    opts.requestTimeout     = std::chrono::milliseconds(3000);

    ASSERT_TRUE(client.start(opts));

    std::vector<std::vector<std::uint8_t>> images = {
        {0xFF, 0xD8, 'X', 0xFF, 0xD9},
    };

    // Missing prompt — mock returns an {"error": "..."} response.
    auto empty = client.analyze(std::span<const std::vector<std::uint8_t>>(
                                     images.data(), images.size()),
                                 "", "evt-empty");
    EXPECT_FALSE(empty) << "empty prompt should produce an error";

    client.stop();
}
