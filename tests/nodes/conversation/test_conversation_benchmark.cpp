// test_conversation_benchmark.cpp -- Token throughput benchmark for ConversationNode
//
// Measures time to process N tokens through the full node pipeline
// (ZMQ PUB/SUB + JSON serialization + token callback).
// Uses MockConversationInferencer -- no GPU required.
//
// This is NOT a micro-benchmark of the model -- it measures the node
// overhead (ZMQ, JSON, callback dispatch) to ensure the pipeline does
// not become a bottleneck when the model is producing tokens fast.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "conversation/conversation_node.hpp"
#include "conversation_inferencer.hpp"
#include "common/constants/conversation_constants.hpp"
#include "common/runtime_defaults.hpp"

#include "tests/mocks/mock_conversation_inferencer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"


class ConversationBenchmarkTest : public ::testing::Test {
protected:
    static constexpr int kTestPubPortBase       = 27572;
    static constexpr int kTestDaemonSubPortBase = 27571;
    static constexpr int kTestUiCmdSubPortBase  = 27570;
    static inline std::atomic<int> portCounter_{0};

    int kTestPubPort{};
    int kTestDaemonSubPort{};
    int kTestUiCmdSubPort{};

    void SetUp() override
    {
        const int offset = portCounter_.fetch_add(3) * 3;
        kTestPubPort       = kTestPubPortBase + offset;
        kTestDaemonSubPort = kTestDaemonSubPortBase + offset;
        kTestUiCmdSubPort  = kTestUiCmdSubPortBase + offset;
    }

    zmq::context_t testCtx_{1};
};

TEST_F(ConversationBenchmarkTest, TokenThroughput_100Tokens)
{
    // Generate 100 tokens to measure pipeline overhead.
    constexpr int tokenCount = 100;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    std::vector<std::string> tokens;
    tokens.reserve(tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        tokens.push_back(std::format("tok{}", i));
    }
    mockInferencer->tokensToEmit = tokens;
    mockInferencer->nativeTts = true;

    ConversationNode::Config cfg;
    cfg.pubPort          = kTestPubPort;
    cfg.daemonSubPort    = kTestDaemonSubPort;
    cfg.uiCommandSubPort = kTestUiCmdSubPort;
    cfg.modelDir         = "/tmp/test_model";
    cfg.modelVariant     = "gemma_e4b";
    cfg.pollTimeout      = std::chrono::milliseconds(20);

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node(cfg, std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Subscribe AFTER node.initialize() + run — avoids ZMQ slow-joiner bug where
    // cleanStaleEndpoint() removes the IPC socket, causing the subscriber to
    // auto-reconnect and lose its SUB filter.
    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "benchmark prompt"},
        {"sequence_id",  1},
        {"generation_params", {
            {"temperature", 0.7},
            {"top_p", 0.9},
            {"max_tokens", 2048},
        }},
    };

    std::vector<nlohmann::json> received;
    const auto startTime = std::chrono::steady_clock::now();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{10000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            for (int attempt = 0; attempt < tokenCount + 10; ++attempt) {
                auto msg = receiveTestMessage(responseSub, 500);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();

    node.stop();
    runThread.join();

    ASSERT_EQ(received.size(), static_cast<size_t>(tokenCount));
    EXPECT_TRUE(received.back().value("finished", false));

    const double tokensPerSecond = (elapsedMs > 0.0)
        ? (tokenCount * 1000.0 / elapsedMs) : 0.0;

    // Log benchmark results (visible in test output).
    std::cerr << "\n[BENCHMARK] " << tokenCount << " tokens in "
              << elapsedMs << " ms (" << tokensPerSecond << " tok/s)\n"
              << "[BENCHMARK] Average per-token pipeline overhead: "
              << (elapsedMs / tokenCount) << " ms\n";

    // Sanity check: pipeline overhead should be well under 10ms per token.
    // Real models are much slower than this -- if the pipeline overhead
    // exceeds 10ms/token, something is wrong with ZMQ or JSON serialization.
    EXPECT_LT(elapsedMs / tokenCount, 10.0)
        << "Pipeline overhead too high -- possible ZMQ or JSON bottleneck";
}

TEST_F(ConversationBenchmarkTest, TokenThroughput_GemmaE4B_50Tokens)
{
    constexpr int tokenCount = 50;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    std::vector<std::string> tokens;
    tokens.reserve(tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        tokens.push_back(std::format("gem{}", i));
    }
    mockInferencer->tokensToEmit = tokens;
    mockInferencer->nativeTts = false;  // Gemma E4B path

    ConversationNode::Config cfg;
    cfg.pubPort          = kTestPubPort;
    cfg.daemonSubPort    = kTestDaemonSubPort;
    cfg.uiCommandSubPort = kTestUiCmdSubPort;
    cfg.modelDir         = "/tmp/test_model";
    cfg.modelVariant     = "gemma_e4b";
    cfg.pollTimeout      = std::chrono::milliseconds(20);

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node(cfg, std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Subscribe AFTER node.initialize() + run — avoids ZMQ slow-joiner bug.
    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "benchmark prompt gemma"},
        {"sequence_id",  2},
    };

    std::vector<nlohmann::json> received;
    const auto startTime = std::chrono::steady_clock::now();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{10000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            for (int attempt = 0; attempt < tokenCount + 10; ++attempt) {
                auto msg = receiveTestMessage(responseSub, 500);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();

    node.stop();
    runThread.join();

    ASSERT_EQ(received.size(), static_cast<size_t>(tokenCount));

    // Verify all tokens have has_audio=false (Gemma E4B path).
    for (const auto& msg : received) {
        EXPECT_FALSE(msg.value("has_audio", true));
    }

    const double tokensPerSecond = (elapsedMs > 0.0)
        ? (tokenCount * 1000.0 / elapsedMs) : 0.0;

    std::cerr << "\n[BENCHMARK-GEMMA] " << tokenCount << " tokens in "
              << elapsedMs << " ms (" << tokensPerSecond << " tok/s)\n";

    EXPECT_LT(elapsedMs / tokenCount, 10.0);
}

TEST_F(ConversationBenchmarkTest, TokenThroughput_NativeTtsVision_50Tokens)
{
    // Capability path: nativeTts=true, nativeVision=true.
    // Verifies pipeline overhead for the VLM code path with
    // audio+vision+text multimodal capabilities (flag-driven coverage).
    constexpr int tokenCount = 50;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    std::vector<std::string> tokens;
    tokens.reserve(tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        tokens.push_back(std::format("vlm{}", i));
    }
    mockInferencer->tokensToEmit = tokens;
    mockInferencer->nativeTts = true;     // Exercise native-TTS branch
    mockInferencer->nativeVision = true;  // Accepts video frames natively

    ConversationNode::Config cfg;
    cfg.pubPort          = kTestPubPort;
    cfg.daemonSubPort    = kTestDaemonSubPort;
    cfg.uiCommandSubPort = kTestUiCmdSubPort;
    cfg.modelDir         = "/tmp/test_model";
    cfg.modelVariant     = "gemma_e2b";
    cfg.pollTimeout      = std::chrono::milliseconds(20);

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node(cfg, std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Subscribe AFTER node.initialize() + run — avoids ZMQ slow-joiner bug.
    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "Describe what you see in this image"},
        {"sequence_id",  3},
    };

    std::vector<nlohmann::json> received;
    const auto startTime = std::chrono::steady_clock::now();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{10000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            for (int attempt = 0; attempt < tokenCount + 10; ++attempt) {
                auto msg = receiveTestMessage(responseSub, 500);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();

    node.stop();
    runThread.join();

    ASSERT_EQ(received.size(), static_cast<size_t>(tokenCount));

    // Native-TTS path: has_audio may be true.
    // Just verify all tokens were received with valid structure.
    for (const auto& msg : received) {
        EXPECT_TRUE(msg.contains("token"))
            << "VLM must emit token field in each response";
    }

    const double tokensPerSecond = (elapsedMs > 0.0)
        ? (tokenCount * 1000.0 / elapsedMs) : 0.0;

    std::cerr << "\n[BENCHMARK-LLAMA-VISION] " << tokenCount << " tokens in "
              << elapsedMs << " ms (" << tokensPerSecond << " tok/s)\n"
              << "[BENCHMARK-LLAMA-VISION] VLM pipeline overhead: "
              << (elapsedMs / tokenCount) << " ms/token\n";

    EXPECT_LT(elapsedMs / tokenCount, 10.0)
        << "VLM pipeline overhead too high -- possible ZMQ or JSON bottleneck";
}

TEST_F(ConversationBenchmarkTest, TokenThroughput_VideoConversation_50Tokens)
{
    // Benchmark video conversation path overhead: toggle + prompt + token stream.
    // Uses mock inferencer with nativeVision=true.  No actual SHM, so the node
    // falls back to text-only generate — this measures the toggle + branch overhead.
    constexpr int tokenCount = 50;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    std::vector<std::string> tokens;
    tokens.reserve(tokenCount);
    for (int i = 0; i < tokenCount; ++i) {
        tokens.push_back(std::format("vid{}", i));
    }
    mockInferencer->tokensToEmit = tokens;
    mockInferencer->nativeTts = false;   // Gemma E4B path (needs TTS sidecar)
    mockInferencer->nativeVision = true; // Vision-capable model

    ConversationNode::Config cfg;
    cfg.pubPort          = kTestPubPort;
    cfg.daemonSubPort    = kTestDaemonSubPort;
    cfg.uiCommandSubPort = kTestUiCmdSubPort;
    cfg.modelDir         = "/tmp/test_model";
    cfg.modelVariant     = "gemma_e4b";
    cfg.pollTimeout      = std::chrono::milliseconds(20);

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node(cfg, std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Subscribe AFTER node.initialize() + run — avoids ZMQ slow-joiner bug.
    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    // Enable video conversation mode
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", 2}, {"type", "video_conversation"}, {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    const nlohmann::json prompt = {
        {"v",            2},
        {"type",         "conversation_prompt"},
        {"text",         "What do you see in the camera?"},
        {"sequence_id",  1},
        {"generation_params", {
            {"temperature", 0.7},
            {"top_p", 0.9},
            {"max_tokens", 2048},
        }},
    };

    std::vector<nlohmann::json> received;
    const auto startTime = std::chrono::steady_clock::now();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{10000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            for (int attempt = 0; attempt < tokenCount + 10; ++attempt) {
                auto msg = receiveTestMessage(responseSub, 500);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    const auto endTime = std::chrono::steady_clock::now();
    const double elapsedMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();

    node.stop();
    runThread.join();

    ASSERT_EQ(received.size(), static_cast<size_t>(tokenCount));
    EXPECT_TRUE(received.back().value("finished", false));

    // Verify has_audio=false (Gemma E4B: needs TTS sidecar)
    for (const auto& msg : received) {
        EXPECT_FALSE(msg.value("has_audio", true));
    }

    const double tokensPerSecond = (elapsedMs > 0.0)
        ? (tokenCount * 1000.0 / elapsedMs) : 0.0;

    std::cerr << "\n[BENCHMARK-VIDEO-CONV] " << tokenCount << " tokens in "
              << elapsedMs << " ms (" << tokensPerSecond << " tok/s)\n"
              << "[BENCHMARK-VIDEO-CONV] Video conversation toggle + text fallback overhead: "
              << (elapsedMs / tokenCount) << " ms/token\n";

    EXPECT_LT(elapsedMs / tokenCount, 10.0)
        << "Video conversation pipeline overhead too high";
}

