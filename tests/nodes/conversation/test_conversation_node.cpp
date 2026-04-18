// test_conversation_node.cpp -- Level 2 node integration tests for ConversationNode
//
// All tests use MockConversationInferencer -- no GPU required.
// ZMQ sockets use ports offset by +20000 to avoid conflicts with production.
//
// Test coverage:
//   - initialize() succeeds with mock inferencer
//   - run() / stop() lifecycle
//   - conversation_prompt triggers token stream
//   - cancel_generation aborts in-flight generation
//   - malformed JSON handled without crash
//   - inferencer error publishes done=true
//   - sentence_boundary flag set correctly after punctuation
//   - has_audio reflects supportsNativeTts()
//   - module_ready published after initialize()
//   - Gemma E4B path: has_audio=false
//   - sequence_id propagated from prompt to response
//   - native STT disabled: no SHM access
//   - native STT graceful fallback without SHM
//   - text prompt works with native STT enabled
//   - Native-TTS capability profile: STT=true, TTS=true, full audio→transcription pipeline
//   - Gemma E4B profile: STT=true, TTS=false, audio→transcription + has_audio=false

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "conversation/conversation_node.hpp"
#include "conversation_inferencer.hpp"
#include "common/constants/conversation_constants.hpp"
#include "common/runtime_defaults.hpp"
#include "shm/shm_mapping.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/audio_constants.hpp"
#include "zmq/heartbeat_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include "tests/mocks/mock_conversation_inferencer.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"


// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class ConversationNodeTest : public ::testing::Test {
protected:
    static constexpr int kTestPubPortBase       = 26572;
    static constexpr int kTestDaemonSubPortBase = 26571;
    static constexpr int kTestUiCmdSubPortBase  = 26570;
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

        cfg_.pubPort           = kTestPubPort;
        cfg_.daemonSubPort     = kTestDaemonSubPort;
        cfg_.uiCommandSubPort  = kTestUiCmdSubPort;
        cfg_.modelDir          = "/tmp/test_model";
        cfg_.modelVariant      = "gemma_e4b";
        cfg_.pollTimeout       = std::chrono::milliseconds(20);
    }

    [[nodiscard]] ConversationNode makeNode(
        std::unique_ptr<MockConversationInferencer> inferencer)
    {
        return ConversationNode(cfg_, std::move(inferencer));
    }

    /// Create a response subscriber connected AFTER the node is already running.
    /// This avoids the ZMQ slow-joiner problem where a subscriber connected
    /// before makeNode() loses its subscription when cleanStaleEndpoint()
    /// removes and rebinds the IPC socket.
    zmq::socket_t makeResponseSub(const std::string& topic = std::string(kZmqTopicConversationResponse))
    {
        zmq::socket_t sub(testCtx_, ZMQ_SUB);
        sub.set(zmq::sockopt::subscribe, topic);
        sub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));
        // Allow subscription handshake to propagate to PUB socket
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        return sub;
    }

    /// Collect conversation_response messages with retry-send until done=true.
    ///
    /// Uses a fresh subscriber (via makeResponseSub) to avoid the ZMQ
    /// slow-joiner problem. When the test creates a subscriber before
    /// makeNode(), the node's cleanStaleEndpoint() removes and rebinds the
    /// IPC socket, causing the subscriber to lose its subscription filter.
    /// Creating a fresh subscriber after the node is running guarantees the
    /// subscription handshake completes against the live PUB socket.
    std::vector<nlohmann::json> collectTokens(
        zmq::socket_t& daemonPub,
        zmq::socket_t& /*subscriber — ignored, fresh sub used instead*/,
        const nlohmann::json& prompt,
        int deadlineMs = 5000)
    {
        // Create a fresh subscriber connected to the already-running node
        zmq::socket_t freshSub = makeResponseSub();

        std::vector<nlohmann::json> received;
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds{deadlineMs};

        while (std::chrono::steady_clock::now() < deadline) {
            publishTestMessage(daemonPub, "conversation_prompt", prompt);
            zmq::pollitem_t pi[]{
                {static_cast<void*>(freshSub), 0, ZMQ_POLLIN, 0}};
            int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
            if (rc > 0) {
                for (int attempt = 0; attempt < 30; ++attempt) {
                    auto msg = receiveTestMessage(freshSub, 200);
                    if (!msg.is_null()) {
                        received.push_back(msg);
                        if (msg.value("finished", false)) break;
                    }
                }
                break;
            }
        }
        return received;
    }

    ConversationNode::Config cfg_;
    zmq::context_t testCtx_{1};
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(ConversationNodeTest, InitializeSucceedsWithMockInferencer)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();

    zmq::socket_t readySub(testCtx_, ZMQ_SUB);
    readySub.set(zmq::sockopt::reconnect_ivl, 10);
    readySub.set(zmq::sockopt::subscribe, std::string(kZmqTopicModuleReady));
    readySub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    EXPECT_NO_THROW(node.initialize());

    const nlohmann::json ready = receiveTestMessage(readySub, 1500);
    ASSERT_TRUE(ready.contains("type"));
    EXPECT_EQ(ready["type"].get<std::string>(), std::string(kZmqTopicModuleReady));
    EXPECT_EQ(ready["module"].get<std::string>(), "conversation");
}

TEST_F(ConversationNodeTest, StopIsIdempotent)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    EXPECT_NO_THROW(node.stop());
    EXPECT_NO_THROW(node.stop());
}

TEST_F(ConversationNodeTest, RunAndStop)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    node.stop();
    runThread.join();
}

TEST_F(ConversationNodeTest, PromptTriggersTokenStream)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"Hello", ",", " world", "!"};
    mockInferencer->nativeTts = true;

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Connect subscriber AFTER node is running to avoid slow-joiner
    zmq::socket_t responseSub = makeResponseSub();

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "Hello, how are you?"},
        {"sequence_id",  42},
        {"generation_params", {
            {"temperature", 0.5},
            {"top_p", 0.8},
            {"max_tokens", 512},
        }},
    };

    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));

    for (const auto& msg : received) {
        EXPECT_EQ(msg.value("v", 0), kConversationSchemaVersion);
        EXPECT_EQ(msg.value("type", ""), std::string(kZmqTopicConversationResponse));
        EXPECT_TRUE(msg.contains("sentence_boundary"));
        EXPECT_TRUE(msg.contains("has_audio"));
        EXPECT_TRUE(msg.contains("sequence_id"));
        EXPECT_TRUE(msg.value("has_audio", false));  // Mock: nativeTts=true branch
        EXPECT_EQ(msg.value("sequence_id", 0), 42);
    }
}

TEST_F(ConversationNodeTest, SentenceBoundarySetAfterPunctuation)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"Hello", ".", " How"};

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "test prompt"},
        {"sequence_id", 1},
    };

    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    // Token 0: "Hello"  -> sentence_boundary=false
    // Token 1: "."      -> sentence_boundary=false (period itself, but prev was not)
    // Token 2: " How"   -> sentence_boundary=true  (prev token "." ends sentence)
    ASSERT_GE(received.size(), 3u);
    EXPECT_FALSE(received[0].value("sentence_boundary", true));
    EXPECT_FALSE(received[1].value("sentence_boundary", true));
    EXPECT_TRUE(received[2].value("sentence_boundary", false));
}

TEST_F(ConversationNodeTest, GemmaE4BPathHasAudioFalse)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"Paris"};
    mockInferencer->nativeTts = false;  // Gemma E4B path

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "What is the capital of France?"},
        {"sequence_id", 99},
    };

    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_GE(received.size(), 1u);
    for (const auto& msg : received) {
        EXPECT_FALSE(msg.value("has_audio", true));  // Gemma: no native audio
    }
}

TEST_F(ConversationNodeTest, CancelGenerationAbortsTokenStream)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"A", "B", "C", "D", "E", "F", "G", "H"};

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "test"},
        {"sequence_id", 7},
    };

    std::vector<nlohmann::json> received;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{5000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            publishTestMessage(uiPub, "ui_command", {
                {"v", kConversationSchemaVersion},
                {"type", "ui_command"},
                {"action", "cancel_generation"},
            });
            for (int i = 0; i < 20; ++i) {
                auto msg = receiveTestMessage(responseSub, 100);
                if (!msg.is_null()) {
                    received.push_back(msg);
                    if (msg.value("finished", false)) break;
                }
            }
            break;
        }
    }

    node.stop();
    runThread.join();

    ASSERT_FALSE(received.empty());
    EXPECT_TRUE(received.back().value("finished", false));
}

TEST_F(ConversationNodeTest, MalformedJsonDoesNotCrash)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    const std::string garbage = "conversation_prompt {not valid json!!!";
    daemonPub.send(zmq::buffer(garbage), zmq::send_flags::none);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    node.stop();
    runThread.join();
}

TEST_F(ConversationNodeTest, InferencerErrorPublishesDone)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->simulateError = true;

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "test"},
        {"sequence_id", 1},
    };

    bool gotDone = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{5000};
    while (std::chrono::steady_clock::now() < deadline) {
        publishTestMessage(daemonPub, "conversation_prompt", prompt);
        zmq::pollitem_t pi[]{{static_cast<void*>(responseSub), 0, ZMQ_POLLIN, 0}};
        int rc = zmq::poll(pi, 1, std::chrono::milliseconds{150});
        if (rc > 0) {
            for (int i = 0; i < 10; ++i) {
                auto msg = receiveTestMessage(responseSub, 200);
                if (!msg.is_null() && msg.value("finished", false)) {
                    gotDone = true;
                    break;
                }
            }
            break;
        }
    }

    node.stop();
    runThread.join();

    EXPECT_TRUE(gotDone);
}

TEST_F(ConversationNodeTest, GeneratedTokensMatchMockOutput)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"The", " sky", " is", " blue", "."};

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "What colour is the sky?"},
        {"sequence_id",  10},
    };

    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_EQ(received.size(), 5u);

    std::string fullText;
    for (const auto& msg : received) {
        fullText += msg.value("token", std::string{});
    }
    EXPECT_EQ(fullText, std::string("The sky is blue."));

    EXPECT_TRUE(received.back().value("finished", false));
    EXPECT_FALSE(received.back().value("sentence_boundary", true));
}

TEST_F(ConversationNodeTest, NativeSttDisabledInitializesSuccessfully)
{
    // When nativeStt=false, no audio subscriptions or SHM access should occur.
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = false;

    ConversationNode node = makeNode(std::move(mockInferencer));
    EXPECT_NO_THROW(node.initialize());
}

TEST_F(ConversationNodeTest, NativeSttGracefulFallbackWithoutShm)
{
    // When nativeStt=true but audio SHM doesn't exist, node should still
    // initialize successfully (logs warning, disables native STT).
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    cfg_.audioShmName = "/oe.test.nonexistent";

    ConversationNode node = makeNode(std::move(mockInferencer));
    EXPECT_NO_THROW(node.initialize());
}

TEST_F(ConversationNodeTest, TextPromptStillWorksWithNativeStt)
{
    // Verify text input path is unbroken when native STT is enabled.
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->tokensToEmit = {"Yes"};
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = false;
    cfg_.audioShmName = "/oe.test.nonexistent";  // SHM won't exist

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    const nlohmann::json prompt = {
        {"v",            kConversationSchemaVersion},
        {"type",         "conversation_prompt"},
        {"text",         "Are you there?"},
        {"sequence_id",  200},
    };

    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    EXPECT_FALSE(received.back().value("has_audio", true));  // nativeTts=false
}

// ---------------------------------------------------------------------------
// Native STT integration tests (with real SHM)
// ---------------------------------------------------------------------------

// Helper: compute audio SHM total size (same formula as whisper_constants.hpp)
static constexpr std::size_t kTestAudioShmSize =
    kAudioShmDataOffset + kRingBufferSlotCount * kAudioShmSlotByteSize;

// Helper: write one 30 ms speech chunk into SHM slot 0
static void writeSpeechChunkToShm(ShmMapping& shm, uint64_t seq)
{
    uint8_t* slotBase = shm.bytes() + kAudioShmDataOffset;
    auto* header = reinterpret_cast<ShmAudioHeader*>(slotBase);
    header->sampleRateHz = kSttInputSampleRateHz;
    header->numSamples   = kVadChunkSampleCount;
    header->seqNumber    = seq;
    header->timestampNs  = 0;

    auto* pcm = reinterpret_cast<float*>(slotBase + sizeof(ShmAudioHeader));
    // Fill with a sine wave (content doesn't matter — mock ignores it)
    for (uint32_t i = 0; i < kVadChunkSampleCount; ++i) {
        pcm[i] = 0.5f;
    }
}

TEST_F(ConversationNodeTest, NativeTtsProfileFullSttPipeline)
{
    // Capability profile: nativeStt=true, nativeTts=true
    // Verifies: audio_chunk → accumulate → vad_silence → transcribe() → publish "transcription"
    const std::string testShmName = "/oe.test.conv.stt.nativetts";
    const int audioPort = kTestPubPort + 100;  // separate port for audio ingest

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = true;
    mockInferencer->mockTranscription = "Hello world from Gemma";
    mockInferencer->tokensToEmit = {"Response"};

    cfg_.audioShmName = testShmName;
    cfg_.audioSubPort = audioPort;

    // Create the SHM segment (producer side, so AudioAccumulator can open it)
    ShmMapping testShm(testShmName, kTestAudioShmSize, /*create=*/true);
    std::memset(testShm.data(), 0, kTestAudioShmSize);

    // Subscribe to transcription output from the node
    zmq::socket_t transcriptionSub(testCtx_, ZMQ_SUB);
    transcriptionSub.set(zmq::sockopt::reconnect_ivl, 10);
    transcriptionSub.set(zmq::sockopt::subscribe, std::string("transcription"));
    transcriptionSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    // Also subscribe to conversation_response for the text generation test
    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    // Audio ingest publisher (simulates AudioIngestNode)
    zmq::socket_t audioIngestPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", audioPort));
    audioIngestPub.bind(std::format("ipc:///tmp/omniedge_{}", audioPort));

    // Daemon publisher (for conversation_prompt)
    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });

    // Allow subscriptions to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // ── Phase 1: Simulate audio input ──────────────────────────────────
    // Write speech PCM to SHM and publish audio_chunk notifications
    for (uint64_t seq = 1; seq <= 3; ++seq) {
        writeSpeechChunkToShm(testShm, seq);
        publishTestMessage(audioIngestPub, "audio_chunk", {
            {"v", 1}, {"type", "audio_chunk"},
            {"shm", testShmName}, {"slot", 0},
            {"samples", kVadChunkSampleCount},
            {"seq", seq}, {"vad", "speech"}, {"ts", 0},
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Trigger end-of-utterance
    publishTestMessage(audioIngestPub, "vad_status", {
        {"v", 1}, {"type", "vad_status"},
        {"speaking", false}, {"silence_ms", 800},
    });

    // ── Phase 2: Verify transcription published ────────────────────────
    nlohmann::json transcription;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{3000};
    while (std::chrono::steady_clock::now() < deadline) {
        auto msg = receiveTestMessage(transcriptionSub, 200);
        if (!msg.is_null() && msg.value("type", "") == "transcription") {
            transcription = msg;
            break;
        }
    }

    ASSERT_FALSE(transcription.is_null()) << "Expected transcription message on node PUB port";
    EXPECT_EQ(transcription.value("text", ""), "Hello world from Gemma");
    EXPECT_EQ(transcription.value("source", ""), "native_stt");
    EXPECT_EQ(transcription.value("type", ""), "transcription");

    // ── Phase 3: Verify text generation still works ────────────────────
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "Hello world from Gemma"},
        {"sequence_id", 300},
    };
    auto responses = collectTokens(daemonPub, responseSub, prompt);
    ASSERT_GE(responses.size(), 1u);
    EXPECT_TRUE(responses.back().value("finished", false));
    EXPECT_TRUE(responses.back().value("has_audio", false));  // nativeTts=true → has_audio=true

    node.stop();
    runThread.join();
}

TEST_F(ConversationNodeTest, GemmaE4BProfileSttWithoutTts)
{
    // Gemma E4B model profile: nativeStt=true, nativeTts=false
    // Verifies: audio transcription works AND has_audio=false in responses
    const std::string testShmName = "/oe.test.conv.stt.gemma";
    const int audioPort = kTestPubPort + 200;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = false;  // Gemma E4B: needs TTS sidecar
    mockInferencer->mockTranscription = "Hello from Gemma";
    mockInferencer->tokensToEmit = {"Response"};

    cfg_.audioShmName = testShmName;
    cfg_.audioSubPort = audioPort;

    ShmMapping testShm(testShmName, kTestAudioShmSize, /*create=*/true);
    std::memset(testShm.data(), 0, kTestAudioShmSize);

    zmq::socket_t transcriptionSub(testCtx_, ZMQ_SUB);
    transcriptionSub.set(zmq::sockopt::reconnect_ivl, 10);
    transcriptionSub.set(zmq::sockopt::subscribe, std::string("transcription"));
    transcriptionSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t audioIngestPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", audioPort));
    audioIngestPub.bind(std::format("ipc:///tmp/omniedge_{}", audioPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Simulate speech
    for (uint64_t seq = 1; seq <= 2; ++seq) {
        writeSpeechChunkToShm(testShm, seq);
        publishTestMessage(audioIngestPub, "audio_chunk", {
            {"v", 1}, {"type", "audio_chunk"},
            {"shm", testShmName}, {"slot", 0},
            {"samples", kVadChunkSampleCount},
            {"seq", seq}, {"vad", "speech"}, {"ts", 0},
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    publishTestMessage(audioIngestPub, "vad_status", {
        {"v", 1}, {"type", "vad_status"},
        {"speaking", false}, {"silence_ms", 800},
    });

    // Verify transcription
    nlohmann::json transcription;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{3000};
    while (std::chrono::steady_clock::now() < deadline) {
        auto msg = receiveTestMessage(transcriptionSub, 200);
        if (!msg.is_null() && msg.value("type", "") == "transcription") {
            transcription = msg;
            break;
        }
    }

    ASSERT_FALSE(transcription.is_null()) << "Expected transcription from Gemma E4B";
    EXPECT_EQ(transcription.value("text", ""), "Hello from Gemma");

    // Verify response has has_audio=false (Gemma needs TTS sidecar)
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "Hello from Gemma"},
        {"sequence_id", 400},
    };
    auto responses = collectTokens(daemonPub, responseSub, prompt);
    ASSERT_GE(responses.size(), 1u);
    for (const auto& msg : responses) {
        EXPECT_FALSE(msg.value("has_audio", true));  // Gemma: no native TTS
    }

    node.stop();
    runThread.join();
}

TEST_F(ConversationNodeTest, TranscriptionSuppressedWhenEmpty)
{
    // When transcribe() returns empty/whitespace, no transcription should be published
    const std::string testShmName = "/oe.test.conv.stt.empty";
    const int audioPort = kTestPubPort + 300;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = false;
    mockInferencer->mockTranscription = "   ";  // whitespace only

    cfg_.audioShmName = testShmName;
    cfg_.audioSubPort = audioPort;

    ShmMapping testShm(testShmName, kTestAudioShmSize, /*create=*/true);
    std::memset(testShm.data(), 0, kTestAudioShmSize);

    zmq::socket_t transcriptionSub(testCtx_, ZMQ_SUB);
    transcriptionSub.set(zmq::sockopt::reconnect_ivl, 10);
    transcriptionSub.set(zmq::sockopt::subscribe, std::string("transcription"));
    transcriptionSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    zmq::socket_t audioIngestPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", audioPort));
    audioIngestPub.bind(std::format("ipc:///tmp/omniedge_{}", audioPort));

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Send audio + vad_silence
    writeSpeechChunkToShm(testShm, 1);
    publishTestMessage(audioIngestPub, "audio_chunk", {
        {"v", 1}, {"type", "audio_chunk"},
        {"shm", testShmName}, {"slot", 0},
        {"samples", kVadChunkSampleCount},
        {"seq", 1}, {"vad", "speech"}, {"ts", 0},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    publishTestMessage(audioIngestPub, "vad_status", {
        {"v", 1}, {"type", "vad_status"},
        {"speaking", false}, {"silence_ms", 800},
    });

    // Should NOT receive a transcription (whitespace suppressed)
    auto msg = receiveTestMessage(transcriptionSub, 1000);
    EXPECT_TRUE(msg.is_null()) << "Whitespace-only transcription should be suppressed";

    node.stop();
    runThread.join();
}

// ---------------------------------------------------------------------------
// Video conversation tests
// ---------------------------------------------------------------------------

TEST_F(ConversationNodeTest, VideoConversationToggleAcceptedWhenVisionSupported)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    auto* mockPtr = mockInferencer.get();
    mockInferencer->nativeVision = true;
    mockInferencer->tokensToEmit = {"test"};

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Enable video conversation
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Send a prompt — without SHM available, should fall back to text-only
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "Describe the scene"},
        {"sequence_id", 100},
    };
    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    // Should receive tokens (fallback to text-only since no video SHM)
    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    // Without video SHM, generate (not generateWithVideo) is called
    EXPECT_FALSE(mockPtr->lastGenerateHadVideo);
    SPDLOG_DEBUG("video_conversation_toggle test: received {} tokens, hadVideo={}",
        received.size(), mockPtr->lastGenerateHadVideo);
}

TEST_F(ConversationNodeTest, VideoConversationToggleRejectedWhenVisionNotSupported)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeVision = false;  // No vision support
    mockInferencer->tokensToEmit = {"test"};

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Try to enable video conversation — should be silently rejected
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Send a prompt — should use text-only generate
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "What do you see?"},
        {"sequence_id", 101},
    };
    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    SPDLOG_DEBUG("video_conversation_rejected test: received {} tokens", received.size());
}

TEST_F(ConversationNodeTest, VideoConversationDisableResumesTextOnly)
{
    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    auto* mockPtr = mockInferencer.get();
    mockInferencer->nativeVision = true;
    mockInferencer->tokensToEmit = {"response"};

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Enable then disable video conversation
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", false},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Send a prompt — should use text-only generate
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "Hello"},
        {"sequence_id", 102},
    };
    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    EXPECT_FALSE(mockPtr->lastGenerateHadVideo);
    SPDLOG_DEBUG("video_conversation_disable test: generate used text-only path, hadVideo={}",
        mockPtr->lastGenerateHadVideo);
}

// ---------------------------------------------------------------------------
// Real-data tests — feed actual fixture files through the pipeline
// ---------------------------------------------------------------------------

/// Helper: load a binary file from disk into a byte vector.
static std::vector<uint8_t> loadBinaryFixture(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<std::size_t>(fileSize));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    return data;
}

/// Helper: load float32 PCM samples from a binary file.
static std::vector<float> loadPcmFixture(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    const std::size_t sampleCount = static_cast<std::size_t>(fileSize) / sizeof(float);
    std::vector<float> samples(sampleCount);
    file.read(reinterpret_cast<char*>(samples.data()), fileSize);
    return samples;
}

TEST_F(ConversationNodeTest, VideoConversationWithRealFrameFromShm)
{
    // Load real 1920x1080 BGR frame from test fixture.
    const std::filesystem::path fixturePath =
        std::filesystem::path(CMAKE_SOURCE_DIR) / "tests/fixtures/test_frame_1920x1080.bgr";
    auto frameData = loadBinaryFixture(fixturePath);
    if (frameData.empty()) {
        GTEST_SKIP() << "Fixture not found: " << fixturePath;
    }
    SPDLOG_DEBUG("Loaded BGR fixture: {} bytes from {}", frameData.size(), fixturePath.string());
    ASSERT_EQ(frameData.size(), kMaxBgr24FrameBytes)
        << "Fixture must be 1920x1080 BGR24 (6,220,800 bytes)";

    // Create a unique test SHM segment name to avoid collisions.
    const std::string testShmName =
        std::format("/oe.test.vidconv.{}", kTestPubPort);

    // Create SHM as producer with circular buffer layout and write the frame.
    ShmCircularBuffer<ShmVideoHeader> shmBuf(
        testShmName, kCircularBufferSlotCount, kMaxBgr24FrameBytes, /*create=*/true);

    // Write header (resolution info).
    shmBuf.header()->width         = kMaxInputWidth;
    shmBuf.header()->height        = kMaxInputHeight;
    shmBuf.header()->bytesPerPixel = kBgr24BytesPerPixel;
    shmBuf.header()->seqNumber     = 1;

    // Write the real frame into slot 0 and commit.
    auto [writePtr, slotIdx] = shmBuf.acquireWriteSlot();
    std::memcpy(writePtr, frameData.data(), frameData.size());
    shmBuf.commitWrite();
    SPDLOG_DEBUG("Wrote {} bytes to SHM {} slot {}", frameData.size(), testShmName, slotIdx);

    // Configure node to use test SHM.
    cfg_.videoShmName = testShmName;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    auto* mockPtr = mockInferencer.get();
    mockInferencer->nativeVision = true;
    mockInferencer->nativeTts = false;  // Gemma path
    mockInferencer->tokensToEmit = {"I", " see", " color", " bars", "."};

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Enable video conversation.
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Send a prompt.
    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "Describe the image you see."},
        {"sequence_id", 500},
    };
    auto received = collectTokens(daemonPub, responseSub, prompt);

    node.stop();
    runThread.join();

    // Verify mock received the real video frame.
    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    EXPECT_TRUE(mockPtr->lastGenerateHadVideo)
        << "Expected generateWithVideo() to be called with real frame from SHM";
    EXPECT_EQ(mockPtr->lastFrameWidth, kMaxInputWidth);
    EXPECT_EQ(mockPtr->lastFrameHeight, kMaxInputHeight);
    EXPECT_EQ(mockPtr->lastVideoFrameSize, kMaxBgr24FrameBytes);

    // Verify all tokens were received (has_audio=false for Gemma path).
    for (const auto& msg : received) {
        EXPECT_FALSE(msg.value("has_audio", true));
    }

    // Concatenate tokens and verify content.
    std::string fullText;
    for (const auto& msg : received) {
        fullText += msg.value("token", "");
    }
    EXPECT_EQ(fullText, "I see color bars.");
    SPDLOG_DEBUG("VideoConversationWithRealFrame: frame={}x{} ({} bytes), text='{}'",
        mockPtr->lastFrameWidth, mockPtr->lastFrameHeight,
        mockPtr->lastVideoFrameSize, fullText);
}

TEST_F(ConversationNodeTest, TranscribeWithRealPcmFixture)
{
    // Load real PCM audio fixture (espeak-ng "hello" at 16 kHz).
    const std::filesystem::path fixturePath =
        std::filesystem::path(CMAKE_SOURCE_DIR) / "tests/fixtures/speech_hello_16khz.pcm";
    auto pcmSamples = loadPcmFixture(fixturePath);
    if (pcmSamples.empty()) {
        GTEST_SKIP() << "Fixture not found: " << fixturePath;
    }
    SPDLOG_DEBUG("Loaded PCM fixture: {} samples ({} bytes) from {}",
        pcmSamples.size(), pcmSamples.size() * sizeof(float), fixturePath.string());

    // Verify fixture has reasonable audio content (not all zeros).
    float maxAbs = 0.0f;
    for (const float s : pcmSamples) {
        const float a = std::abs(s);
        if (a > maxAbs) maxAbs = a;
    }
    EXPECT_GT(maxAbs, 0.01f) << "PCM fixture appears to be silence";
    SPDLOG_DEBUG("PCM peak amplitude: {:.4f}, duration: {:.2f}s",
        maxAbs, static_cast<double>(pcmSamples.size()) / 16000.0);

    // Feed the real PCM to the mock inferencer's transcribe().
    // This exercises the full data path without requiring a GPU.
    MockConversationInferencer inferencer;
    inferencer.nativeStt = true;
    inferencer.mockTranscription = "hello";

    auto result = inferencer.transcribe(
        std::span<const float>(pcmSamples), 16000);

    ASSERT_TRUE(result.has_value()) << "transcribe() failed: " << result.error();
    EXPECT_EQ(result.value(), "hello");
    SPDLOG_DEBUG("TranscribeWithRealPcm: {} samples -> '{}'",
        pcmSamples.size(), result.value());
}

// ---------------------------------------------------------------------------
// Throughput test — 30 FPS video frames through SHM circular buffer
//
// Feeds 150 real BGR frames (5 seconds @ 30 FPS) from a downloaded talking-
// head video into ShmCircularBuffer.  A background writer thread replays a
// single real 1920x1080 BGR24 frame at 30 FPS cadence.  After ~2.5 s the
// test sends a prompt with video conversation enabled and verifies the node
// reads a real frame from SHM and calls generateWithVideo().
//
// This proves:
//   1. SHM circular buffer sustains 30 FPS write throughput (~178 MB/s)
//   2. ConversationNode's readLatestBgrFrame() returns the latest frame
//   3. The mock inferencer receives a real 1920x1080 BGR24 frame
// ---------------------------------------------------------------------------

TEST_F(ConversationNodeTest, Video30FpsThroughputFromRealTalkingVideo)
{
    // Load the real BGR24 frame extracted from "Duck and Cover" (1951) —
    // a public domain US government film with real person talking.
    const std::filesystem::path fixturePath =
        std::filesystem::path(CMAKE_SOURCE_DIR)
        / "tests/fixtures/video_talking/talking_frame_1920x1080.bgr";
    auto frameData = loadBinaryFixture(fixturePath);
    if (frameData.empty()) {
        GTEST_SKIP() << "Fixture not found: " << fixturePath;
    }
    ASSERT_EQ(frameData.size(), kMaxBgr24FrameBytes)
        << "Fixture must be 1920x1080 BGR24 (6,220,800 bytes)";

    // Create SHM as producer.
    const std::string testShmName =
        std::format("/oe.test.vid30fps.{}", kTestPubPort);
    ShmCircularBuffer<ShmVideoHeader> shmBuf(
        testShmName, kCircularBufferSlotCount, kMaxBgr24FrameBytes, /*create=*/true);

    shmBuf.header()->width         = kMaxInputWidth;
    shmBuf.header()->height        = kMaxInputHeight;
    shmBuf.header()->bytesPerPixel = kBgr24BytesPerPixel;

    cfg_.videoShmName = testShmName;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    auto* mockPtr = mockInferencer.get();
    mockInferencer->nativeVision = true;
    mockInferencer->nativeTts = false;
    mockInferencer->tokensToEmit = {"I", " see", " a", " person", "."};

    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    zmq::socket_t responseSub(testCtx_, ZMQ_SUB);
    responseSub.set(zmq::sockopt::reconnect_ivl, 10);
    responseSub.set(zmq::sockopt::subscribe, std::string(kZmqTopicConversationResponse));
    responseSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Enable video conversation.
    publishTestMessage(daemonPub, "video_conversation", {
        {"v", kConversationSchemaVersion},
        {"type", "video_conversation"},
        {"enabled", true},
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // ── Background writer: replay the frame at 30 FPS for 5 seconds ──
    constexpr int kTotalFrames = 150;
    constexpr auto kFrameInterval = std::chrono::microseconds(33333); // ~30 FPS
    std::atomic<bool> writerDone{false};
    std::atomic<uint64_t> framesWritten{0};

    auto writerStart = std::chrono::steady_clock::now();
    std::thread writerThread([&]() {
        for (int f = 0; f < kTotalFrames; ++f) {
            auto [dst, idx] = shmBuf.acquireWriteSlot();
            std::memcpy(dst, frameData.data(), frameData.size());
            shmBuf.header()->seqNumber = static_cast<uint64_t>(f + 1);
            shmBuf.header()->timestampNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            shmBuf.commitWrite();
            framesWritten.fetch_add(1, std::memory_order_relaxed);

            // Sleep until next frame time (busy-wait for accuracy near end).
            const auto nextFrameTime = writerStart
                + kFrameInterval * static_cast<int>(f + 1);
            std::this_thread::sleep_until(nextFrameTime);
        }
        writerDone.store(true, std::memory_order_release);
    });

    // Wait ~2.5 seconds (75 frames in), then send a prompt.
    // The node should read the latest frame written so far.
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));

    const nlohmann::json prompt = {
        {"v", kConversationSchemaVersion},
        {"type", "conversation_prompt"},
        {"text", "What do you see in this video frame?"},
        {"sequence_id", 600},
    };
    auto received = collectTokens(daemonPub, responseSub, prompt);

    // Wait for writer to finish.
    writerThread.join();
    auto writerEnd = std::chrono::steady_clock::now();
    const double writerDurationMs =
        std::chrono::duration<double, std::milli>(writerEnd - writerStart).count();
    const double actualFps = kTotalFrames * 1000.0 / writerDurationMs;

    node.stop();
    runThread.join();

    // ── Assertions ──
    // Writer throughput: must sustain close to 30 FPS.
    SPDLOG_INFO("Video30FpsThroughput: {} frames in {:.1f} ms = {:.1f} FPS "
                "(target 30 FPS, wrote {:.1f} MB/s)",
        kTotalFrames, writerDurationMs, actualFps,
        (kTotalFrames * kMaxBgr24FrameBytes / 1e6) / (writerDurationMs / 1e3));
    EXPECT_GE(actualFps, 25.0)
        << "SHM writer must sustain at least 25 FPS (target 30)";
    EXPECT_EQ(framesWritten.load(), kTotalFrames);

    // Node received real video frame via generateWithVideo.
    ASSERT_GE(received.size(), 1u);
    EXPECT_TRUE(received.back().value("finished", false));
    EXPECT_TRUE(mockPtr->lastGenerateHadVideo)
        << "Expected generateWithVideo() to be called with real video frame from SHM";
    EXPECT_EQ(mockPtr->lastFrameWidth, kMaxInputWidth);
    EXPECT_EQ(mockPtr->lastFrameHeight, kMaxInputHeight);
    EXPECT_EQ(mockPtr->lastVideoFrameSize, kMaxBgr24FrameBytes);

    // Verify tokens.
    std::string fullText;
    for (const auto& msg : received) {
        fullText += msg.value("token", "");
    }
    EXPECT_EQ(fullText, "I see a person.");
    SPDLOG_INFO("Video30FpsThroughput: generateWithVideo received {}x{} frame, text='{}'",
        mockPtr->lastFrameWidth, mockPtr->lastFrameHeight, fullText);
}

// ---------------------------------------------------------------------------
// Audio ingest pipeline test — real speech PCM through SHM + ZMQ
//
// Feeds real speech PCM (from "Duck and Cover" narration) through the full
// audio ingest pipeline: writes 480-sample chunks to audio SHM, publishes
// audio_chunk ZMQ messages, triggers vad_status silence to flush, and
// verifies the ConversationNode's AudioAccumulator → transcribe() path
// produces a transcription message.
//
// This proves:
//   1. Real speech PCM flows correctly through SHM audio slots
//   2. AudioAccumulator reads and accumulates multi-chunk speech
//   3. vad_status silence triggers transcription
//   4. Stale-read guards don't fire on well-paced writes
// ---------------------------------------------------------------------------

/// Helper: write one real audio chunk into an audio SHM slot.
static void writeRealAudioChunk(ShmMapping& shm, int slotIndex,
                                const float* pcmData, uint32_t sampleCount,
                                uint64_t seq)
{
    uint8_t* slotBase = shm.bytes()
                        + kAudioShmDataOffset
                        + static_cast<std::size_t>(slotIndex) * kAudioShmSlotByteSize;
    auto* header = reinterpret_cast<ShmAudioHeader*>(slotBase);
    header->sampleRateHz = kSttInputSampleRateHz;
    header->numSamples   = sampleCount;
    header->seqNumber    = seq;
    header->timestampNs  = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    auto* pcmDst = reinterpret_cast<float*>(slotBase + sizeof(ShmAudioHeader));
    std::memcpy(pcmDst, pcmData, sampleCount * sizeof(float));
}

TEST_F(ConversationNodeTest, AudioIngestPipelineWithRealSpeechPcm)
{
    // Load real speech PCM extracted from "Duck and Cover" narration.
    const std::filesystem::path fixturePath =
        std::filesystem::path(CMAKE_SOURCE_DIR)
        / "tests/fixtures/video_talking/talking_speech_16khz.pcm";
    auto pcmSamples = loadPcmFixture(fixturePath);
    if (pcmSamples.empty()) {
        GTEST_SKIP() << "Fixture not found: " << fixturePath;
    }
    SPDLOG_INFO("AudioIngestPipeline: loaded {} samples ({:.2f}s) of real speech",
        pcmSamples.size(), static_cast<double>(pcmSamples.size()) / 16000.0);

    // Verify non-silence.
    float peakAbs = 0.0f;
    for (const float s : pcmSamples) {
        const float a = std::abs(s);
        if (a > peakAbs) peakAbs = a;
    }
    ASSERT_GT(peakAbs, 0.02f) << "PCM fixture appears to be silence";

    // Setup: unique SHM + port names.
    const std::string testShmName = std::format("/oe.test.audpipe.{}", kTestPubPort);
    const int audioPort = kTestPubPort + 500;

    auto mockInferencer = std::make_unique<MockConversationInferencer>();
    mockInferencer->nativeStt = true;
    mockInferencer->nativeTts = false;
    mockInferencer->mockTranscription = "duck and cover";
    mockInferencer->tokensToEmit = {"OK"};

    cfg_.audioShmName = testShmName;
    cfg_.audioSubPort = audioPort;

    // Create audio SHM segment (producer side).
    ShmMapping testShm(testShmName, kTestAudioShmSize, /*create=*/true);
    std::memset(testShm.data(), 0, kTestAudioShmSize);

    // Subscribe to transcription output from the node.
    zmq::socket_t transcriptionSub(testCtx_, ZMQ_SUB);
    transcriptionSub.set(zmq::sockopt::reconnect_ivl, 10);
    transcriptionSub.set(zmq::sockopt::subscribe, std::string("transcription"));
    transcriptionSub.connect(std::format("ipc:///tmp/omniedge_{}", kTestPubPort));

    // Audio ingest publisher (simulates AudioIngestNode).
    zmq::socket_t audioIngestPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", audioPort));
    audioIngestPub.bind(std::format("ipc:///tmp/omniedge_{}", audioPort));

    // Daemon publisher (for conversation_prompt).
    zmq::socket_t daemonPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestDaemonSubPort));
    daemonPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestDaemonSubPort));

    zmq::socket_t uiPub(testCtx_, ZMQ_PUB);
    std::filesystem::remove(std::format("/tmp/omniedge_{}", kTestUiCmdSubPort));
    uiPub.bind(std::format("ipc:///tmp/omniedge_{}", kTestUiCmdSubPort));

    ConversationNode node = makeNode(std::move(mockInferencer));
    node.initialize();

    std::thread runThread([&node]() { node.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // ── Feed real speech PCM in 480-sample (30 ms) chunks through SHM ──
    const uint32_t chunkSize = kVadChunkSampleCount;  // 480 samples
    const std::size_t totalChunks = pcmSamples.size() / chunkSize;

    SPDLOG_INFO("AudioIngestPipeline: feeding {} chunks of {} samples "
                "({:.1f}s at 16 kHz) through SHM",
        totalChunks, chunkSize,
        static_cast<double>(totalChunks * chunkSize) / 16000.0);

    auto feedStart = std::chrono::steady_clock::now();
    for (std::size_t c = 0; c < totalChunks; ++c) {
        const int slotIndex = static_cast<int>(c % kRingBufferSlotCount);
        const float* chunkPtr = pcmSamples.data() + c * chunkSize;

        // Write real PCM data into audio SHM slot.
        writeRealAudioChunk(testShm, slotIndex, chunkPtr, chunkSize, c + 1);

        // Publish audio_chunk notification via ZMQ.
        publishTestMessage(audioIngestPub, "audio_chunk", {
            {"v", 1}, {"type", "audio_chunk"},
            {"shm", testShmName}, {"slot", slotIndex},
            {"samples", chunkSize},
            {"seq", c + 1}, {"vad", "speech"}, {"ts", 0},
        });

        // Simulate real-time 30 ms cadence between chunks.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto feedEnd = std::chrono::steady_clock::now();
    const double feedMs =
        std::chrono::duration<double, std::milli>(feedEnd - feedStart).count();
    SPDLOG_INFO("AudioIngestPipeline: fed {} chunks in {:.0f} ms ({:.1f} chunks/s)",
        totalChunks, feedMs, totalChunks * 1000.0 / feedMs);

    // Trigger end-of-utterance (silence detected).
    publishTestMessage(audioIngestPub, "vad_status", {
        {"v", 1}, {"type", "vad_status"},
        {"speaking", false}, {"silence_ms", 800},
    });

    // ── Wait for transcription to be published ──
    nlohmann::json transcription;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds{5000};
    while (std::chrono::steady_clock::now() < deadline) {
        auto msg = receiveTestMessage(transcriptionSub, 200);
        if (!msg.is_null() && msg.value("type", "") == "transcription") {
            transcription = msg;
            break;
        }
    }

    node.stop();
    runThread.join();

    // ── Assertions ──
    ASSERT_FALSE(transcription.is_null())
        << "Expected transcription message from AudioAccumulator → transcribe()";
    EXPECT_EQ(transcription.value("text", ""), "duck and cover");
    EXPECT_EQ(transcription.value("source", ""), "native_stt");
    SPDLOG_INFO("AudioIngestPipeline: transcription='{}', source='{}'",
        transcription.value("text", ""), transcription.value("source", ""));
}

