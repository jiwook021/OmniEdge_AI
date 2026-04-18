#include <gtest/gtest.h>

#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/constants/conversation_constants.hpp"
#include "tts_inferencer.hpp"
#include "tts/tts_node.hpp"
#include "tests/mocks/mock_tts_inferencer.hpp"

// ---------------------------------------------------------------------------
// Unit tests for TTSNode.
//
// All tests inject MockTTSInferencer so no GPU, ONNX model, or SHM is required.
// ZMQ ports are offset by +20000 to avoid conflicts with production.
//
// CPU-only — no LABELS gpu.
// ---------------------------------------------------------------------------


// ===========================================================================
// Test fixture
// ===========================================================================

class TTSNodeTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		// Use test ports (+20000 offset from production) to avoid conflicts
		// with any running OmniEdge_AI instance.
		config_.subLlmPort    = 25561;
		config_.subDaemonPort = 25571;
		config_.pubPort       = 25565;
		// Use a test SHM name unique to this test run.
		config_.shmOutput     = "/oe.aud.tts.test";
		// Fast poll timeout for snappy tests.
		config_.pollTimeout   = std::chrono::milliseconds(10);

		mock_ = new MinimalMockInferencer();
		node_ = std::make_unique<TTSNode>(
			config_,
			std::unique_ptr<TTSInferencer>(mock_));  // node takes ownership
	}

	void TearDown() override
	{
		// Ensure the SHM segment is cleaned up.
		shm_unlink(config_.shmOutput.c_str());
	}

	TTSNode::Config        config_;
	MinimalMockInferencer*          mock_{nullptr};  // borrowed; node_ owns
	std::unique_ptr<TTSNode> node_;
};

// ===========================================================================
// initialize
// ===========================================================================

TEST_F(TTSNodeTest, InitializeSucceeds)
{
	EXPECT_NO_THROW(node_->initialize());
}

TEST_F(TTSNodeTest, InitializeCreatesShmSegment)
{
	node_->initialize();
	// SHM segment should be openable after initialize().
	const int fd = shm_open(config_.shmOutput.c_str(), O_RDONLY, 0);
	EXPECT_NE(fd, -1);
	if (fd != -1) {
		close(fd);
	}
}

// ===========================================================================
// stop() is idempotent
// ===========================================================================

TEST_F(TTSNodeTest, StopBeforeRunIsIdempotent)
{
	node_->initialize();
	// stop() before run() must not crash.
	EXPECT_NO_THROW(node_->stop());
	EXPECT_NO_THROW(node_->stop());
}

// ===========================================================================
// run() exits when stop() is called from another thread
// ===========================================================================

TEST_F(TTSNodeTest, RunExitsOnStop)
{
	node_->initialize();

	std::thread runner([this]() {
		node_->run();
	});

	// Give run() a moment to enter the poll loop.
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	node_->stop();

	// run() should return within the poll timeout.
	runner.join();
	SUCCEED();
}

// ===========================================================================
// SHM segment is removed on destruction (TearDown verifies indirectly)
// ===========================================================================

TEST_F(TTSNodeTest, ShmSegmentExistsAfterInitialize)
{
	node_->initialize();
	const int fd = shm_open(config_.shmOutput.c_str(), O_RDONLY, 0);
	EXPECT_NE(fd, -1) << "SHM segment should exist after initialize()";
	if (fd != -1) close(fd);
}

// ===========================================================================
// Synthesize path via mock inferencer
// The node's run loop is not active here; we test synthesizeAndPublish via
// a white-box approach by sending ZMQ messages to a running node in a thread.
// ===========================================================================

TEST_F(TTSNodeTest, SynthesizeTriggeredBySentenceBoundary)
{
	node_->initialize();

	// Start the node's run loop in a background thread.
	std::thread runner([this]() { node_->run(); });

	// Give the poll loop time to bind.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// Publish a mock llm_response with a complete sentence via ZMQ.
	// Use a ephemeral ZMQ context for the test publisher.
	zmq::context_t testCtx{1};
	zmq::socket_t  testPub{testCtx, ZMQ_PUB};
	std::filesystem::remove(std::string("/tmp/omniedge_") + std::to_string(config_.subLlmPort));
	testPub.bind(std::string("ipc:///tmp/omniedge_") +
	             std::to_string(config_.subLlmPort));

	// Retry-publish pattern: send until the node processes the message,
	// replacing fixed sleep for ZMQ slow-joiner sync.
	nlohmann::json msg = {
		{"v",                1},
		{"type",             std::string(kZmqTopicConversationResponse)},
		{"token",            "Hello world. "},
		{"finished",         false},
		{"sentence_boundary",false},
	};
	const std::string frame = "llm_response " + msg.dump();

	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (std::chrono::steady_clock::now() < deadline) {
		testPub.send(zmq::buffer(frame), zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (mock_->synthesizeCallCount >= 1) break;
	}

	node_->stop();
	runner.join();

	// The sentence "Hello world." should have triggered one synthesize call.
	EXPECT_GE(mock_->synthesizeCallCount, 1);
	EXPECT_FALSE(mock_->lastSynthesizedText.empty());
}

TEST_F(TTSNodeTest, FlushOnFinishedTrue)
{
	node_->initialize();

	std::thread runner([this]() { node_->run(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	zmq::context_t testCtx{1};
	zmq::socket_t  testPub{testCtx, ZMQ_PUB};
	std::filesystem::remove(std::string("/tmp/omniedge_") + std::to_string(config_.subLlmPort));
	testPub.bind(std::string("ipc:///tmp/omniedge_") +
	             std::to_string(config_.subLlmPort));

	// Retry-publish pattern for ZMQ slow-joiner sync.
	nlohmann::json tok = {
		{"v",                1},
		{"type",             std::string(kZmqTopicConversationResponse)},
		{"token",            "Incomplete sentence"},
		{"finished",         false},
		{"sentence_boundary",false},
	};
	nlohmann::json fin = {
		{"v",                1},
		{"type",             std::string(kZmqTopicConversationResponse)},
		{"token",            ""},
		{"finished",         true},
		{"sentence_boundary",false},
	};

	// Send token with retry until node processes it, then send finished.
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	bool sentFinished = false;
	while (std::chrono::steady_clock::now() < deadline) {
		if (!sentFinished) {
			testPub.send(zmq::buffer("llm_response " + tok.dump()),
			             zmq::send_flags::none);
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			testPub.send(zmq::buffer("llm_response " + fin.dump()),
			             zmq::send_flags::none);
			sentFinished = true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (mock_->synthesizeCallCount >= 1) break;
		// Re-send if node hasn't processed yet
		sentFinished = false;
	}

	node_->stop();
	runner.join();

	// The incomplete sentence should be flushed on finished:true.
	EXPECT_GE(mock_->synthesizeCallCount, 1);
}

TEST_F(TTSNodeTest, FlushTtsCommandClearsBuffer)
{
	node_->initialize();
	// Reset synthesize count after warmup synthesis in initialize().
	mock_->synthesizeCallCount = 0;

	std::thread runner([this]() { node_->run(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	zmq::context_t testCtx{1};
	zmq::socket_t  llmPub{testCtx, ZMQ_PUB};
	zmq::socket_t  daemonPub{testCtx, ZMQ_PUB};

	std::filesystem::remove(std::string("/tmp/omniedge_") + std::to_string(config_.subLlmPort));
	llmPub.bind(std::string("ipc:///tmp/omniedge_") +
	            std::to_string(config_.subLlmPort));
	std::filesystem::remove(std::string("/tmp/omniedge_") + std::to_string(config_.subDaemonPort));
	daemonPub.bind(std::string("ipc:///tmp/omniedge_") +
	               std::to_string(config_.subDaemonPort));

	// Retry-publish pattern for ZMQ slow-joiner sync.
	nlohmann::json tok = {
		{"v",    1},
		{"type", std::string(kZmqTopicConversationResponse)},
		{"token","Partial"},
		{"finished",         false},
		{"sentence_boundary",false},
	};
	nlohmann::json flush = {{"action", "flush_tts"}};
	nlohmann::json fin = {
		{"v",    1},
		{"type", std::string(kZmqTopicConversationResponse)},
		{"token",""},
		{"finished", true},
		{"sentence_boundary", false},
	};

	// Send partial token, flush, then finished:true with retry loop.
	// The flush_tts should clear the buffer so finished:true produces nothing.
	for (int attempt = 0; attempt < 20; ++attempt) {
		llmPub.send(zmq::buffer("llm_response " + tok.dump()),
		            zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	// Send flush command
	for (int attempt = 0; attempt < 10; ++attempt) {
		daemonPub.send(zmq::buffer("module_status " + flush.dump()),
		               zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	// Send finished:true — buffer should have been reset by flush_tts.
	for (int attempt = 0; attempt < 10; ++attempt) {
		llmPub.send(zmq::buffer("llm_response " + fin.dump()),
		            zmq::send_flags::none);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	node_->stop();
	runner.join();

	// After flush_tts the pending "Partial" was discarded, so synthesize
	// should NOT have been called for it.
	EXPECT_EQ(mock_->synthesizeCallCount, 0);
}

