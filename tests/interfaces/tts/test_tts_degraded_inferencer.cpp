#include <gtest/gtest.h>

#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <memory>
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
// TTSNode degraded-inferencer tests — CPU-only.
//
// Verifies that TTSNode handles inferencer failures gracefully: errors on
// synthesize(), empty audio return, and oversized audio return.  The node
// must not crash and should continue processing subsequent sentences.
//
// Port offset: +20000 (avoids collisions with production).
// Ports start at 25680 to avoid collisions with test_tts_node.cpp (25561-25571).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// DegradedMockInferencer — extends MinimalMockInferencer with conditional failures.
//
// - failOnCallNumber:       return error on Nth synthesize() call (0 = never)
// - emptyAudioOnCallNumber: return empty audio on Nth call (0 = never)
// - oversizedSamples:       return this many samples if > 0 (overrides normal)
// ---------------------------------------------------------------------------

class DegradedMockInferencer : public MinimalMockInferencer {
public:
	[[nodiscard]] tl::expected<std::vector<float>, std::string> synthesize(
		std::string_view text,
		std::string_view voiceName,
		float            speed) override
	{
		++callCounter_;

		if (failOnCallNumber > 0 && callCounter_ == failOnCallNumber) {
			++synthesizeCallCount;
			lastSynthesizedText = std::string(text);
			return tl::unexpected(std::string("mock inferencer error on call ")
			                  + std::to_string(callCounter_));
		}

		if (emptyAudioOnCallNumber > 0 && callCounter_ == emptyAudioOnCallNumber) {
			++synthesizeCallCount;
			lastSynthesizedText = std::string(text);
			return std::vector<float>{};  // empty audio
		}

		if (oversizedSamples > 0) {
			++synthesizeCallCount;
			lastSynthesizedText = std::string(text);
			return std::vector<float>(static_cast<std::size_t>(oversizedSamples), 0.0f);
		}

		// Delegate to base class for normal operation
		return MinimalMockInferencer::synthesize(text, voiceName, speed);
	}

	/// Return an error on this call number (1-based). 0 = never fail.
	int failOnCallNumber{0};

	/// Return empty audio on this call number (1-based). 0 = never.
	int emptyAudioOnCallNumber{0};

	/// Return this many samples if > 0 (overrides normal 24000-sample silence).
	int oversizedSamples{0};

private:
	int callCounter_{0};
};

// ===========================================================================
// Test fixture — same pattern as TTSNodeTest in test_tts_node.cpp
// ===========================================================================

class TTSDegradedTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		// Use test ports — offset to avoid collisions with test_tts_node.cpp
		config_.subLlmPort    = 25681;
		config_.subDaemonPort = 25691;
		config_.pubPort       = 25685;
		config_.shmOutput     = "/oe.aud.tts.degraded.test";
		config_.pollTimeout   = std::chrono::milliseconds(10);

		degradedMock_ = new DegradedMockInferencer();
		node_ = std::make_unique<TTSNode>(
			config_,
			std::unique_ptr<TTSInferencer>(degradedMock_));
	}

	void TearDown() override
	{
		shm_unlink(config_.shmOutput.c_str());
	}

	/// Send llm_response tokens through the LLM PUB socket.
	/// Returns the ZMQ context (caller must keep alive until done).
	struct TestPublisher {
		zmq::context_t ctx{1};
		zmq::socket_t  pub{ctx, ZMQ_PUB};
	};

	/// Create a PUB socket bound to the LLM port.
	[[nodiscard]] std::unique_ptr<TestPublisher> makeLlmPublisher()
	{
		auto tp = std::make_unique<TestPublisher>();
		std::filesystem::remove(std::string("/tmp/omniedge_") + std::to_string(config_.subLlmPort));
		tp->pub.bind(std::string("ipc:///tmp/omniedge_") +
		             std::to_string(config_.subLlmPort));
		return tp;
	}

	/// Send a sentence token + finished:true flush via the given publisher.
	void sendSentence(zmq::socket_t& pub, const std::string& sentence, bool finished)
	{
		nlohmann::json tok = {
			{"v",                 1},
			{"type",              std::string(kZmqTopicConversationResponse)},
			{"token",             sentence},
			{"finished",          false},
			{"sentence_boundary", false},
		};
		pub.send(zmq::buffer("llm_response " + tok.dump()), zmq::send_flags::none);

		if (finished) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			nlohmann::json fin = {
				{"v",                 1},
				{"type",              std::string(kZmqTopicConversationResponse)},
				{"token",             ""},
				{"finished",          true},
				{"sentence_boundary", false},
			};
			pub.send(zmq::buffer("llm_response " + fin.dump()), zmq::send_flags::none);
		}
	}

	TTSNode::Config              config_;
	DegradedMockInferencer*         degradedMock_{nullptr};  // borrowed; node_ owns
	std::unique_ptr<TTSNode>     node_;
};

// ===========================================================================
// TTSDegraded_ErrorOnFirstSentence_ContinuesRunning
//
// Bug caught: if the first synthesize() call throws or returns an error,
// the node exits the run loop or enters a broken state, dropping all
// subsequent sentences.
// ===========================================================================

TEST_F(TTSDegradedTest, ErrorOnFirstSentence_ContinuesRunning)
{
	degradedMock_->failOnCallNumber = 1;  // fail on first synthesize call

	node_->initialize();

	std::thread runner([this]() { node_->run(); });

	auto llmPub = makeLlmPublisher();

	// Phase 1: Retry-send the first sentence until the inferencer is called (fails)
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (std::chrono::steady_clock::now() < deadline) {
		sendSentence(llmPub->pub, "First sentence fails. ", true);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		if (degradedMock_->synthesizeCallCount >= 1) break;
	}

	// Phase 2: Retry-send the second sentence until the inferencer is called (succeeds)
	const auto deadline2 =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (std::chrono::steady_clock::now() < deadline2) {
		sendSentence(llmPub->pub, "Second sentence succeeds. ", true);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		if (degradedMock_->synthesizeCallCount >= 2) break;
	}

	node_->stop();
	runner.join();

	// The node must have survived the first error and processed the second
	EXPECT_GE(degradedMock_->synthesizeCallCount, 2)
		<< "Node should continue processing after first synthesize() error";
}

// ===========================================================================
// TTSDegraded_EmptyAudio_NodeSurvives
//
// Bug caught: synthesize() returns an empty vector, and the node tries to
// write zero bytes to SHM or divides by zero when calculating duration,
// causing a crash or hang.
// ===========================================================================

TEST_F(TTSDegradedTest, EmptyAudio_NodeSurvives)
{
	degradedMock_->emptyAudioOnCallNumber = 1;  // first call returns empty audio

	node_->initialize();

	std::thread runner([this]() { node_->run(); });

	auto llmPub = makeLlmPublisher();

	// Retry-publish: send sentence until the node processes it
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (std::chrono::steady_clock::now() < deadline) {
		sendSentence(llmPub->pub, "This returns empty audio. ", true);
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		if (degradedMock_->synthesizeCallCount >= 1) break;
	}

	// Give the node time to process the empty audio result
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Node must still be alive — stop cleanly
	EXPECT_NO_THROW(node_->stop());
	runner.join();

	EXPECT_GE(degradedMock_->synthesizeCallCount, 1)
		<< "Inferencer should have been called at least once";
}

// ===========================================================================
// TTSDegraded_InferencerErrorRecovery
//
// Bug caught: after a inferencer error, the node's internal state is corrupted
// (e.g. sentence buffer not cleared), and all subsequent synthesize() calls
// are skipped or produce incorrect output.
// ===========================================================================

TEST_F(TTSDegradedTest, InferencerErrorRecovery)
{
	degradedMock_->failOnCallNumber = 1;  // first call fails, subsequent succeed

	node_->initialize();

	std::thread runner([this]() { node_->run(); });

	auto llmPub = makeLlmPublisher();

	// Send two separate sentences with retry
	const auto deadline =
		std::chrono::steady_clock::now() + std::chrono::seconds(3);
	int phase = 0;  // 0 = send first, 1 = wait for first + send second

	while (std::chrono::steady_clock::now() < deadline) {
		if (phase == 0) {
			sendSentence(llmPub->pub, "Fails on inferencer. ", true);
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			if (degradedMock_->synthesizeCallCount >= 1) {
				phase = 1;
			}
		} else {
			sendSentence(llmPub->pub, "Recovers and succeeds. ", true);
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			if (degradedMock_->synthesizeCallCount >= 2) break;
		}
	}

	node_->stop();
	runner.join();

	// Inferencer must have been called at least twice: once (fail) + once (success)
	EXPECT_GE(degradedMock_->synthesizeCallCount, 2)
		<< "After first inferencer error, node should recover and call inferencer "
		   "again for the next sentence";
	EXPECT_FALSE(degradedMock_->lastSynthesizedText.empty())
		<< "Last synthesized text should be the second (successful) sentence";
	EXPECT_NE(degradedMock_->lastSynthesizedText.find("Recovers"), std::string::npos)
		<< "Last synthesized text should contain 'Recovers': got '"
		<< degradedMock_->lastSynthesizedText << "'";
}

