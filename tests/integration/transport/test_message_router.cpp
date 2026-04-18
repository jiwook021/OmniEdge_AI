#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "zmq/message_router.hpp"
#include "zmq/heartbeat_constants.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

// ---------------------------------------------------------------------------
// Unit tests for MessageRouter.
//
// Uses high port offsets (+30000) to avoid conflicts with production ports
// and other test suites.  All tests are tagged with RESOURCE_LOCK
// zmq_test_ports in CMakeLists.txt to prevent parallel port collisions.
//
// CPU-only — no LABELS gpu.
// ---------------------------------------------------------------------------


// ===========================================================================
// Test fixture
// ===========================================================================

class MessageRouterTest : public ::testing::Test {
protected:
	/// Base port offset for this test suite (well above production and other tests).
	static constexpr int kBasePort = 30100;
	/// Track next available port to avoid reuse across tests within a run.
	static int nextPort()
	{
		static std::atomic<int> port{kBasePort};
		return port.fetch_add(10, std::memory_order_relaxed);
	}
};

// ===========================================================================
// PublishReachesSubscriber
// ===========================================================================

TEST_F(MessageRouterTest, PublishReachesSubscriber)
{
	const int pubPort = nextPort();
	const int subPort = pubPort;  // subscribe to the router's own PUB port

	MessageRouter::Config cfg{
	    .moduleName  = "test_pub_sub",
	    .pubPort     = pubPort,
	    .pollTimeout = std::chrono::milliseconds(20),
	};
	MessageRouter router(cfg);

	// Track received messages.
	std::atomic<bool> received{false};
	nlohmann::json receivedPayload;

	router.subscribe(subPort, "test_topic", /*conflate=*/false,
	    [&](const nlohmann::json& msg) {
	        receivedPayload = msg;
	        received.store(true, std::memory_order_release);
	    });

	// Run the router in a background thread.
	std::thread runner([&]() { router.run(); });

	// ZMQ slow-joiner: give SUB socket time to connect to PUB.
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Publish on the router's own topic with retry.
	const auto deadline =
	    std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (!received.load(std::memory_order_acquire) &&
	       std::chrono::steady_clock::now() < deadline) {
	    router.publish("test_topic", {{"key", "value"}});
	    std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}

	router.stop();
	runner.join();

	ASSERT_TRUE(received.load()) << "Handler should have been called";
	EXPECT_EQ(receivedPayload.value("key", ""), "value");
}

// ===========================================================================
// StopInterruptsPoll
// ===========================================================================

TEST_F(MessageRouterTest, StopInterruptsPoll)
{
	const int pubPort = nextPort();

	MessageRouter::Config cfg{
	    .moduleName  = "test_stop",
	    .pubPort     = pubPort,
	    .pollTimeout = std::chrono::milliseconds(200),
	};
	MessageRouter router(cfg);

	std::thread runner([&]() { router.run(); });

	// Give run() time to enter the poll loop.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	EXPECT_TRUE(router.isRunning());

	router.stop();
	runner.join();

	EXPECT_FALSE(router.isRunning())
	    << "run() should have returned after stop()";
}

// ===========================================================================
// PublishModuleReadyContainsRequiredFields
// ===========================================================================

TEST_F(MessageRouterTest, PublishModuleReadyContainsRequiredFields)
{
	const int pubPort = nextPort();

	MessageRouter::Config cfg{
	    .moduleName  = "test_ready",
	    .pubPort     = pubPort,
	    .pollTimeout = std::chrono::milliseconds(20),
	};
	MessageRouter router(cfg);

	// Create a raw ZMQ SUB socket to capture the module_ready message.
	zmq::socket_t sub(router.context(), ZMQ_SUB);
	sub.set(zmq::sockopt::subscribe, std::string(kZmqTopicModuleReady));
	sub.set(zmq::sockopt::linger, 0);
	sub.set(zmq::sockopt::rcvtimeo, 3000);
	sub.connect("ipc:///tmp/omniedge_" + std::to_string(pubPort));

	// Give the SUB socket time to connect.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// publishModuleReady() sleeps 100ms internally for slow-joiner mitigation.
	router.publishModuleReady();

	// Receive and parse.
	nlohmann::json msg = receiveTestMessage(sub, 3000);
	ASSERT_FALSE(msg.is_discarded()) << "Should have received module_ready message";

	EXPECT_EQ(msg.value("type", ""), std::string(kZmqTopicModuleReady));
	EXPECT_EQ(msg.value("module", ""), "test_ready");
	EXPECT_TRUE(msg.contains("pid"));
	EXPECT_TRUE(msg["pid"].is_number());
}

