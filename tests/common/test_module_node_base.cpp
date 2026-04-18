#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "common/module_node_base.hpp"
#include <tl/expected.hpp>
#include "zmq/message_router.hpp"

// ---------------------------------------------------------------------------
// Unit tests for ModuleNodeBase<Derived> CRTP lifecycle.
//
// TestNode exercises all optional hooks (onBeforeRun, onBeforeStop,
// onPublishReady), required hooks with injectable failures, and
// error propagation.
//
// Port convention: test ports start at kTestPort (+20000) to avoid
// collision with production ports.
// ---------------------------------------------------------------------------

class TestNode : public ModuleNodeBase<TestNode> {
	friend class ModuleNodeBase<TestNode>;
public:
	explicit TestNode(int port)
		: messageRouter_(MessageRouter::Config{
			.moduleName = "test_node",
			.pubPort    = port,
		})
	{}

	bool configCalled{false};
	bool inferencerCalled{false};
	bool beforeRunCalled{false};
	bool beforeStopCalled{false};
	bool publishReadyCalled{false};
	std::string configError;
	std::string inferencerError;

	[[nodiscard]] tl::expected<void, std::string> configureTransport()
	{
		configCalled = true;
		if (!configError.empty()) return tl::unexpected(configError);
		return {};
	}

	[[nodiscard]] tl::expected<void, std::string> loadInferencer()
	{
		inferencerCalled = true;
		if (!inferencerError.empty()) return tl::unexpected(inferencerError);
		return {};
	}

	void onBeforeRun()          { beforeRunCalled = true; }
	void onBeforeStop() noexcept { beforeStopCalled = true; }
	void onPublishReady()       { publishReadyCalled = true; }

	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return "test_node"; }

private:
	MessageRouter messageRouter_;
};

// ===========================================================================
// Test port base — offset +20000 from production ports
// ===========================================================================

static constexpr int kTestPort = 25000;

// ===========================================================================
// Tests
// ===========================================================================

TEST(ModuleNodeBaseTest, InitializeCallsConfigThenInferencer)
{
	TestNode node(kTestPort);
	node.initialize();
	EXPECT_TRUE(node.configCalled);
	EXPECT_TRUE(node.inferencerCalled);
	SPDLOG_DEBUG("InitializeCallsConfigThenInferencer: PASSED");
}

TEST(ModuleNodeBaseTest, ConfigFailureStopsInitAndReportsError)
{
	TestNode node(kTestPort + 1);
	node.configError = "bad config";
	try {
		node.initialize();
		FAIL() << "Expected std::runtime_error";
	} catch (const std::runtime_error& e) {
		std::string msg = e.what();
		EXPECT_NE(msg.find("test_node"), std::string::npos);
		EXPECT_NE(msg.find("configureTransport"), std::string::npos);
	}
	EXPECT_TRUE(node.configCalled);
	EXPECT_FALSE(node.inferencerCalled);
}

TEST(ModuleNodeBaseTest, InferencerFailureReportsError)
{
	TestNode node(kTestPort + 2);
	node.inferencerError = "bad model";
	try {
		node.initialize();
		FAIL() << "Expected std::runtime_error";
	} catch (const std::runtime_error& e) {
		std::string msg = e.what();
		EXPECT_NE(msg.find("test_node"), std::string::npos);
		EXPECT_NE(msg.find("loadInferencer"), std::string::npos);
	}
	EXPECT_TRUE(node.configCalled);
	EXPECT_TRUE(node.inferencerCalled);
}

TEST(ModuleNodeBaseTest, IsRunningDefaultsFalse)
{
	TestNode node(kTestPort + 4);
	EXPECT_FALSE(node.isRunning());
}

TEST(ModuleNodeBaseTest, PublishReadyOverrideIsCalled)
{
	TestNode node(kTestPort + 5);
	node.initialize();
	EXPECT_TRUE(node.publishReadyCalled);
	SPDLOG_DEBUG("PublishReadyOverrideIsCalled: PASSED");
}

TEST(ModuleNodeBaseTest, StopIsIdempotent)
{
	TestNode node(kTestPort + 6);
	node.initialize();
	node.stop();
	EXPECT_FALSE(node.isRunning());
	// Calling stop again should not crash
	node.stop();
	EXPECT_FALSE(node.isRunning());
	SPDLOG_DEBUG("StopIsIdempotent: PASSED");
}
