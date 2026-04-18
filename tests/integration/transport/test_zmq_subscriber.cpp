#include <gtest/gtest.h>

#include "zmq/zmq_subscriber.hpp"

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

// ---------------------------------------------------------------------------
// ZmqSubscriber tests — CPU-only, no GPU required.
//
// Tests verify:
//   - Construction / connection succeeds
//   - tryReceive() returns nullopt when no message is available
//   - tryReceive() parses well-formed topic-prefixed JSON frames
//   - tryReceive() returns nullopt on malformed frames (no topic prefix)
//   - tryReceive() returns nullopt on invalid JSON
//   - Conflated subscriber drops intermediate messages (latest-only)
// ---------------------------------------------------------------------------


namespace {
constexpr int kTestPortBase = 20200;  // distinct range from other test files
}

// ---------------------------------------------------------------------------
// Helper: in-process PUB socket that sends raw topic-prefixed frames
// ---------------------------------------------------------------------------
class TestPublisher {
public:
	explicit TestPublisher(zmq::context_t& ctx, int port)
		: socket_(ctx, ZMQ_PUB), port_(port)
	{
		socket_.set(zmq::sockopt::sndhwm, 100);
		std::filesystem::remove("/tmp/omniedge_" + std::to_string(port));
		socket_.bind("ipc:///tmp/omniedge_" + std::to_string(port));
	}

	void send(std::string_view topic, const nlohmann::json& payload)
	{
		const std::string frame =
			std::string{topic} + " " + payload.dump();
		socket_.send(zmq::buffer(frame), zmq::send_flags::dontwait);
	}

	void sendRaw(std::string_view raw)
	{
		socket_.send(zmq::buffer(raw), zmq::send_flags::dontwait);
	}

private:
	zmq::socket_t socket_;
	int           port_;
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(ZmqSubscriberTest, ConnectSucceeds)
{
	zmq::context_t ctx{1};
	EXPECT_NO_THROW({
		ZmqSubscriber sub(ctx, kTestPortBase, {"test_topic"});
	});
}

// ---------------------------------------------------------------------------
// tryReceive when no publisher / no message
// ---------------------------------------------------------------------------

TEST(ZmqSubscriberTest, TryReceiveReturnsNulloptWhenEmpty)
{
	zmq::context_t ctx{1};
	ZmqSubscriber sub(ctx, kTestPortBase + 1, {"test_topic"});
	// No publisher connected — must return nullopt immediately
	const auto result = sub.tryReceive();
	EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Round-trip: publish → tryReceive
// ---------------------------------------------------------------------------

class ZmqSubscriberRoundTripTest : public ::testing::Test {
protected:
	static constexpr int kPort = kTestPortBase + 10;

	void SetUp() override
	{
		pub_ = std::make_unique<TestPublisher>(ctx_, kPort);
		sub_ = std::make_unique<ZmqSubscriber>(ctx_, kPort, std::vector<std::string>{"sensor"});
		// Allow slow-joiner
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}

	void TearDown() override
	{
		sub_.reset();
		pub_.reset();
	}

	/** Poll up to 200 ms for a message from the subscriber. */
	std::optional<nlohmann::json> pollReceive()
	{
		zmq::pollitem_t item{static_cast<void*>(sub_->socket()), 0, ZMQ_POLLIN, 0};
		zmq::poll(&item, 1, std::chrono::milliseconds(200));
		if (!(item.revents & ZMQ_POLLIN)) { return std::nullopt; }
		return sub_->tryReceive();
	}

	zmq::context_t                        ctx_{1};
	std::unique_ptr<TestPublisher>        pub_;
	std::unique_ptr<ZmqSubscriber>        sub_;
};

TEST_F(ZmqSubscriberRoundTripTest, WellFormedMessageParsed)
{
	pub_->send("sensor", {{"v", 1}, {"seq", 42}, {"data", "hello"}});

	const auto result = pollReceive();
	ASSERT_TRUE(result.has_value()) << "No message received within deadline";
	EXPECT_EQ((*result)["seq"].get<int>(), 42);
	EXPECT_EQ((*result)["data"].get<std::string>(), "hello");
}

TEST_F(ZmqSubscriberRoundTripTest, TopicPrefixIsStripped)
{
	// The subscriber must return only the JSON part — no topic residue
	pub_->send("sensor", {{"v", 1}, {"type", "sensor_event"}});

	const auto result = pollReceive();
	ASSERT_TRUE(result.has_value());
	// If the topic prefix were not stripped, "type" would be "sensor sensor_event"
	EXPECT_EQ((*result)["type"].get<std::string>(), "sensor_event");
	EXPECT_FALSE(result->contains("sensor"));
}

TEST_F(ZmqSubscriberRoundTripTest, MalformedFrameNoSpaceReturnsNullopt)
{
	// Frame with no space — cannot split topic from JSON
	pub_->sendRaw("nospace");

	// Poll to give the message time to arrive, then try to receive
	zmq::pollitem_t item{static_cast<void*>(sub_->socket()), 0, ZMQ_POLLIN, 0};
	zmq::poll(&item, 1, std::chrono::milliseconds(200));

	const auto result = sub_->tryReceive();
	// malformed frame → nullopt (not a throw, not a crash)
	EXPECT_FALSE(result.has_value());
}

TEST_F(ZmqSubscriberRoundTripTest, InvalidJsonReturnsNullopt)
{
	// Valid topic prefix but the payload is not JSON
	pub_->sendRaw("sensor {not valid json!!!}");

	zmq::pollitem_t item{static_cast<void*>(sub_->socket()), 0, ZMQ_POLLIN, 0};
	zmq::poll(&item, 1, std::chrono::milliseconds(200));

	const auto result = sub_->tryReceive();
	EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Conflated subscriber — latest-only delivery
// ---------------------------------------------------------------------------

TEST(ZmqSubscriberConflateTest, LatestMessageWinsUnderLoad)
{
	// Conflated subscribers only keep the most-recent message when multiple
	// are queued.  We publish several before the subscriber polls.
	zmq::context_t ctx{1};
	constexpr int kPort = kTestPortBase + 30;

	TestPublisher pub(ctx, kPort);
	// conflate=true, hwm=1 — only the most-recent frame is kept
	ZmqSubscriber sub(ctx, kPort, {"frame"}, /*conflate=*/true, /*hwm=*/1);

	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// Publish 5 messages before the subscriber polls
	for (int i = 0; i < 5; ++i) {
		pub.send("frame", {{"v", 1}, {"seq", i}});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// With ZMQ_CONFLATE the subscriber should see only the latest
	zmq::pollitem_t item{static_cast<void*>(sub.socket()), 0, ZMQ_POLLIN, 0};
	zmq::poll(&item, 1, std::chrono::milliseconds(200));
	ASSERT_TRUE(item.revents & ZMQ_POLLIN);

	const auto result = sub.tryReceive();
	ASSERT_TRUE(result.has_value());
	// Must be seq=4 (or at least not seq=0 — conflation dropped stale frames)
	EXPECT_EQ((*result)["seq"].get<int>(), 4);

	// No more messages buffered
	EXPECT_FALSE(sub.tryReceive().has_value());
}

