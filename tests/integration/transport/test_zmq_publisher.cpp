#include <gtest/gtest.h>

#include "zmq/zmq_publisher.hpp"
#include "common/runtime_defaults.hpp"

#include <chrono>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

// ---------------------------------------------------------------------------
// ZmqPublisher tests — CPU-only, no GPU required.
//
// Each test uses a unique port (offset from 20000 base) to avoid cross-test
// and cross-process clashes.  Tests verify:
//   - Bind succeeds on a valid port
//   - Schema fields "v", "ts", "mono_ns" are injected when absent
//   - Caller-supplied "v" is preserved (not overwritten)
//   - Published messages reach a connected SUB socket
// ---------------------------------------------------------------------------


namespace {
constexpr int kTestPortBase = 20100;  // well above all production ports
}

// ---------------------------------------------------------------------------
// Bind / construction
// ---------------------------------------------------------------------------

TEST(ZmqPublisherTest, BindSucceedsOnOpenPort)
{
	zmq::context_t ctx{1};
	EXPECT_NO_THROW({
		ZmqPublisher pub(ctx, kTestPortBase, /*hwm=*/2);
	});
}

TEST(ZmqPublisherTest, SecondBindToSamePortSilentlyRebinds)
{
	// IPC transport allows rebinding — ZMQ unlinks the socket file and creates
	// a new one.  The first publisher's socket silently goes dead (subscribers
	// that were connected to it receive nothing).  This is by-design for IPC;
	// the architectural rule "one PUB per port" is enforced by the daemon's
	// module launcher, not at the socket level.
	zmq::context_t ctx{1};
	ZmqPublisher first(ctx, kTestPortBase + 1, 2);
	EXPECT_NO_THROW(ZmqPublisher second(ctx, kTestPortBase + 1, 2));
}

// ---------------------------------------------------------------------------
// Schema injection (verified by subscribing and receiving)
// ---------------------------------------------------------------------------

class ZmqPublisherRoundTripTest : public ::testing::Test {
protected:
	static constexpr int kPort = kTestPortBase + 10;

	void SetUp() override
	{
		pub_ = std::make_unique<ZmqPublisher>(ctx_, kPort, /*hwm=*/10);

		sub_ = std::make_unique<zmq::socket_t>(ctx_, ZMQ_SUB);
		sub_->set(zmq::sockopt::rcvhwm, 100);
		sub_->connect("ipc:///tmp/omniedge_" + std::to_string(kPort));
		sub_->set(zmq::sockopt::subscribe, "");  // subscribe-all

		// Poll-based slow-joiner sync: send probe messages until subscriber
		// receives one, replacing the old fixed sleep(50ms).
		for (int attempt = 0; attempt < 20; ++attempt) {
			pub_->publish("_probe", {{"probe", true}});
			zmq::pollitem_t item{static_cast<void*>(*sub_), 0, ZMQ_POLLIN, 0};
			zmq::poll(&item, 1, std::chrono::milliseconds(10));
			if (item.revents & ZMQ_POLLIN) {
				// Drain the probe message
				zmq::message_t msg;
				(void)sub_->recv(msg, zmq::recv_flags::dontwait);
				break;
			}
		}
	}

	void TearDown() override
	{
		sub_.reset();
		pub_.reset();
	}

	/** Receive one frame with a 200 ms deadline. Returns empty string on timeout. */
	std::string recvOne()
	{
		zmq::pollitem_t item{static_cast<void*>(*sub_), 0, ZMQ_POLLIN, 0};
		zmq::poll(&item, 1, std::chrono::milliseconds(200));
		if (!(item.revents & ZMQ_POLLIN)) { return {}; }
		zmq::message_t msg;
		if (!sub_->recv(msg, zmq::recv_flags::dontwait)) { return {}; }
		return std::string(static_cast<const char*>(msg.data()), msg.size());
	}

	zmq::context_t                      ctx_{1};
	std::unique_ptr<ZmqPublisher>       pub_;
	std::unique_ptr<zmq::socket_t>      sub_;
};

TEST_F(ZmqPublisherRoundTripTest, SchemaVersionInjectedWhenAbsent)
{
	pub_->publish("test_topic", {{"data", 42}});

	const std::string raw = recvOne();
	ASSERT_FALSE(raw.empty()) << "No message received within deadline";

	// Wire format: "test_topic {json}"
	const auto space = raw.find(' ');
	ASSERT_NE(space, std::string::npos);
	const auto json = nlohmann::json::parse(raw.substr(space + 1));

	EXPECT_TRUE(json.contains("v"));
	EXPECT_EQ(json["v"].get<int>(), kSchemaVersion);
}

TEST_F(ZmqPublisherRoundTripTest, TimestampFieldsInjectedWhenAbsent)
{
	pub_->publish("test_topic", {{"data", 1}});

	const std::string raw = recvOne();
	ASSERT_FALSE(raw.empty());

	const auto space = raw.find(' ');
	ASSERT_NE(space, std::string::npos);
	const auto json = nlohmann::json::parse(raw.substr(space + 1));

	EXPECT_TRUE(json.contains("ts"))      << "\"ts\" field missing";
	EXPECT_TRUE(json.contains("mono_ns")) << "\"mono_ns\" field missing";

	// ts is Unix ms — sanity check it is > year 2020
	constexpr int64_t kYear2020Ms = 1'577'836'800'000LL;
	EXPECT_GT(json["ts"].get<int64_t>(), kYear2020Ms);
}

TEST_F(ZmqPublisherRoundTripTest, CallerSuppliedSchemaVersionPreserved)
{
	// Caller sets "v" — publisher must NOT overwrite it
	pub_->publish("test_topic", {{"v", 99}, {"data", 1}});

	const std::string raw = recvOne();
	ASSERT_FALSE(raw.empty());

	const auto space = raw.find(' ');
	ASSERT_NE(space, std::string::npos);
	const auto json = nlohmann::json::parse(raw.substr(space + 1));

	EXPECT_EQ(json["v"].get<int>(), 99);
}

TEST_F(ZmqPublisherRoundTripTest, TopicPrefixInWireFrame)
{
	pub_->publish("my_topic", {{"x", 1}});

	const std::string raw = recvOne();
	ASSERT_FALSE(raw.empty());
	EXPECT_EQ(raw.substr(0, 8), "my_topic");
}

TEST_F(ZmqPublisherRoundTripTest, DomainFieldsPassedThrough)
{
	pub_->publish("test_topic", {{"sensor_id", "cam0"}, {"seq", 7}});

	const std::string raw = recvOne();
	ASSERT_FALSE(raw.empty());

	const auto space = raw.find(' ');
	ASSERT_NE(space, std::string::npos);
	const auto json = nlohmann::json::parse(raw.substr(space + 1));

	EXPECT_EQ(json["sensor_id"].get<std::string>(), "cam0");
	EXPECT_EQ(json["seq"].get<int>(), 7);
}

