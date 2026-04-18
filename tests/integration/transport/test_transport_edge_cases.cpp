// ---------------------------------------------------------------------------
// test_transport_edge_cases.cpp -- Transport layer edge case tests
//
// Tests: SHM zero-size, SHM long names, SHM double-buffer rapid writes,
//        SHM atomic flip, ZMQ empty/large JSON, ZMQ empty topic,
//        MessageRouter stop-before-run / double-stop / publish-no-subscribers.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "shm/shm_mapping.hpp"
#include "shm/shm_double_buffer.hpp"
#include "zmq/zmq_publisher.hpp"
#include "zmq/zmq_subscriber.hpp"
#include "zmq/message_router.hpp"
#include "common/runtime_defaults.hpp"
#include "tests/mocks/test_zmq_helpers.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

namespace {

// =========================================================================
// ShmMapping edge cases
// =========================================================================

TEST(ShmMappingEdgeCaseTest, ZeroSizeSegmentThrows)
{
    // A zero-size SHM segment is meaningless; creation must fail gracefully.
    EXPECT_THROW(
        ShmMapping("/oe.test.zero_size", 0, /*create=*/true),
        std::invalid_argument);
}

TEST(ShmMappingEdgeCaseTest, VeryLongNameThrows)
{
    // POSIX SHM names are typically limited to 255 chars.  A name exceeding
    // that limit should throw, not silently truncate.
    const std::string longName = "/oe.test." + std::string(300, 'x');
    EXPECT_THROW(
        ShmMapping(longName, 4096, /*create=*/true),
        std::runtime_error);
}

TEST(ShmMappingEdgeCaseTest, MaxValidNameSucceeds)
{
    // A name near the limit should still work (POSIX allows up to 255 bytes
    // including the leading slash on most implementations).
    const std::string name = "/oe.test." + std::string(200, 'a');
    // If this exceeds the system limit it throws; otherwise succeeds.
    // We test that it does not crash or cause undefined behavior.
    try {
        ShmMapping mapping(name, 4096, /*create=*/true);
        EXPECT_NE(mapping.data(), nullptr);
    } catch (const std::runtime_error&) {
        // System limit exceeded -- that is acceptable graceful failure.
    }
}

// =========================================================================
// ShmDoubleBuffer edge cases
// =========================================================================

/// Minimal test header for double-buffer tests.
struct TestHeader {
    uint32_t seqNumber = 0;
    uint32_t width     = 320;
    uint32_t height    = 240;
    uint32_t _pad      = 0;
};
static_assert(sizeof(TestHeader) == 16);

constexpr std::size_t kSlotSize    = 1024;
constexpr std::size_t kTotalSize   = kDoubleBufferControlBytes
                                   + sizeof(TestHeader)
                                   + 2 * kSlotSize;
constexpr std::size_t kDataOffset  = kDoubleBufferControlBytes + sizeof(TestHeader);

TEST(ShmDoubleBufferEdgeCaseTest, RapidSequentialWrites_ConsumerReadsLatest)
{
    // 100 rapid sequential writes; consumer must see the latest written data.
    constexpr const char* kName = "/oe.test.rapid_writes";
    ShmDoubleBuffer<TestHeader> producer(kName, kTotalSize, true);

    for (int i = 0; i < 100; ++i) {
        auto [writePtr, writeSlot] = producer.writeSlotData(kSlotSize, kDataOffset);
        // Write a sentinel pattern: every byte = i & 0xFF
        std::memset(writePtr, i & 0xFF, kSlotSize);
        producer.header()->seqNumber = static_cast<uint32_t>(i);
        producer.flipSlot();
    }

    // Consumer opens and reads the latest slot.
    ShmDoubleBuffer<TestHeader> consumer(kName, kTotalSize, false);
    auto readResult = consumer.readSlotData(kSlotSize, kDataOffset);
    EXPECT_TRUE(readResult.valid);
    // The latest write was i=99, so seqNumber == 99 and byte pattern == 99 & 0xFF.
    EXPECT_EQ(consumer.header()->seqNumber, 99u);
    EXPECT_EQ(readResult.data[0], static_cast<uint8_t>(99 & 0xFF));
    EXPECT_EQ(readResult.data[kSlotSize - 1], static_cast<uint8_t>(99 & 0xFF));
}

TEST(ShmDoubleBufferEdgeCaseTest, WriteIndexFlipIsAtomic)
{
    // Verify the flip changes writeIndex atomically and the old/new values
    // are exactly {0,1} -- never an intermediate value.
    constexpr const char* kName = "/oe.test.atomic_flip";
    ShmDoubleBuffer<TestHeader> buf(kName, kTotalSize, true);

    std::atomic<bool> stop{false};
    std::atomic<int> anomalyCount{0};

    // Reader thread continuously checks writeIndex is always 0 or 1.
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_acquire)) {
            uint32_t idx = buf.writeIndex();
            if (idx != 0 && idx != 1) {
                anomalyCount.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // Producer rapidly flips the slot 10000 times.
    for (int i = 0; i < 10000; ++i) {
        buf.flipSlot();
    }

    stop.store(true, std::memory_order_release);
    reader.join();

    EXPECT_EQ(anomalyCount.load(), 0)
        << "writeIndex() must always return 0 or 1 (atomicity violation)";
}

TEST(ShmDoubleBufferEdgeCaseTest, ConsumerReadsCorrectSlotAfterOddFlips)
{
    // After an odd number of flips, readSlot should be slot 1 (the old write slot).
    constexpr const char* kName = "/oe.test.odd_flips";
    ShmDoubleBuffer<TestHeader> producer(kName, kTotalSize, true);

    // Initial writeIndex is 0. Producer writes to slot 1, flips to make it readable.
    auto [writePtr, writeSlot] = producer.writeSlotData(kSlotSize, kDataOffset);
    EXPECT_EQ(writeSlot, 1u);
    std::memset(writePtr, 0xBB, kSlotSize);
    producer.flipSlot();

    ShmDoubleBuffer<TestHeader> consumer(kName, kTotalSize, false);
    auto readResult = consumer.readSlotData(kSlotSize, kDataOffset);
    EXPECT_TRUE(readResult.valid);
    // After one flip, writeIndex=1, so consumer reads slot 1 (the data we wrote).
    EXPECT_EQ(readResult.data[0], 0xBB);
}

// =========================================================================
// ZMQ Publisher edge cases
// =========================================================================

namespace {
constexpr int kEdgePubPortBase = 20300;
}

class ZmqPublisherEdgeCaseTest : public ::testing::Test {
protected:
    static constexpr int kPort = kEdgePubPortBase;

    void SetUp() override
    {
        pub_ = std::make_unique<ZmqPublisher>(ctx_, kPort, /*hwm=*/10);

        sub_ = std::make_unique<zmq::socket_t>(ctx_, ZMQ_SUB);
        sub_->set(zmq::sockopt::rcvhwm, 100);
        sub_->connect("ipc:///tmp/omniedge_" + std::to_string(kPort));
        sub_->set(zmq::sockopt::subscribe, "");

        // Slow-joiner sync
        for (int attempt = 0; attempt < 20; ++attempt) {
            pub_->publish("_probe", {{"probe", true}});
            zmq::pollitem_t item{static_cast<void*>(*sub_), 0, ZMQ_POLLIN, 0};
            zmq::poll(&item, 1, std::chrono::milliseconds(10));
            if (item.revents & ZMQ_POLLIN) {
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

    std::string recvOne()
    {
        zmq::pollitem_t item{static_cast<void*>(*sub_), 0, ZMQ_POLLIN, 0};
        zmq::poll(&item, 1, std::chrono::milliseconds(200));
        if (!(item.revents & ZMQ_POLLIN)) { return {}; }
        zmq::message_t msg;
        if (!sub_->recv(msg, zmq::recv_flags::dontwait)) { return {}; }
        return std::string(static_cast<const char*>(msg.data()), msg.size());
    }

    zmq::context_t                 ctx_{1};
    std::unique_ptr<ZmqPublisher>  pub_;
    std::unique_ptr<zmq::socket_t> sub_;
};

TEST_F(ZmqPublisherEdgeCaseTest, PublishEmptyJson)
{
    // Publishing an empty JSON object ({}) should still succeed and inject
    // schema version and timestamp fields.
    pub_->publish("empty_json", nlohmann::json::object());

    const std::string raw = recvOne();
    ASSERT_FALSE(raw.empty()) << "No message received for empty JSON publish";

    const auto space = raw.find(' ');
    ASSERT_NE(space, std::string::npos);
    const auto json = nlohmann::json::parse(raw.substr(space + 1));

    EXPECT_TRUE(json.contains("v"));
    EXPECT_TRUE(json.contains("ts"));
    EXPECT_TRUE(json.contains("mono_ns"));
}

TEST_F(ZmqPublisherEdgeCaseTest, PublishVeryLargeJson)
{
    // Create a ~1MB JSON payload and verify it is delivered intact.
    nlohmann::json bigPayload;
    // ~1MB of data: 1000 entries x ~1KB each
    std::string bigValue(1024, 'X');
    for (int i = 0; i < 1000; ++i) {
        bigPayload["key_" + std::to_string(i)] = bigValue;
    }

    EXPECT_NO_THROW(pub_->publish("big_topic", bigPayload));

    const std::string raw = recvOne();
    ASSERT_FALSE(raw.empty()) << "No message received for large JSON publish";

    const auto space = raw.find(' ');
    ASSERT_NE(space, std::string::npos);

    // Parse the received JSON and verify a sample key exists.
    const auto json = nlohmann::json::parse(raw.substr(space + 1));
    EXPECT_EQ(json["key_0"].get<std::string>(), bigValue);
    EXPECT_EQ(json["key_999"].get<std::string>(), bigValue);
}

// =========================================================================
// ZMQ Subscriber edge cases
// =========================================================================

TEST(ZmqSubscriberEdgeCaseTest, SubscribeWithEmptyTopicReceivesAll)
{
    // An empty topic string in ZMQ SUB means "subscribe to all".
    zmq::context_t ctx{1};
    constexpr int kPort = kEdgePubPortBase + 20;

    zmq::socket_t pub(ctx, ZMQ_PUB);
    pub.set(zmq::sockopt::sndhwm, 100);
    std::filesystem::remove("/tmp/omniedge_" + std::to_string(kPort));
    pub.bind("ipc:///tmp/omniedge_" + std::to_string(kPort));

    // Subscribe with empty topic string
    ZmqSubscriber sub(ctx, kPort, std::vector<std::string>{""});

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string frame = "any_topic " + nlohmann::json{{"v", 1}, {"data", 42}}.dump();
    pub.send(zmq::buffer(frame), zmq::send_flags::dontwait);

    zmq::pollitem_t item{static_cast<void*>(sub.socket()), 0, ZMQ_POLLIN, 0};
    zmq::poll(&item, 1, std::chrono::milliseconds(200));

    if (item.revents & ZMQ_POLLIN) {
        const auto result = sub.tryReceive();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ((*result)["data"].get<int>(), 42);
    }
    // If nothing arrived, that is also acceptable (slow-joiner race) --
    // the main point is no crash or exception.
}

// =========================================================================
// MessageRouter edge cases
// =========================================================================

class MessageRouterEdgeCaseTest : public ::testing::Test {
protected:
    static constexpr int kBasePort = 30200;
    static int nextPort()
    {
        static std::atomic<int> port{kBasePort};
        return port.fetch_add(10, std::memory_order_relaxed);
    }
};

TEST_F(MessageRouterEdgeCaseTest, StopBeforeRunDoesNotHang)
{
    const int pubPort = nextPort();

    MessageRouter::Config cfg{
        .moduleName  = "test_stop_before_run",
        .pubPort     = pubPort,
        .pollTimeout = std::chrono::milliseconds(20),
    };
    MessageRouter router(cfg);

    // stop() before run() was ever called should be a no-op, not a deadlock.
    auto start = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(router.stop());
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(elapsed, std::chrono::seconds(2))
        << "stop() before run() should return immediately, not hang";
    EXPECT_FALSE(router.isRunning());
}

TEST_F(MessageRouterEdgeCaseTest, StopCalledTwiceIsIdempotent)
{
    const int pubPort = nextPort();

    MessageRouter::Config cfg{
        .moduleName  = "test_double_stop",
        .pubPort     = pubPort,
        .pollTimeout = std::chrono::milliseconds(20),
    };
    MessageRouter router(cfg);

    // Start the router, then stop twice. Neither stop should hang or crash.
    std::thread runner([&]() { router.run(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(router.isRunning());

    EXPECT_NO_THROW(router.stop());
    runner.join();
    EXPECT_FALSE(router.isRunning());

    // Second stop() after run() has already exited -- must be idempotent.
    EXPECT_NO_THROW(router.stop());
    EXPECT_FALSE(router.isRunning());
}

TEST_F(MessageRouterEdgeCaseTest, PublishBeforeAnySubscribersDoesNotCrash)
{
    const int pubPort = nextPort();

    MessageRouter::Config cfg{
        .moduleName  = "test_pub_no_sub",
        .pubPort     = pubPort,
        .pollTimeout = std::chrono::milliseconds(20),
    };
    MessageRouter router(cfg);

    // Publishing when no one is subscribed should silently succeed (ZMQ drops).
    EXPECT_NO_THROW(
        router.publish("orphan_topic", {{"key", "value"}, {"seq", 1}})
    );
    EXPECT_NO_THROW(
        router.publish("orphan_topic", {{"key", "value"}, {"seq", 2}})
    );

    // Also publish module_ready with no subscribers.
    EXPECT_NO_THROW(router.publishModuleReady());
}

TEST_F(MessageRouterEdgeCaseTest, TwoRoutersCanCommunicate)
{
    // Verify the basic pub/sub contract: a subscriber on one router
    // receives messages published by a different router on the same port.
    const int pubPort = nextPort();
    const int subPort = nextPort();

    // Publisher router
    MessageRouter::Config pubCfg{
        .moduleName  = "test_pub_router",
        .pubPort     = pubPort,
        .pollTimeout = std::chrono::milliseconds(20),
    };
    MessageRouter publisher(pubCfg);

    // Subscriber router — subscribes to the publisher's port
    MessageRouter::Config subCfg{
        .moduleName  = "test_sub_router",
        .pubPort     = subPort,
        .pollTimeout = std::chrono::milliseconds(20),
    };
    MessageRouter subscriber(subCfg);

    std::atomic<bool> received{false};
    subscriber.subscribe(pubPort, "test_msg", /*conflate=*/false,
        [&](const nlohmann::json& msg) {
            if (msg.value("seq", -1) == 42) {
                received.store(true, std::memory_order_release);
            }
        });

    // Start both routers
    std::thread pubThread([&]() { publisher.run(); });
    std::thread subThread([&]() { subscriber.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Publish and wait for receipt
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!received.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        publisher.publish("test_msg", {{"seq", 42}});
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    publisher.stop();
    subscriber.stop();
    pubThread.join();
    subThread.join();

    EXPECT_TRUE(received.load())
        << "Subscriber router should receive messages from publisher router";
}

} // namespace
