#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "zmq/heartbeat_constants.hpp"
#include "zmq/zmq_publisher.hpp"
#include "zmq/zmq_subscriber.hpp"
#include "zmq/zmq_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — MessageRouter
//
// Consolidates all ZMQ networking into a single class per module process.
// Each module creates one MessageRouter that owns:
//   - A ZMQ context (one per process)
//   - A PUB socket bound to the module's assigned port
//   - Multiple SUB sockets connected to upstream publishers
//   - A PAIR interrupt socket for clean stop() from signal handlers
//   - A blocking poll loop that dispatches messages to registered handlers
// ---------------------------------------------------------------------------


class MessageRouter {
public:
    struct Config {
        std::string moduleName;
        int         pubPort;
        int         pubHwm      = kPublisherControlHighWaterMark;
        std::chrono::milliseconds pollTimeout{kPollTimeoutMs};
    };

    explicit MessageRouter(const Config& config);

    MessageRouter(const MessageRouter&)            = delete;
    MessageRouter& operator=(const MessageRouter&) = delete;

    ~MessageRouter();

    void subscribe(int port, std::string_view topic, bool conflate,
                   std::function<void(const nlohmann::json&)> handler);

    void publish(std::string_view topic, nlohmann::json payload);
    void publishModuleReady();

    void setOnPollCallback(std::function<bool()> cb);

    void run();
    void stop() noexcept;

    [[nodiscard]] bool isRunning() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    [[nodiscard]] zmq::context_t& context() noexcept { return zmqContext_; }
    [[nodiscard]] ZmqPublisher& publisher() noexcept { return *publisher_; }
    [[nodiscard]] const std::string& moduleName() const noexcept { return config_.moduleName; }

private:
    Config            config_;
    std::atomic<bool> running_{false};

    zmq::context_t zmqContext_{1};

    std::unique_ptr<ZmqPublisher> publisher_;

    struct SubscriptionEntry {
        std::unique_ptr<ZmqSubscriber> subscriber;
        std::function<void(const nlohmann::json&)> handler;
        std::string                    topic;
    };
    std::vector<SubscriptionEntry> subscriptions_;

    std::function<bool()> onPollCallback_;

    // Heartbeat state — tracks liveness for the daemon watchdog.
    std::chrono::steady_clock::time_point startTime_{};
    std::chrono::steady_clock::time_point lastHeartbeat_{};
    std::chrono::steady_clock::time_point lastWorkTs_{};

    zmq::socket_t interruptSendSocket_{zmqContext_, ZMQ_PAIR};
    zmq::socket_t interruptReceiveSocket_{zmqContext_, ZMQ_PAIR};
};

