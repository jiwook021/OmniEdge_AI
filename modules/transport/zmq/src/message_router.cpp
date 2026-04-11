#include "zmq/message_router.hpp"

#include <chrono>
#include <format>
#include <stdexcept>
#include <thread>

#include <unistd.h>

#include "common/oe_tracy.hpp"
#include "common/oe_logger.hpp"
#include "zmq/zmq_endpoint.hpp"


// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

MessageRouter::MessageRouter(const Config& config)
    : config_(config)
{
    OeLogger::instance().setModule(config_.moduleName);

    cleanStaleEndpoint(config_.pubPort);
    publisher_ = std::make_unique<ZmqPublisher>(
        zmqContext_, config_.pubPort, config_.pubHwm);

    const std::string interruptAddress =
        std::format("inproc://{}_router_interrupt", config_.moduleName);

    constexpr int kLingerMs = 0;
    interruptSendSocket_.set(zmq::sockopt::linger, kLingerMs);
    interruptReceiveSocket_.set(zmq::sockopt::linger, kLingerMs);

    interruptSendSocket_.bind(interruptAddress);
    interruptReceiveSocket_.connect(interruptAddress);

    OE_LOG_INFO("message_router_created: module={}, pub_port={}",
              config_.moduleName, config_.pubPort);
}

MessageRouter::~MessageRouter()
{
    if (running_.load(std::memory_order_acquire)) {
        stop();
    }
}

// ---------------------------------------------------------------------------
// subscribe()
// ---------------------------------------------------------------------------

void MessageRouter::subscribe(int port, std::string_view topic, bool conflate,
                              std::function<void(const nlohmann::json&)> handler)
{
    auto subscriber = std::make_unique<ZmqSubscriber>(
        zmqContext_, port,
        std::vector<std::string>{std::string(topic)},
        conflate);

    subscriptions_.push_back({
        std::move(subscriber),
        std::move(handler),
        std::string(topic),
    });

    OE_LOG_INFO("message_router_subscribed: port={}, topic={}, conflate={}",
              port, topic, conflate);
}

// ---------------------------------------------------------------------------
// publish()
// ---------------------------------------------------------------------------

void MessageRouter::publish(std::string_view topic, nlohmann::json payload)
{
    publisher_->publish(topic, std::move(payload));
}

// ---------------------------------------------------------------------------
// publishModuleReady()
// ---------------------------------------------------------------------------

void MessageRouter::publishModuleReady()
{
    // ZMQ slow-joiner mitigation: SUB sockets that connect() after a PUB
    // socket bind() will silently drop messages published before the TCP
    // handshake completes.  There is no ZMQ API to detect when a subscriber
    // has fully joined, so we sleep briefly to give the daemon's SUB socket
    // time to connect and subscribe.  100 ms is sufficient on localhost; a
    // more robust solution would use XPUB welcome messages or an explicit
    // handshake topic, but the added complexity is not warranted for the
    // single-machine deployment that OmniEdge_AI targets.
    std::this_thread::sleep_for(std::chrono::milliseconds{kSlowJoinerDelayMs});

    publisher_->publish("module_ready", {
        {"v",      kSchemaVersion},
        {"type",   "module_ready"},
        {"module", config_.moduleName},
        {"pid",    static_cast<int>(::getpid())},
    });

    OE_LOG_INFO("module_ready_published: module={}", config_.moduleName);
}

// ---------------------------------------------------------------------------
// setOnPollCallback()
// ---------------------------------------------------------------------------

void MessageRouter::setOnPollCallback(std::function<bool()> cb)
{
    onPollCallback_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// run() — blocking poll loop with handler dispatch
// ---------------------------------------------------------------------------

void MessageRouter::run()
{
    running_.store(true, std::memory_order_release);

    startTime_     = std::chrono::steady_clock::now();
    lastHeartbeat_ = startTime_;
    lastWorkTs_    = startTime_;

    // Build pollitem_t vector: [sub0, sub1, ..., subN, interrupt]
    std::vector<zmq::pollitem_t> pollItems;
    pollItems.reserve(subscriptions_.size() + 1);

    for (auto& entry : subscriptions_) {
        pollItems.push_back(
            {static_cast<void*>(entry.subscriber->socket()), 0, ZMQ_POLLIN, 0});
    }
    pollItems.push_back(
        {static_cast<void*>(interruptReceiveSocket_), 0, ZMQ_POLLIN, 0});

    const std::size_t interruptIndex = pollItems.size() - 1;

    OE_LOG_INFO("message_router_run_start: module={}, subscriptions={}",
              config_.moduleName, subscriptions_.size());

    // Re-publish module_ready for late-connecting subscribers.  The daemon's
    // own MessageRouter subscribes before modules are spawned, but the launchAll
    // readiness socket may miss it.  This costs ~100ms (the sleep in
    // publishModuleReady) but ensures the message is sent after the poll items
    // are constructed and the PUB socket is fully active.
    publishModuleReady();

    while (running_.load(std::memory_order_acquire)) {
        OE_FRAME_MARK;
        try {
            zmq::poll(pollItems, config_.pollTimeout);
        } catch (const zmq::error_t& e) {
            if (e.num() == EINTR) { break; }
            throw;
        }

        if (onPollCallback_) {
            try {
                if (!onPollCallback_()) {
                    break;
                }
            } catch (const std::exception& e) {
                OE_LOG_ERROR("poll_callback_exception: module={}, error={}",
                             config_.moduleName, e.what());
                break;
            }
        }

        // Publish heartbeat if the interval has elapsed.
        {
            auto now = std::chrono::steady_clock::now();
            if (now - lastHeartbeat_ >= std::chrono::milliseconds{kIntervalMs}) {
                const auto uptimeS =
                    std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
                const auto workAgoMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - lastWorkTs_).count();

                publisher_->publish("heartbeat", {
                    {"type",             "heartbeat"},
                    {"module",           config_.moduleName},
                    {"pid",              static_cast<int>(::getpid())},
                    {"uptime_s",         uptimeS},
                    {"last_work_ms_ago", workAgoMs},
                });
                lastHeartbeat_ = now;
            }
        }

        // Check interrupt first — stop() was called.
        if (pollItems[interruptIndex].revents & ZMQ_POLLIN) {
            zmq::message_t discardFrame;
            (void)interruptReceiveSocket_.recv(discardFrame, zmq::recv_flags::dontwait);
            break;
        }

        // Dispatch messages to registered handlers.
        for (std::size_t i = 0; i < subscriptions_.size(); ++i) {
            if (!(pollItems[i].revents & ZMQ_POLLIN)) {
                continue;
            }

            while (true) {
                auto parsedMessage = subscriptions_[i].subscriber->tryReceive();
                if (!parsedMessage.has_value()) {
                    break;
                }

                try {
                    subscriptions_[i].handler(*parsedMessage);
                    lastWorkTs_ = std::chrono::steady_clock::now();
                } catch (const std::exception& e) {
                    OE_LOG_WARN("message_handler_exception: topic={}, error={}",
                              subscriptions_[i].topic, e.what());
                }
            }
        }
    }

    running_.store(false, std::memory_order_release);
    OE_LOG_INFO("message_router_run_stopped: module={}", config_.moduleName);
}

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

void MessageRouter::stop() noexcept
{
    running_.store(false, std::memory_order_release);
    try {
        interruptSendSocket_.send(zmq::str_buffer("stop"),
                                  zmq::send_flags::dontwait);
    } catch (...) {}
}

