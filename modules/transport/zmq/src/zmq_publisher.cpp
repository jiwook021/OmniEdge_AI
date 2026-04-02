#include "zmq/zmq_publisher.hpp"

#include "common/runtime_defaults.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/time_utils.hpp"
#include "zmq/zmq_endpoint.hpp"

#include <chrono>
#include <cstdint>
#include <format>
#include <stdexcept>


namespace {

[[nodiscard]] int64_t unixMs() noexcept
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // anonymous namespace

ZmqPublisher::ZmqPublisher(zmq::context_t& ctx, int port, int hwm)
    : socket_(ctx, ZMQ_PUB), port_(port)
{
    socket_.set(zmq::sockopt::sndhwm, hwm);
    socket_.set(zmq::sockopt::linger, 0);

    const std::string endpoint = zmqEndpoint(port);
    try {
        socket_.bind(endpoint);
    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::format("[ZmqPublisher] bind('{}') failed: {}", endpoint, e.what()));
    }

    OE_LOG_INFO("zmq_pub_bound: port={}, hwm={}", port_, hwm);
}

void ZmqPublisher::publish(std::string_view topic, nlohmann::json payload)
{
    OE_LOG_DEBUG("zmq_pub_send: port={}, topic={}, payload_keys={}",
               port_, topic, payload.size());
    // Inject mandatory schema fields if the caller omitted them
    if (!payload.contains("v")) {
        payload["v"] = kSchemaVersion;
    }
    if (!payload.contains("ts")) {
        payload["ts"] = unixMs();
    }
    if (!payload.contains("mono_ns")) {
        payload["mono_ns"] = steadyClockNanoseconds();
    }

    const std::string frame = std::format("{} {}", topic, payload.dump());

    try {
        socket_.send(zmq::buffer(frame), zmq::send_flags::dontwait);
    } catch (const zmq::error_t& e) {
        if (e.num() != EAGAIN) {
            OE_LOG_WARN("zmq_send_error: topic={}, error={}", topic, e.what());
        }
    }
}

