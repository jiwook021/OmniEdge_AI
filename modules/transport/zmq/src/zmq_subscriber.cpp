#include "zmq/zmq_subscriber.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "zmq/zmq_endpoint.hpp"
#include "zmq/zmq_socket_config.hpp"

#include <format>
#include <stdexcept>


ZmqSubscriber::ZmqSubscriber(zmq::context_t&         ctx,
                              int                      port,
                              std::vector<std::string> topics,
                              bool                     conflate,
                              int                      hwm)
    : socket_(ctx, ZMQ_SUB), port_(port)
{
    configureZmqSocket(socket_, ZmqSocketRole::Subscriber, hwm);

    if (conflate) {
        socket_.set(zmq::sockopt::conflate, 1);
    }

    const std::string endpoint = zmqEndpoint(port);
    try {
        socket_.connect(endpoint);
    } catch (const zmq::error_t& e) {
        throw std::runtime_error(
            std::format("[ZmqSubscriber] connect('{}') failed: {}",
                        endpoint, e.what()));
    }

    for (const auto& topic : topics) {
        socket_.set(zmq::sockopt::subscribe, topic);
        OE_LOG_DEBUG("zmq_sub_topic_added: port={}, topic={}", port_, topic);
    }

    OE_LOG_INFO("zmq_sub_connected: port={}, conflate={}, hwm={}", port_, conflate, hwm);
}

std::optional<nlohmann::json> ZmqSubscriber::tryReceive()
{
    zmq::message_t msg;
    const auto result = socket_.recv(msg, zmq::recv_flags::dontwait);
    if (!result) {
        return std::nullopt;
    }

    const std::string_view raw(static_cast<const char*>(msg.data()), msg.size());

    const auto space = raw.find(' ');
    if (space == std::string_view::npos) {
        OE_LOG_WARN("zmq_malformed_frame: raw={}", raw);
        return std::nullopt;
    }

    const std::string_view jsonPart = raw.substr(space + 1);
    try {
        auto parsed = nlohmann::json::parse(jsonPart);
        OE_LOG_DEBUG("zmq_sub_received: port={}, topic={}, payload_size={}",
                   port_, raw.substr(0, space), jsonPart.size());
        return parsed;
    } catch (const nlohmann::json::parse_error& e) {
        OE_LOG_WARN("zmq_json_parse_error: error={}, snippet={}", e.what(), jsonPart.substr(0, 120));
        return std::nullopt;
    }
}

