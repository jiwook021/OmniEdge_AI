#pragma once

#include <string_view>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "zmq/zmq_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — ZMQ PUB Socket Wrapper
//
// Each module binary instantiates exactly ONE ZmqPublisher, bound to its
// designated port from the conventions.md port table.
//
// Wire format (topic-prefixed, single ZMQ frame):
//   "<topic> <json_payload>"
//
// publish() automatically injects "v", "ts", and "mono_ns" if absent.
// ---------------------------------------------------------------------------


class ZmqPublisher {
public:
    explicit ZmqPublisher(zmq::context_t& ctx, int port, int hwm = kPublisherDataHighWaterMark);

    ZmqPublisher(const ZmqPublisher&)            = delete;
    ZmqPublisher& operator=(const ZmqPublisher&) = delete;

    /// Publish a JSON payload with the given topic prefix.
    ///
    /// Automatically injects the following fields if not already present
    /// in the payload (operates on a by-value copy — caller's original
    /// is unaffected after move):
    ///   - "v"       — schema version (kSchemaVersion)
    ///   - "ts"      — wall-clock milliseconds since epoch
    ///   - "mono_ns" — monotonic nanoseconds (for latency measurement)
    void publish(std::string_view topic, nlohmann::json payload);

private:
    zmq::socket_t socket_;
    int           port_;
};

