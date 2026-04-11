#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include "zmq/zmq_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — ZMQ SUB Socket Wrapper
//
// Conflation policy (from conventions.md):
//   ZMQ_CONFLATE = 1 (latest-only): video_frame, audio_chunk, blurred_frame
//   ZMQ_CONFLATE = 0 (queue all):   ui_command, llm_prompt, transcription, etc.
//
// IMPORTANT: ZMQ_CONFLATE is per-socket. When subscribing to multiple topics,
// use SEPARATE ZmqSubscriber instances — one per topic.
// ---------------------------------------------------------------------------


class ZmqSubscriber {
public:
    explicit ZmqSubscriber(zmq::context_t&          ctx,
                          int                       port,
                          std::vector<std::string>  topics,
                          bool                      conflate = false,
                          int                       hwm      = kSubscriberControlHighWaterMark);

    ZmqSubscriber(const ZmqSubscriber&)            = delete;
    ZmqSubscriber& operator=(const ZmqSubscriber&) = delete;

    [[nodiscard]] std::optional<nlohmann::json> tryReceive();

    [[nodiscard]] zmq::socket_t& socket() noexcept { return socket_; }

private:
    zmq::socket_t socket_;
    int           port_;
};

