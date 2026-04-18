#pragma once

/*
 * PipelineEdge — directed data-flow connection between two vertices.
 *
 * Three transport types: SHM (common), ZMQ (pub/sub), WebSocket (frontend).
 * Multiple edges can share the same SHM path (fan-out).
 * Edge state is tracked separately by EdgeStateMachine.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

namespace core::graph {

enum class TransportType : std::uint8_t {
    kShm = 0,
    kZmq,
    kWebsocket
};

enum class BufferType : std::uint8_t {
    kCircularBuffer = 0,
    kDoubleBuffer,
    kFlatMapping
};

enum class ContentType : std::uint8_t {
    kBinaryJpeg = 0,
    kBinaryPcm,
    kJson
};

struct ShmAttributes {
    std::string shmPath;          /* e.g. "/oe.vid.ingest" */
    BufferType bufferType{BufferType::kCircularBuffer};
    std::size_t slotCount{4};
    std::size_t slotSizeBytes{0};
    std::size_t alignmentBytes{64};
};

struct ZmqAttributes {
    std::uint16_t port{0};
    std::string topic;
    bool conflate{false};
};

struct WebsocketAttributes {
    std::string channel;          /* e.g. "/video" */
    ContentType contentType{ContentType::kJson};
};

struct PipelineEdge {
    std::string id;               /* stable edge ID */
    std::string fromVertex;
    std::string toVertex;
    TransportType transport{TransportType::kShm};
    std::variant<ShmAttributes, ZmqAttributes, WebsocketAttributes> attributes;

    [[nodiscard]] bool operator==(const PipelineEdge& other) const {
        return id == other.id;
    }
};

/* Extract SHM path from edge attributes, empty string if not SHM transport */
[[nodiscard]] std::string shmPathOf(const PipelineEdge& edge);

} /* namespace core::graph */
