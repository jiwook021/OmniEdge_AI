#include "graph/graph_serialization.hpp"

#include <nlohmann/json.hpp>

namespace core::graph::serialization {

static nlohmann::json serializeVertex(const PipelineVertex& v) {
    nlohmann::json j;
    j["id"] = v.id;
    j["binary_name"] = v.binaryName;
    j["display_label"] = v.displayLabel;
    j["vram_budget_mib"] = v.vramBudgetMiB;
    j["spawn_args"] = v.spawnArgs;
    if (v.zmqPubPort.has_value()) {
        j["zmq_pub_port"] = v.zmqPubPort.value();
    }
    return j;
}

static const char* transportTypeToString(TransportType t) {
    switch (t) {
        case TransportType::kShm:       return "shm";
        case TransportType::kZmq:       return "zmq";
        case TransportType::kWebsocket: return "websocket";
    }
    return "unknown";
}

static nlohmann::json serializeAttributes(const PipelineEdge& edge) {
    nlohmann::json j;
    switch (edge.transport) {
        case TransportType::kShm: {
            const auto& attrs = std::get<ShmAttributes>(edge.attributes);
            j["shm_path"] = attrs.shmPath;
            j["slot_count"] = attrs.slotCount;
            j["slot_size_bytes"] = attrs.slotSizeBytes;
            j["alignment_bytes"] = attrs.alignmentBytes;
            break;
        }
        case TransportType::kZmq: {
            const auto& attrs = std::get<ZmqAttributes>(edge.attributes);
            j["port"] = attrs.port;
            j["topic"] = attrs.topic;
            j["conflate"] = attrs.conflate;
            break;
        }
        case TransportType::kWebsocket: {
            const auto& attrs = std::get<WebsocketAttributes>(edge.attributes);
            j["channel"] = attrs.channel;
            break;
        }
    }
    return j;
}

static nlohmann::json serializeEdge(
    const PipelineEdge& edge,
    const std::unordered_map<std::string, EdgeState>& edgeStates) {

    nlohmann::json j;
    j["id"] = edge.id;
    j["from"] = edge.fromVertex;
    j["to"] = edge.toVertex;
    j["transport"] = transportTypeToString(edge.transport);
    j["attributes"] = serializeAttributes(edge);

    const auto stateIt = edgeStates.find(edge.id);
    if (stateIt != edgeStates.end()) {
        j["state"] = edgeStateToString(stateIt->second);
    } else {
        j["state"] = "unknown";
    }
    return j;
}

std::string serializeToJson(
    const PipelineGraph& graph,
    const std::unordered_map<std::string, EdgeState>& edgeStates) {

    nlohmann::json root;
    root["schema_version"] = defaults::kSchemaVersion;

    auto& vertices = root["vertices"];
    vertices = nlohmann::json::array();
    for (const auto& vid : graph.allVertexIds()) {
        const auto vertexResult = graph.getVertex(vid);
        if (vertexResult) {
            vertices.push_back(serializeVertex(*vertexResult.value()));
        }
    }

    auto& edges = root["edges"];
    edges = nlohmann::json::array();
    for (const auto& eid : graph.allEdgeIds()) {
        const auto edgeResult = graph.getEdge(eid);
        if (edgeResult) {
            edges.push_back(serializeEdge(*edgeResult.value(), edgeStates));
        }
    }

    root["vertex_count"] = graph.vertexCount();
    root["edge_count"] = graph.edgeCount();

    return root.dump(2);
}

} /* namespace core::graph::serialization */
