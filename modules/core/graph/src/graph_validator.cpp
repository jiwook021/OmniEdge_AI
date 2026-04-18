#include "graph/graph_validator.hpp"
#include "graph/graph_topology.hpp"

#include <unordered_set>

namespace core::graph::validator {

const char* validationErrorToString(ValidationError error) {
    switch (error) {
        case ValidationError::kCycleDetected:       return "CycleDetected";
        case ValidationError::kDisconnectedVertex:  return "DisconnectedVertex";
        case ValidationError::kVramBudgetExceeded:  return "VramBudgetExceeded";
        case ValidationError::kDuplicateShmProducer: return "DuplicateShmProducer";
    }
    return "Unknown";
}

tl::expected<void, ValidationError>
validateDag(const PipelineGraph& graph) {
    const auto result = topology::topologicalSort(graph);
    if (!result) {
        return tl::unexpected(ValidationError::kCycleDetected);
    }
    return {};
}

tl::expected<void, ValidationError>
validateConnectedSources(const PipelineGraph& graph) {
    /* Vertices-only graphs (no edges) are valid by design.
       Orphan detection only applies once edges are introduced. */
    if (graph.edgeCount() == 0) {
        return {};
    }
    for (const auto& vid : graph.allVertexIds()) {
        const auto incoming = graph.incomingEdges(vid);
        const auto outgoing = graph.outgoingEdges(vid);

        /* A source vertex has no incoming edges — that's fine.
           A non-source vertex (has outgoing or is a sink with incoming) must have >= 1 incoming */
        if (incoming.empty() && !outgoing.empty()) {
            /* Has outgoing but no incoming — could be a legitimate source */
            continue;
        }
        if (incoming.empty() && outgoing.empty()) {
            /* Isolated vertex with no edges at all — disconnected */
            return tl::unexpected(ValidationError::kDisconnectedVertex);
        }
    }
    return {};
}

tl::expected<void, ValidationError>
validateVramBudget(const PipelineGraph& graph, std::size_t gpuBudgetMiB) {
    std::size_t total = 0;
    for (const auto& vid : graph.allVertexIds()) {
        const auto vertexResult = graph.getVertex(vid);
        if (vertexResult) {
            total += vertexResult.value()->vramBudgetMiB;
        }
    }
    if (total > gpuBudgetMiB) {
        return tl::unexpected(ValidationError::kVramBudgetExceeded);
    }
    return {};
}

tl::expected<void, ValidationError>
validateShmPathUniqueness(const PipelineGraph& graph) {
    /* Check that no two edges from DIFFERENT producers share the same SHM output path */
    std::unordered_map<std::string, std::string> pathToProducer;  /* shmPath → fromVertex */

    for (const auto& eid : graph.allEdgeIds()) {
        const auto edgeResult = graph.getEdge(eid);
        if (!edgeResult) {
            continue;
        }
        const auto& edge = *edgeResult.value();
        const auto path = shmPathOf(edge);
        if (path.empty()) {
            continue;
        }

        const auto [it, inserted] = pathToProducer.try_emplace(path, edge.fromVertex);
        if (!inserted && it->second != edge.fromVertex) {
            /* Two different producers write to the same SHM path */
            return tl::unexpected(ValidationError::kDuplicateShmProducer);
        }
    }
    return {};
}

tl::expected<void, ValidationError>
validateAll(const PipelineGraph& graph, std::size_t gpuBudgetMiB) {
    auto result = validateDag(graph);
    if (!result) { return result; }

    result = validateConnectedSources(graph);
    if (!result) { return result; }

    result = validateVramBudget(graph, gpuBudgetMiB);
    if (!result) { return result; }

    return validateShmPathUniqueness(graph);
}

} /* namespace core::graph::validator */
