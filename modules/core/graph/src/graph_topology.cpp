#include "graph/graph_topology.hpp"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace core::graph::topology {

const char* topologyErrorToString(TopologyError error) {
    switch (error) {
        case TopologyError::kCycleDetected: return "CycleDetected";
        case TopologyError::kVertexNotFound: return "VertexNotFound";
    }
    return "Unknown";
}

tl::expected<std::vector<std::string>, TopologyError>
topologicalSort(const PipelineGraph& graph) {
    const auto vertexIds = graph.allVertexIds();

    /* Build in-degree map */
    std::unordered_map<std::string, std::size_t> inDegree;
    for (const auto& vid : vertexIds) {
        inDegree[vid] = graph.incomingEdges(vid).size();
    }

    /* Seed queue with zero in-degree vertices */
    std::queue<std::string> ready;
    for (const auto& [vid, degree] : inDegree) {
        if (degree == 0) {
            ready.push(vid);
        }
    }

    /* Kahn's algorithm */
    std::vector<std::string> sorted;
    sorted.reserve(vertexIds.size());

    while (!ready.empty()) {
        const auto current = ready.front();
        ready.pop();
        sorted.push_back(current);

        for (const auto& edgeId : graph.outgoingEdges(current)) {
            const auto edgeResult = graph.getEdge(edgeId);
            if (!edgeResult) {
                continue;
            }
            const auto& toVertex = edgeResult.value()->toVertex;
            auto& deg = inDegree[toVertex];
            if (deg > 0) {
                --deg;
            }
            if (deg == 0) {
                ready.push(toVertex);
            }
        }
    }

    if (sorted.size() != vertexIds.size()) {
        return tl::unexpected(TopologyError::kCycleDetected);
    }
    return sorted;
}

tl::expected<std::vector<std::string>, TopologyError>
reverseTopologicalSort(const PipelineGraph& graph) {
    auto result = topologicalSort(graph);
    if (!result) {
        return result;
    }
    std::reverse(result->begin(), result->end());
    return result;
}

std::vector<std::string>
ancestors(const PipelineGraph& graph, const std::string& vertexId) {
    std::vector<std::string> result;
    std::unordered_set<std::string> visited;
    std::queue<std::string> work;

    /* Seed with immediate predecessors */
    for (const auto& edgeId : graph.incomingEdges(vertexId)) {
        const auto edgeResult = graph.getEdge(edgeId);
        if (!edgeResult) {
            continue;
        }
        const auto& from = edgeResult.value()->fromVertex;
        if (visited.insert(from).second) {
            work.push(from);
        }
    }

    /* BFS backwards */
    while (!work.empty()) {
        const auto current = work.front();
        work.pop();
        result.push_back(current);

        for (const auto& edgeId : graph.incomingEdges(current)) {
            const auto edgeResult = graph.getEdge(edgeId);
            if (!edgeResult) {
                continue;
            }
            const auto& from = edgeResult.value()->fromVertex;
            if (visited.insert(from).second) {
                work.push(from);
            }
        }
    }
    return result;
}

std::vector<std::string>
descendants(const PipelineGraph& graph, const std::string& vertexId) {
    std::vector<std::string> result;
    std::unordered_set<std::string> visited;
    std::queue<std::string> work;

    /* Seed with immediate successors */
    for (const auto& edgeId : graph.outgoingEdges(vertexId)) {
        const auto edgeResult = graph.getEdge(edgeId);
        if (!edgeResult) {
            continue;
        }
        const auto& to = edgeResult.value()->toVertex;
        if (visited.insert(to).second) {
            work.push(to);
        }
    }

    /* BFS forward */
    while (!work.empty()) {
        const auto current = work.front();
        work.pop();
        result.push_back(current);

        for (const auto& edgeId : graph.outgoingEdges(current)) {
            const auto edgeResult = graph.getEdge(edgeId);
            if (!edgeResult) {
                continue;
            }
            const auto& to = edgeResult.value()->toVertex;
            if (visited.insert(to).second) {
                work.push(to);
            }
        }
    }
    return result;
}

} /* namespace core::graph::topology */
