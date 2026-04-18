#include "graph/pipeline_graph.hpp"

#include <algorithm>

namespace core::graph {

const char* graphErrorToString(GraphError error) {
    switch (error) {
        case GraphError::kVertexNotFound:        return "VertexNotFound";
        case GraphError::kEdgeNotFound:          return "EdgeNotFound";
        case GraphError::kDuplicateVertex:       return "DuplicateVertex";
        case GraphError::kDuplicateEdge:         return "DuplicateEdge";
        case GraphError::kEndpointVertexMissing: return "EndpointVertexMissing";
        case GraphError::kVertexHasEdges:        return "VertexHasEdges";
    }
    return "Unknown";
}

/* --- Queries --- */

tl::expected<const PipelineVertex*, GraphError>
PipelineGraph::getVertex(const std::string& id) const {
    const auto it = vertices_.find(id);
    if (it == vertices_.end()) {
        return tl::unexpected(GraphError::kVertexNotFound);
    }
    return &it->second;
}

tl::expected<const PipelineEdge*, GraphError>
PipelineGraph::getEdge(const std::string& id) const {
    const auto it = edges_.find(id);
    if (it == edges_.end()) {
        return tl::unexpected(GraphError::kEdgeNotFound);
    }
    return &it->second;
}

std::vector<std::string> PipelineGraph::incomingEdges(const std::string& vertexId) const {
    const auto it = adjacencyIn_.find(vertexId);
    if (it == adjacencyIn_.end()) {
        return {};
    }
    return it->second;
}

std::vector<std::string> PipelineGraph::outgoingEdges(const std::string& vertexId) const {
    const auto it = adjacencyOut_.find(vertexId);
    if (it == adjacencyOut_.end()) {
        return {};
    }
    return it->second;
}

std::vector<std::string> PipelineGraph::edgesOnShmPath(const std::string& shmPath) const {
    std::vector<std::string> result;
    for (const auto& [edgeId, edge] : edges_) {
        if (shmPathOf(edge) == shmPath) {
            result.push_back(edgeId);
        }
    }
    return result;
}

std::vector<std::string> PipelineGraph::allVertexIds() const {
    std::vector<std::string> result;
    result.reserve(vertices_.size());
    for (const auto& [id, _] : vertices_) {
        result.push_back(id);
    }
    return result;
}

std::vector<std::string> PipelineGraph::allEdgeIds() const {
    std::vector<std::string> result;
    result.reserve(edges_.size());
    for (const auto& [id, _] : edges_) {
        result.push_back(id);
    }
    return result;
}

std::size_t PipelineGraph::vertexCount() const { return vertices_.size(); }
std::size_t PipelineGraph::edgeCount() const { return edges_.size(); }

bool PipelineGraph::hasVertex(const std::string& id) const {
    return vertices_.count(id) > 0;
}

bool PipelineGraph::hasEdge(const std::string& id) const {
    return edges_.count(id) > 0;
}

/* --- Mutations --- */

tl::expected<void, GraphError> PipelineGraph::addVertex(PipelineVertex vertex) {
    const auto& id = vertex.id;
    if (vertices_.count(id) > 0) {
        return tl::unexpected(GraphError::kDuplicateVertex);
    }
    adjacencyOut_[id];  /* ensure entry exists */
    adjacencyIn_[id];
    vertices_.emplace(id, std::move(vertex));
    return {};
}

tl::expected<void, GraphError> PipelineGraph::addEdge(PipelineEdge edge) {
    if (edges_.count(edge.id) > 0) {
        return tl::unexpected(GraphError::kDuplicateEdge);
    }
    if (vertices_.count(edge.fromVertex) == 0 || vertices_.count(edge.toVertex) == 0) {
        return tl::unexpected(GraphError::kEndpointVertexMissing);
    }

    const auto& edgeId = edge.id;
    adjacencyOut_[edge.fromVertex].push_back(edgeId);
    adjacencyIn_[edge.toVertex].push_back(edgeId);
    edges_.emplace(edgeId, std::move(edge));
    return {};
}

tl::expected<void, GraphError> PipelineGraph::removeVertex(const std::string& id) {
    if (vertices_.count(id) == 0) {
        return tl::unexpected(GraphError::kVertexNotFound);
    }

    /* Fail if vertex still has edges attached */
    const auto& outEdges = adjacencyOut_[id];
    const auto& inEdges = adjacencyIn_[id];
    if (!outEdges.empty() || !inEdges.empty()) {
        return tl::unexpected(GraphError::kVertexHasEdges);
    }

    vertices_.erase(id);
    adjacencyOut_.erase(id);
    adjacencyIn_.erase(id);
    return {};
}

tl::expected<void, GraphError> PipelineGraph::removeEdge(const std::string& id) {
    const auto it = edges_.find(id);
    if (it == edges_.end()) {
        return tl::unexpected(GraphError::kEdgeNotFound);
    }

    const auto& edge = it->second;

    /* Remove from adjacency lists */
    auto& outList = adjacencyOut_[edge.fromVertex];
    outList.erase(std::remove(outList.begin(), outList.end(), id), outList.end());

    auto& inList = adjacencyIn_[edge.toVertex];
    inList.erase(std::remove(inList.begin(), inList.end(), id), inList.end());

    edges_.erase(it);
    return {};
}

} /* namespace core::graph */
