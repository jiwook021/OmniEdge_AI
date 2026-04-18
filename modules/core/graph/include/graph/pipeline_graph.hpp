#pragma once

/*
 * PipelineGraph — DAG container for pipeline vertices and edges.
 *
 * Pure data structure with query and mutation methods.
 * Does NOT track edge runtime state (that's EdgeStateMachine's job).
 * Does NOT perform I/O, syscalls, or threading.
 */

#include "graph/pipeline_vertex.hpp"
#include "graph/pipeline_edge.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <tl/expected.hpp>

namespace core::graph {

enum class GraphError : std::uint8_t {
    kVertexNotFound = 0,
    kEdgeNotFound,
    kDuplicateVertex,
    kDuplicateEdge,
    kEndpointVertexMissing,
    kVertexHasEdges
};

[[nodiscard]] const char* graphErrorToString(GraphError error);

class PipelineGraph {
public:
    /* --- Queries (all const) --- */

    [[nodiscard]] tl::expected<const PipelineVertex*, GraphError>
        getVertex(const std::string& id) const;

    [[nodiscard]] tl::expected<const PipelineEdge*, GraphError>
        getEdge(const std::string& id) const;

    [[nodiscard]] std::vector<std::string> incomingEdges(const std::string& vertexId) const;
    [[nodiscard]] std::vector<std::string> outgoingEdges(const std::string& vertexId) const;
    [[nodiscard]] std::vector<std::string> edgesOnShmPath(const std::string& shmPath) const;

    [[nodiscard]] std::vector<std::string> allVertexIds() const;
    [[nodiscard]] std::vector<std::string> allEdgeIds() const;

    [[nodiscard]] std::size_t vertexCount() const;
    [[nodiscard]] std::size_t edgeCount() const;

    [[nodiscard]] bool hasVertex(const std::string& id) const;
    [[nodiscard]] bool hasEdge(const std::string& id) const;

    /* --- Mutations --- */

    [[nodiscard]] tl::expected<void, GraphError> addVertex(PipelineVertex vertex);
    [[nodiscard]] tl::expected<void, GraphError> addEdge(PipelineEdge edge);
    [[nodiscard]] tl::expected<void, GraphError> removeVertex(const std::string& id);
    [[nodiscard]] tl::expected<void, GraphError> removeEdge(const std::string& id);

private:
    std::unordered_map<std::string, PipelineVertex> vertices_;
    std::unordered_map<std::string, PipelineEdge> edges_;
    std::unordered_map<std::string, std::vector<std::string>> adjacencyOut_;
    std::unordered_map<std::string, std::vector<std::string>> adjacencyIn_;
};

} /* namespace core::graph */
