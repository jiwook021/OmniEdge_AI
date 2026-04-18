#pragma once

/*
 * GraphTopology — topological sort and reachability queries.
 *
 * All functions are pure, O(V+E), with no side effects.
 * topologicalSort uses Kahn's algorithm (BFS-based).
 */

#include "graph/pipeline_graph.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <tl/expected.hpp>

namespace core::graph::topology {

enum class TopologyError : std::uint8_t {
    kCycleDetected = 0,
    kVertexNotFound
};

[[nodiscard]] const char* topologyErrorToString(TopologyError error);

/* Kahn's algorithm — returns vertices in spawn (forward) order */
[[nodiscard]] tl::expected<std::vector<std::string>, TopologyError>
    topologicalSort(const PipelineGraph& graph);

/* Reverse of topo sort — teardown order */
[[nodiscard]] tl::expected<std::vector<std::string>, TopologyError>
    reverseTopologicalSort(const PipelineGraph& graph);

/* All vertices reachable backwards (upstream dependencies) */
[[nodiscard]] std::vector<std::string>
    ancestors(const PipelineGraph& graph, const std::string& vertexId);

/* All vertices reachable forward (downstream dependents) */
[[nodiscard]] std::vector<std::string>
    descendants(const PipelineGraph& graph, const std::string& vertexId);

} /* namespace core::graph::topology */
