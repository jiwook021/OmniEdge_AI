#pragma once

/*
 * GraphSerialization — JSON serialization of pipeline graph state.
 *
 * Used by the GET /graph introspection endpoint.
 * Includes schema version for future compatibility.
 */

#include "graph/pipeline_graph.hpp"
#include "graph/edge_state_machine.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace core::graph::serialization {

namespace defaults {
    constexpr std::uint32_t kSchemaVersion = 1;
} /* namespace defaults */

/* Serialize graph + edge states to JSON string */
[[nodiscard]] std::string serializeToJson(
    const PipelineGraph& graph,
    const std::unordered_map<std::string, EdgeState>& edgeStates);

} /* namespace core::graph::serialization */
