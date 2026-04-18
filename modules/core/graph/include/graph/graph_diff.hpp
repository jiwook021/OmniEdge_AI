#pragma once

/*
 * GraphDiff — compute the delta between two pipeline graphs.
 *
 * Used to determine what to stop, start, and clean up during a profile switch.
 * shmPathsToUnlink contains paths with zero remaining consumers in the target.
 */

#include "graph/pipeline_graph.hpp"

#include <string>
#include <vector>

namespace core::graph {

struct GraphDiff {
    std::vector<std::string> edgesToRemove;
    std::vector<std::string> edgesToAdd;
    std::vector<std::string> edgesKept;
    std::vector<std::string> verticesToEvict;
    std::vector<std::string> verticesToSpawn;
    std::vector<std::string> verticesKept;
    std::vector<std::string> shmPathsToUnlink;
};

/* Deterministic, pure diff between current and target graphs */
[[nodiscard]] GraphDiff computeDiff(
    const PipelineGraph& current,
    const PipelineGraph& target);

} /* namespace core::graph */
