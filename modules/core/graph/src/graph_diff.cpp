#include "graph/graph_diff.hpp"

#include <algorithm>
#include <unordered_set>

namespace core::graph {

GraphDiff computeDiff(const PipelineGraph& current, const PipelineGraph& target) {
    GraphDiff diff;

    /* Vertex diff */
    const auto currentVertices = current.allVertexIds();
    const auto targetVertices = target.allVertexIds();

    const std::unordered_set<std::string> currentSet(currentVertices.begin(), currentVertices.end());
    const std::unordered_set<std::string> targetSet(targetVertices.begin(), targetVertices.end());

    for (const auto& vid : currentVertices) {
        if (targetSet.count(vid) > 0) {
            diff.verticesKept.push_back(vid);
        } else {
            diff.verticesToEvict.push_back(vid);
        }
    }
    for (const auto& vid : targetVertices) {
        if (currentSet.count(vid) == 0) {
            diff.verticesToSpawn.push_back(vid);
        }
    }

    /* Edge diff */
    const auto currentEdges = current.allEdgeIds();
    const auto targetEdges = target.allEdgeIds();

    const std::unordered_set<std::string> currentEdgeSet(currentEdges.begin(), currentEdges.end());
    const std::unordered_set<std::string> targetEdgeSet(targetEdges.begin(), targetEdges.end());

    for (const auto& eid : currentEdges) {
        if (targetEdgeSet.count(eid) > 0) {
            diff.edgesKept.push_back(eid);
        } else {
            diff.edgesToRemove.push_back(eid);
        }
    }
    for (const auto& eid : targetEdges) {
        if (currentEdgeSet.count(eid) == 0) {
            diff.edgesToAdd.push_back(eid);
        }
    }

    /* SHM paths to unlink: paths in current edges being removed where
       no edge in target references the same path */
    std::unordered_set<std::string> targetShmPaths;
    for (const auto& eid : targetEdges) {
        const auto edgeResult = target.getEdge(eid);
        if (edgeResult) {
            const auto path = shmPathOf(*edgeResult.value());
            if (!path.empty()) {
                targetShmPaths.insert(path);
            }
        }
    }

    std::unordered_set<std::string> unlinkSet;
    for (const auto& eid : diff.edgesToRemove) {
        const auto edgeResult = current.getEdge(eid);
        if (edgeResult) {
            const auto path = shmPathOf(*edgeResult.value());
            if (!path.empty() && targetShmPaths.count(path) == 0) {
                unlinkSet.insert(path);
            }
        }
    }
    diff.shmPathsToUnlink.assign(unlinkSet.begin(), unlinkSet.end());
    std::sort(diff.shmPathsToUnlink.begin(), diff.shmPathsToUnlink.end());

    return diff;
}

} /* namespace core::graph */
