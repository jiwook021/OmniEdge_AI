/* modules/core/pipeline_orchestrator/src/crash_handler.cpp */

#include "pipeline_orchestrator/crash_handler.hpp"

#include <unordered_set>

#include "graph/graph_topology.hpp"
#include "graph/pipeline_edge.hpp"

namespace core::pipeline_orchestrator {

CrashHandler::CrashHandler(CrashHandlerConfig config)
    : config_(std::move(config))
{
}

tl::expected<CrashImpact, CrashHandlerError>
CrashHandler::analyzeImpact(const core::graph::PipelineGraph& graph,
                            const std::string&                crashedVertexId) const
{
    if (!graph.hasVertex(crashedVertexId)) {
        return tl::make_unexpected(CrashHandlerError::kVertexNotFound);
    }

    if (auto it = disabled_.find(crashedVertexId);
        it != disabled_.end() && it->second) {
        return tl::make_unexpected(CrashHandlerError::kPermanentlyDisabled);
    }

    CrashImpact impact;
    impact.crashedVertex = crashedVertexId;

    /* ── Find all downstream vertices ──────────────────────────────── */
    impact.affectedDescendants =
        core::graph::topology::descendants(graph, crashedVertexId);

    /* ── Collect edges touching the crashed vertex or its descendants ─ */
    std::unordered_set<std::string> affectedVertices;
    affectedVertices.insert(crashedVertexId);
    for (const auto& desc : impact.affectedDescendants) {
        affectedVertices.insert(desc);
    }

    std::unordered_set<std::string> shmPaths;
    for (const auto& edgeId : graph.allEdgeIds()) {
        auto edgeResult = graph.getEdge(edgeId);
        if (!edgeResult.has_value()) continue;

        const auto& edge = *edgeResult.value();
        if (affectedVertices.count(edge.fromVertex) ||
            affectedVertices.count(edge.toVertex)) {
            impact.disconnectedEdges.push_back(edgeId);

            auto path = core::graph::shmPathOf(edge);
            if (!path.empty()) {
                shmPaths.insert(path);
            }
        }
    }

    impact.shmPathsToCleanup.assign(shmPaths.begin(), shmPaths.end());
    return impact;
}

void CrashHandler::applyEdgeTransitions(
    const CrashImpact& impact,
    std::unordered_map<std::string, core::graph::EdgeStateMachine>& edgeStates) const
{
    for (const auto& edgeId : impact.disconnectedEdges) {
        auto it = edgeStates.find(edgeId);
        if (it != edgeStates.end()) {
            it->second.forceReset();
        }
    }
}

bool CrashHandler::recordCrash(const std::string& vertexId)
{
    auto& count = crashCounts_[vertexId];
    ++count;

    if (count >= config_.maxCrashesBeforeDisable) {
        disabled_[vertexId] = true;
        return true;
    }
    return false;
}

bool CrashHandler::isDisabled(const std::string& vertexId) const
{
    auto it = disabled_.find(vertexId);
    return it != disabled_.end() && it->second;
}

void CrashHandler::resetCrashCount(const std::string& vertexId)
{
    crashCounts_.erase(vertexId);
    disabled_.erase(vertexId);
}

std::size_t CrashHandler::crashCount(const std::string& vertexId) const
{
    auto it = crashCounts_.find(vertexId);
    return (it != crashCounts_.end()) ? it->second : 0;
}

} /* namespace core::pipeline_orchestrator */
