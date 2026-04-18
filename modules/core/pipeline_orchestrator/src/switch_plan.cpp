/* modules/core/pipeline_orchestrator/src/switch_plan.cpp */

#include "pipeline_orchestrator/switch_plan.hpp"

#include <algorithm>
#include <unordered_set>

#include "graph/graph_topology.hpp"

namespace core::pipeline_orchestrator {

bool SwitchPlan::isEmpty() const noexcept
{
    return edgesToDrain.empty()
        && verticesToEvict.empty()
        && shmPathsToUnlink.empty()
        && verticesToSpawn.empty()
        && edgesToConnect.empty();
}

SwitchPlan buildSwitchPlan(
    const core::graph::PipelineGraph& current,
    const core::graph::PipelineGraph& target,
    const core::graph::GraphDiff&     diff,
    const SwitchPlanConfig&           config)
{
    SwitchPlan plan;
    plan.config = config;

    /* ── 1. Edges to drain = edges being removed ────────────────────── */
    plan.edgesToDrain = diff.edgesToRemove;

    /* ── 2. Eviction order from current graph (reverse-topo) ────────── */
    {
        std::unordered_set<std::string> evictSet(
            diff.verticesToEvict.begin(), diff.verticesToEvict.end());

        auto reverseOrder = core::graph::topology::reverseTopologicalSort(current);
        if (reverseOrder.has_value()) {
            for (const auto& vertexId : *reverseOrder) {
                if (evictSet.count(vertexId)) {
                    plan.verticesToEvict.push_back(vertexId);
                }
            }
        } else {
            /* Fallback: use diff order if cycle detected (shouldn't happen) */
            plan.verticesToEvict = diff.verticesToEvict;
        }
    }

    /* ── 3. SHM paths to unlink ─────────────────────────────────────── */
    plan.shmPathsToUnlink = diff.shmPathsToUnlink;

    /* ── 4. Spawn order from target graph (forward-topo) ────────────── */
    {
        std::unordered_set<std::string> spawnSet(
            diff.verticesToSpawn.begin(), diff.verticesToSpawn.end());

        auto topoOrder = core::graph::topology::topologicalSort(target);
        if (topoOrder.has_value()) {
            for (const auto& vertexId : *topoOrder) {
                if (spawnSet.count(vertexId)) {
                    plan.verticesToSpawn.push_back(vertexId);
                }
            }
        } else {
            plan.verticesToSpawn = diff.verticesToSpawn;
        }
    }

    /* ── 5. Edges to connect = edges being added ────────────────────── */
    plan.edgesToConnect = diff.edgesToAdd;

    return plan;
}

} /* namespace core::pipeline_orchestrator */
