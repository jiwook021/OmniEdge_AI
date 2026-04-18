/* modules/core/pipeline_orchestrator/include/pipeline_orchestrator/switch_plan.hpp
 *
 * Ordered execution plan for a profile switch.
 * Built from a GraphDiff + topological ordering.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "graph/graph_diff.hpp"
#include "graph/pipeline_graph.hpp"

namespace core::pipeline_orchestrator {

/* ── Configurable timeouts ─────────────────────────────────────────── */
/* Compile-time fallback only — INI [orchestrator_graph] drives the
 * runtime value through OrchestratorGraphConfig, which uses
 * kOrchestratorDrainTimeoutMs from common/runtime_defaults.hpp.
 */
struct SwitchPlanConfig {
    std::size_t drainTimeoutMs{5000};
    std::size_t connectRetryCount{10};
    std::size_t connectRetryIntervalMs{500};
};

/* ── Switch plan ───────────────────────────────────────────────────── */

struct SwitchPlan {
    /* Phase ordering (each vector is processed sequentially):
     *   1. drain   — edges to drain before teardown
     *   2. evict   — vertices to stop (reverse-topo order)
     *   3. unlink  — SHM paths to shm_unlink
     *   4. spawn   — vertices to start (topo order)
     *   5. connect — edges to establish after spawn
     */
    std::vector<std::string> edgesToDrain;
    std::vector<std::string> verticesToEvict;
    std::vector<std::string> shmPathsToUnlink;
    std::vector<std::string> verticesToSpawn;
    std::vector<std::string> edgesToConnect;

    SwitchPlanConfig config;

    [[nodiscard]] bool isEmpty() const noexcept;
};

/* Build a SwitchPlan from a diff and the target graph.
 * Uses topological sort on target for spawn order,
 * reverse-topo on current for eviction order. */
[[nodiscard]] SwitchPlan buildSwitchPlan(
    const core::graph::PipelineGraph& current,
    const core::graph::PipelineGraph& target,
    const core::graph::GraphDiff&     diff,
    const SwitchPlanConfig&           config = {});

} /* namespace core::pipeline_orchestrator */
