/* modules/core/pipeline_orchestrator/include/pipeline_orchestrator/pipeline_orchestrator.hpp
 *
 * Top-level graph-based orchestrator.
 * Owns the live PipelineGraph + edge state machines.
 * Delegates lifecycle to ModuleLauncher, VramGate, ShmRegistry.
 */

#pragma once

#include <string>
#include <unordered_map>

#include <tl/expected.hpp>
#include "graph/edge_state_machine.hpp"
#include "graph/pipeline_graph.hpp"
#include "graph/graph_serialization.hpp"
#include "pipeline_orchestrator/crash_handler.hpp"
#include "pipeline_orchestrator/switch_plan.hpp"

namespace core::pipeline_orchestrator {

/* ── Error codes ───────────────────────────────────────────────────── */

enum class OrchestratorError {
    kValidationFailed,
    kPlanBuildFailed,
    kDrainTimeout,
    kEvictFailed,
    kSpawnFailed,
    kAlreadyAtTarget
};

/* ── Configuration ─────────────────────────────────────────────────── */

struct OrchestratorConfig {
    bool        shadowMode{true};
    std::size_t vramBudgetMiB{11264};
    SwitchPlanConfig switchPlan;
    CrashHandlerConfig crashHandler;
};

/* ── PipelineOrchestrator ──────────────────────────────────────────── */

class PipelineOrchestrator {
public:
    explicit PipelineOrchestrator(OrchestratorConfig config = {});

    /* ── Profile switching ─────────────────────────────────────────── */

    /* Build a target graph for the given profile, compute diff, and
     * generate a switch plan. In shadow mode, only logs divergences. */
    [[nodiscard]] tl::expected<SwitchPlan, OrchestratorError>
    switchToProfile(const core::graph::PipelineGraph& targetGraph);

    /* ── Graph access ──────────────────────────────────────────────── */

    [[nodiscard]] const core::graph::PipelineGraph& currentGraph() const noexcept;
    [[nodiscard]] core::graph::PipelineGraph& currentGraph() noexcept;

    /* Replace the current graph (used during initial load). */
    void setCurrentGraph(core::graph::PipelineGraph graph);

    /* ── Edge state management ─────────────────────────────────────── */

    [[nodiscard]] const std::unordered_map<std::string, core::graph::EdgeStateMachine>&
    edgeStates() const noexcept;

    void setEdgeState(const std::string& edgeId, core::graph::EdgeState state);

    /* ── Introspection ─────────────────────────────────────────────── */

    /* Serialize current graph + edge states as JSON. */
    [[nodiscard]] std::string introspectAsJson() const;

    /* ── Crash handling ────────────────────────────────────────────── */

    [[nodiscard]] CrashHandler& crashHandler() noexcept;
    [[nodiscard]] const CrashHandler& crashHandler() const noexcept;

    /* ── Configuration ─────────────────────────────────────────────── */

    [[nodiscard]] const OrchestratorConfig& config() const noexcept;
    [[nodiscard]] bool isShadowMode() const noexcept;

private:
    OrchestratorConfig config_;
    core::graph::PipelineGraph currentGraph_;
    std::unordered_map<std::string, core::graph::EdgeStateMachine> edgeStates_;
    CrashHandler crashHandler_;
};

} /* namespace core::pipeline_orchestrator */
