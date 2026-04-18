/* modules/core/pipeline_orchestrator/src/pipeline_orchestrator.cpp */

#include "pipeline_orchestrator/pipeline_orchestrator.hpp"

#include "graph/graph_diff.hpp"
#include "graph/graph_validator.hpp"

namespace core::pipeline_orchestrator {

PipelineOrchestrator::PipelineOrchestrator(OrchestratorConfig config)
    : config_(std::move(config))
    , crashHandler_(config_.crashHandler)
{
}

/* ── Profile switching ─────────────────────────────────────────────── */

tl::expected<SwitchPlan, OrchestratorError>
PipelineOrchestrator::switchToProfile(const core::graph::PipelineGraph& targetGraph)
{
    /* Validate target graph */
    auto validationResult = core::graph::validator::validateAll(
        targetGraph, config_.vramBudgetMiB);
    if (!validationResult.has_value()) {
        return tl::make_unexpected(OrchestratorError::kValidationFailed);
    }

    /* Compute diff */
    auto diff = core::graph::computeDiff(currentGraph_, targetGraph);

    /* Check if anything changed */
    if (diff.verticesToEvict.empty() && diff.verticesToSpawn.empty()
        && diff.edgesToRemove.empty() && diff.edgesToAdd.empty()) {
        return tl::make_unexpected(OrchestratorError::kAlreadyAtTarget);
    }

    /* Build execution plan */
    auto plan = buildSwitchPlan(currentGraph_, targetGraph, diff, config_.switchPlan);

    if (!config_.shadowMode) {
        /* In live mode, apply the graph transition.
         * The daemon calls executeDrain/executeEvict/etc. using ModuleLauncher.
         * We update our graph state here. */
        currentGraph_ = targetGraph;

        /* Initialize edge state machines for new edges */
        for (const auto& edgeId : plan.edgesToConnect) {
            edgeStates_.emplace(edgeId, core::graph::EdgeStateMachine{edgeId});
        }

        /* Remove state machines for removed edges */
        for (const auto& edgeId : plan.edgesToDrain) {
            edgeStates_.erase(edgeId);
        }
    }

    return plan;
}

/* ── Graph access ──────────────────────────────────────────────────── */

const core::graph::PipelineGraph& PipelineOrchestrator::currentGraph() const noexcept
{
    return currentGraph_;
}

core::graph::PipelineGraph& PipelineOrchestrator::currentGraph() noexcept
{
    return currentGraph_;
}

void PipelineOrchestrator::setCurrentGraph(core::graph::PipelineGraph graph)
{
    currentGraph_ = std::move(graph);
    edgeStates_.clear();

    /* Initialize edge states for all edges in the graph */
    for (const auto& edgeId : currentGraph_.allEdgeIds()) {
        edgeStates_.emplace(edgeId, core::graph::EdgeStateMachine{edgeId});
    }
}

/* ── Edge state management ─────────────────────────────────────────── */

const std::unordered_map<std::string, core::graph::EdgeStateMachine>&
PipelineOrchestrator::edgeStates() const noexcept
{
    return edgeStates_;
}

void PipelineOrchestrator::setEdgeState(const std::string& edgeId,
                                        core::graph::EdgeState state)
{
    auto it = edgeStates_.find(edgeId);
    if (it == edgeStates_.end()) return;

    auto& esm = it->second;
    switch (state) {
    case core::graph::EdgeState::kConnecting:
        (void)esm.beginConnecting();
        break;
    case core::graph::EdgeState::kActive:
        (void)esm.confirmActive();
        break;
    case core::graph::EdgeState::kDraining:
        (void)esm.beginDraining();
        break;
    case core::graph::EdgeState::kDisconnected:
        (void)esm.confirmDisconnected();
        break;
    case core::graph::EdgeState::kIdle:
        esm.forceReset();
        break;
    }
}

/* ── Introspection ─────────────────────────────────────────────────── */

std::string PipelineOrchestrator::introspectAsJson() const
{
    /* Build state map from edge state machines */
    std::unordered_map<std::string, core::graph::EdgeState> stateMap;
    for (const auto& [edgeId, esm] : edgeStates_) {
        stateMap[edgeId] = esm.current();
    }

    return core::graph::serialization::serializeToJson(currentGraph_, stateMap);
}

/* ── Crash handling ────────────────────────────────────────────────── */

CrashHandler& PipelineOrchestrator::crashHandler() noexcept
{
    return crashHandler_;
}

const CrashHandler& PipelineOrchestrator::crashHandler() const noexcept
{
    return crashHandler_;
}

/* ── Configuration ─────────────────────────────────────────────────── */

const OrchestratorConfig& PipelineOrchestrator::config() const noexcept
{
    return config_;
}

bool PipelineOrchestrator::isShadowMode() const noexcept
{
    return config_.shadowMode;
}

} /* namespace core::pipeline_orchestrator */
