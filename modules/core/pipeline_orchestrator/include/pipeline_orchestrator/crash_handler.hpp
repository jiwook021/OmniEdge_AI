/* modules/core/pipeline_orchestrator/include/pipeline_orchestrator/crash_handler.hpp
 *
 * Graph-aware crash recovery.
 * Identifies impacted subgraph via topology::descendants(),
 * transitions affected edges, and coordinates SHM cleanup + restart.
 */

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <tl/expected.hpp>
#include "graph/edge_state_machine.hpp"
#include "graph/pipeline_graph.hpp"

namespace core::pipeline_orchestrator {

/* ── Error codes ───────────────────────────────────────────────────── */

enum class CrashHandlerError {
    kVertexNotFound,
    kPermanentlyDisabled,
    kRestartFailed
};

/* ── Crash impact report ───────────────────────────────────────────── */

struct CrashImpact {
    std::string                crashedVertex;
    std::vector<std::string>   affectedDescendants;
    std::vector<std::string>   disconnectedEdges;
    std::vector<std::string>   shmPathsToCleanup;
};

/* ── Configuration ─────────────────────────────────────────────────── */

struct CrashHandlerConfig {
    std::size_t maxCrashesBeforeDisable{5};
};

/* ── CrashHandler ──────────────────────────────────────────────────── */

class CrashHandler {
public:
    explicit CrashHandler(CrashHandlerConfig config = {});

    /* Analyze the impact of a crashed vertex on the graph.
     * Identifies descendants, affected edges, and SHM paths. */
    [[nodiscard]] tl::expected<CrashImpact, CrashHandlerError>
    analyzeImpact(const core::graph::PipelineGraph& graph,
                  const std::string&                crashedVertexId) const;

    /* Apply crash impact to edge state machines.
     * Transitions all affected edges to kDisconnected. */
    void applyEdgeTransitions(
        const CrashImpact& impact,
        std::unordered_map<std::string, core::graph::EdgeStateMachine>& edgeStates) const;

    /* Record a crash for a vertex. Returns true if the vertex
     * should be permanently disabled (exceeded max crashes). */
    bool recordCrash(const std::string& vertexId);

    /* Check if a vertex is permanently disabled. */
    [[nodiscard]] bool isDisabled(const std::string& vertexId) const;

    /* Reset crash count for a vertex (e.g., after successful long run). */
    void resetCrashCount(const std::string& vertexId);

    /* Get current crash count for a vertex. */
    [[nodiscard]] std::size_t crashCount(const std::string& vertexId) const;

private:
    CrashHandlerConfig config_;
    std::unordered_map<std::string, std::size_t> crashCounts_;
    std::unordered_map<std::string, bool>        disabled_;
};

} /* namespace core::pipeline_orchestrator */
