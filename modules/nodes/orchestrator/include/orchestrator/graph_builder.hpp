/* modules/nodes/orchestrator/include/orchestrator/graph_builder.hpp
 *
 * Converts ModeDefinition + ModuleDescriptors into a core::graph::PipelineGraph.
 * Vertices only — no edges.
 */

#pragma once

#include <string>
#include <vector>

#include <tl/expected.hpp>
#include "graph/pipeline_graph.hpp"
#include "orchestrator/mode_definition.hpp"
#include "orchestrator/module_launcher.hpp"


/* Build a PipelineGraph from a mode's required modules.
 * Static-only class — no state, no instance needed. */
class GraphBuilder {
public:
    /* Build a vertices-only graph for the given mode definition.
     * Each module in mode.required is matched against the modules vector
     * to produce a PipelineVertex. Returns an error if any required
     * module name has no matching ModuleDescriptor. */
    [[nodiscard]] static tl::expected<core::graph::PipelineGraph, std::string>
    buildForMode(const ModeDefinition& mode,
                 const std::vector<ModuleDescriptor>& modules);

    GraphBuilder() = delete;
};
