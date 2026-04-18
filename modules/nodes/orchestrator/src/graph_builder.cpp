/* modules/nodes/orchestrator/src/graph_builder.cpp */

#include "orchestrator/graph_builder.hpp"

#include <algorithm>
#include <format>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "graph/pipeline_vertex.hpp"


tl::expected<core::graph::PipelineGraph, std::string>
GraphBuilder::buildForMode(const ModeDefinition& mode,
                           const std::vector<ModuleDescriptor>& modules)
{
    OE_ZONE_SCOPED;
    core::graph::PipelineGraph graph;

    for (const auto& requiredName : mode.required) {
        /* Linear scan — mode.required is typically 5-7 entries,
         * modules is 7-15 entries. O(N*M) is fine. */
        auto it = std::ranges::find_if(modules,
            [&](const ModuleDescriptor& desc) { return desc.name == requiredName; });

        if (it == modules.end()) {
            return tl::make_unexpected(
                std::format("mode '{}': unknown module '{}'", mode.name, requiredName));
        }

        const auto& desc = *it;
        core::graph::PipelineVertex vertex{
            .id            = desc.name,
            .binaryName    = desc.binaryPath,
            .displayLabel  = desc.name,
            .vramBudgetMiB = desc.vramBudgetMb,
            .spawnArgs     = desc.args,
            .zmqPubPort    = desc.zmqPubPort > 0
                                 ? std::optional<std::uint16_t>(
                                       static_cast<std::uint16_t>(desc.zmqPubPort))
                                 : std::nullopt,
        };

        auto addResult = graph.addVertex(std::move(vertex));
        if (!addResult) {
            return tl::make_unexpected(
                std::format("mode '{}': duplicate vertex '{}'", mode.name, desc.name));
        }
    }

    OE_LOG_DEBUG("graph_builder: mode='{}' → {} vertices, 0 edges",
                 mode.name, graph.vertexCount());
    return graph;
}
