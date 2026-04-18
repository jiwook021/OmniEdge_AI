#pragma once

/*
 * PipelineVertex — immutable descriptor for a single spawnable process.
 *
 * One PID = one vertex. Threads inside a process are NOT vertices.
 * Vertex IDs derive from omniedge_config.yaml module names.
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace core::graph {

struct PipelineVertex {
    std::string id;                              /* stable ID from YAML, e.g. "blur", "conv" */
    std::string binaryName;                      /* e.g. "omniedge_video_ingest" */
    std::string displayLabel;                    /* human-readable, e.g. "VideoIngest" */
    std::size_t vramBudgetMiB{0};
    std::vector<std::string> spawnArgs;
    std::optional<std::uint16_t> zmqPubPort;

    [[nodiscard]] bool operator==(const PipelineVertex& other) const {
        return id == other.id;
    }
};

} /* namespace core::graph */
