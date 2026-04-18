#include "graph/pipeline_edge.hpp"

namespace core::graph {

std::string shmPathOf(const PipelineEdge& edge) {
    if (edge.transport != TransportType::kShm) {
        return {};
    }
    return std::get<ShmAttributes>(edge.attributes).shmPath;
}

} /* namespace core::graph */
