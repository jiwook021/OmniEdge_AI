#pragma once

/*
 * GraphValidator — validation suite for pipeline graph correctness.
 *
 * Call validateAll() at config load time. Refuse to start if validation fails.
 */

#include "graph/pipeline_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tl/expected.hpp>

namespace core::graph::validator {

enum class ValidationError : std::uint8_t {
    kCycleDetected = 0,
    kDisconnectedVertex,
    kVramBudgetExceeded,
    kDuplicateShmProducer
};

[[nodiscard]] const char* validationErrorToString(ValidationError error);

/* No cycles in the graph */
[[nodiscard]] tl::expected<void, ValidationError>
    validateDag(const PipelineGraph& graph);

/* Every non-source vertex has at least one incoming edge */
[[nodiscard]] tl::expected<void, ValidationError>
    validateConnectedSources(const PipelineGraph& graph);

/* Sum of vertex VRAM budgets does not exceed gpuBudgetMiB */
[[nodiscard]] tl::expected<void, ValidationError>
    validateVramBudget(const PipelineGraph& graph, std::size_t gpuBudgetMiB);

/* No two producers write to the same SHM path */
[[nodiscard]] tl::expected<void, ValidationError>
    validateShmPathUniqueness(const PipelineGraph& graph);

/* Runs all validators, returns first failure */
[[nodiscard]] tl::expected<void, ValidationError>
    validateAll(const PipelineGraph& graph, std::size_t gpuBudgetMiB);

} /* namespace core::graph::validator */
