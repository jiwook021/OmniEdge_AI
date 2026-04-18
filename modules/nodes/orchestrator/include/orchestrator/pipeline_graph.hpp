#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

#include <tl/expected.hpp>

#include "common/string_hash.hpp"
#include "common/pipeline_types.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — PipelineGraph
//
// Parses the `pipelines:` YAML section, validates the graph (cycle detection,
// SHM name collision), and generates per-module CLI args for pipeline chaining.
//
// Each pipeline chain is a linear sequence of stages:
//   video_ingest → video_denoise → beauty → background_blur
//
// The graph resolver:
//   1. Parses YAML into PipelineStageDesc structs
//   2. Validates: no cycles, no duplicate SHM names, all modules registered
//   3. Generates CLI args for each stage: --shm-input, --output-format, etc.
//   4. Registers pipeline groups with PriorityScheduler for group eviction
// ---------------------------------------------------------------------------


/// Description of one stage in a pipeline chain.
struct PipelineStageDesc {
	std::string moduleName;         ///< e.g. "video_denoise"
	std::string inputShmName;       ///< e.g. "/oe.vid.ingest"
	std::string outputShmName;      ///< e.g. "/oe.cv.denoise.bgr"
	OutputFormat outputFormat = OutputFormat::kJpeg;
	int         inputPort    = 0;   ///< ZMQ SUB port for input notifications
	std::string inputTopic;         ///< ZMQ topic for input notifications
	int         outputPort   = 0;   ///< ZMQ PUB port for output notifications
	std::string outputTopic;        ///< ZMQ topic for output notifications
};

/// Description of a complete pipeline (linear chain of stages).
struct PipelineDesc {
	std::string              name;            ///< e.g. "enhanced_portrait"
	std::vector<PipelineStageDesc> stages;
	int                      groupPriority{0};      ///< eviction group priority
	std::size_t              groupVramBudgetMiB{0};  ///< total VRAM for all stages
};

/// CLI arguments generated for one module instance within a pipeline.
struct PipelineModuleArgs {
	std::string moduleName;
	std::vector<std::string> args;  ///< e.g. ["--shm-input", "/oe.cv.denoise.bgr", ...]
};

class PipelineGraph {
public:
	PipelineGraph() = default;

	/// Add a pipeline descriptor (parsed from YAML).
	void addPipeline(PipelineDesc pipeline);

	/// Resolve all pipelines: validate, detect cycles, check SHM collisions.
	/// Returns error string on validation failure.
	[[nodiscard]] tl::expected<void, std::string> resolve();

	/// Get all pipeline descriptors.
	[[nodiscard]] const std::vector<PipelineDesc>& pipelines() const { return pipelines_; }

	/// Generate CLI args for a specific module within a specific pipeline.
	/// Returns empty if the module is not part of any pipeline (standalone).
	[[nodiscard]] std::optional<PipelineModuleArgs>
	argsForModule(std::string_view pipelineName, std::string_view moduleName) const;

	/// Get all module args for a pipeline (in topological order).
	[[nodiscard]] std::vector<PipelineModuleArgs>
	allArgsForPipeline(std::string_view pipelineName) const;

	/// Get the pipeline that a module belongs to (if any).
	[[nodiscard]] std::optional<std::string>
	pipelineForModule(std::string_view moduleName) const;

	/// Check if the graph has been resolved.
	[[nodiscard]] bool isResolved() const { return resolved_; }

private:
	std::vector<PipelineDesc> pipelines_;

	// Module → pipeline name mapping (populated by resolve())
	std::unordered_map<std::string, std::string, StringHash, std::equal_to<>>
		moduleToPipeline_;

	// All SHM names used across all pipelines (for collision detection)
	std::unordered_map<std::string, std::string, StringHash, std::equal_to<>>
		shmNameOwner_;  ///< SHM name → "pipeline:module" owner

	bool resolved_{false};

	[[nodiscard]] tl::expected<void, std::string> validateNoCycles() const;
	[[nodiscard]] tl::expected<void, std::string> validateNoShmCollisions() const;
};
