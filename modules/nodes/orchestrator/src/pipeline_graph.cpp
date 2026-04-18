#include "orchestrator/pipeline_graph.hpp"

#include <format>
#include <unordered_set>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


void PipelineGraph::addPipeline(PipelineDesc pipeline)
{
	pipelines_.push_back(std::move(pipeline));
	resolved_ = false;  // invalidate on any change
}

tl::expected<void, std::string> PipelineGraph::resolve()
{
	OE_ZONE_SCOPED;

	moduleToPipeline_.clear();
	shmNameOwner_.clear();

	// Build module→pipeline mapping
	for (const auto& pipeline : pipelines_) {
		for (const auto& stage : pipeline.stages) {
			auto [it, inserted] = moduleToPipeline_.try_emplace(
				stage.moduleName, pipeline.name);
			if (!inserted && it->second != pipeline.name) {
				return tl::unexpected(std::format(
					"Module '{}' appears in pipelines '{}' and '{}'",
					stage.moduleName, it->second, pipeline.name));
			}
		}
	}

	// Validate
	if (auto r = validateNoCycles(); !r) return r;
	if (auto r = validateNoShmCollisions(); !r) return r;

	resolved_ = true;
	SPDLOG_INFO("PipelineGraph: resolved {} pipelines, {} modules",
	            pipelines_.size(), moduleToPipeline_.size());
	return {};
}

std::optional<PipelineModuleArgs>
PipelineGraph::argsForModule(std::string_view pipelineName,
                             std::string_view moduleName) const
{
	for (const auto& pipeline : pipelines_) {
		if (pipeline.name != pipelineName) continue;

		for (std::size_t i = 0; i < pipeline.stages.size(); ++i) {
			const auto& stage = pipeline.stages[i];
			if (stage.moduleName != moduleName) continue;

			PipelineModuleArgs result;
			result.moduleName = stage.moduleName;

			// Input SHM override (stages after the first get chained input)
			if (!stage.inputShmName.empty()) {
				result.args.push_back("--shm-input");
				result.args.push_back(stage.inputShmName);
			}

			// Input port override
			if (stage.inputPort > 0) {
				result.args.push_back("--input-port");
				result.args.push_back(std::to_string(stage.inputPort));
			}

			// Input topic override
			if (!stage.inputTopic.empty()) {
				result.args.push_back("--input-topic");
				result.args.push_back(stage.inputTopic);
			}

			// Output format
			if (stage.outputFormat == OutputFormat::kBgr24) {
				result.args.push_back("--output-format");
				result.args.push_back("bgr24");
			}

			return result;
		}
	}
	return std::nullopt;
}

std::vector<PipelineModuleArgs>
PipelineGraph::allArgsForPipeline(std::string_view pipelineName) const
{
	std::vector<PipelineModuleArgs> result;

	for (const auto& pipeline : pipelines_) {
		if (pipeline.name != pipelineName) continue;

		for (const auto& stage : pipeline.stages) {
			auto args = argsForModule(pipelineName, stage.moduleName);
			if (args) {
				result.push_back(std::move(*args));
			}
		}
		break;
	}
	return result;
}

std::optional<std::string>
PipelineGraph::pipelineForModule(std::string_view moduleName) const
{
	auto it = moduleToPipeline_.find(moduleName);
	if (it == moduleToPipeline_.end()) return std::nullopt;
	return it->second;
}

// ── Validation ─────────────────────────────────────────────────────────────

tl::expected<void, std::string> PipelineGraph::validateNoCycles() const
{
	// Pipelines are linear chains by definition (ordered vector of stages).
	// A cycle would require stage N's output to be stage M's input where M < N.
	// Check by building a set of all output SHM names as we walk each chain
	// and verifying no input references a later stage's output.

	for (const auto& pipeline : pipelines_) {
		std::unordered_set<std::string, StringHash, std::equal_to<>> outputs;

		for (const auto& stage : pipeline.stages) {
			// Input must not be an output of a later stage
			// (at this point, outputs only contains earlier stages)
			if (!stage.outputShmName.empty()) {
				outputs.insert(stage.outputShmName);
			}
		}

		// Walk backwards: each stage's input must NOT be an output that
		// appears after it in the chain
		for (std::size_t i = 0; i < pipeline.stages.size(); ++i) {
			const auto& stage = pipeline.stages[i];
			// Check if this stage's input is produced by a stage after it
			for (std::size_t j = i + 1; j < pipeline.stages.size(); ++j) {
				if (pipeline.stages[j].outputShmName == stage.inputShmName &&
				    !stage.inputShmName.empty()) {
					return tl::unexpected(std::format(
						"Pipeline '{}': cycle detected — stage '{}' reads from '{}' "
						"which is produced by later stage '{}'",
						pipeline.name, stage.moduleName,
						stage.inputShmName, pipeline.stages[j].moduleName));
				}
			}
		}
	}
	return {};
}

tl::expected<void, std::string> PipelineGraph::validateNoShmCollisions() const
{
	std::unordered_map<std::string, std::string, StringHash, std::equal_to<>>
		shmOwners;

	for (const auto& pipeline : pipelines_) {
		for (const auto& stage : pipeline.stages) {
			if (stage.outputShmName.empty()) continue;

			const std::string owner = pipeline.name + ":" + stage.moduleName;
			auto [it, inserted] = shmOwners.try_emplace(stage.outputShmName, owner);
			if (!inserted) {
				return tl::unexpected(std::format(
					"SHM name collision: '{}' claimed by '{}' and '{}'",
					stage.outputShmName, it->second, owner));
			}
		}
	}
	return {};
}
