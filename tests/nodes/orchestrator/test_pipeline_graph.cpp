// ---------------------------------------------------------------------------
// test_pipeline_graph.cpp — PipelineGraph unit tests
//
// Tests: pipeline resolution, cycle detection, SHM collision validation,
//        per-module CLI arg generation, module-to-pipeline lookup.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "orchestrator/pipeline_graph.hpp"

namespace {

// ── Helper to build a simple two-stage pipeline ───────────────────────────

PipelineDesc makeTwoStagePipeline(const std::string& name,
                                   const std::string& mod1,
                                   const std::string& mod2)
{
	PipelineDesc desc;
	desc.name = name;

	PipelineStageDesc s1;
	s1.moduleName   = mod1;
	s1.outputShmName = "/oe.test." + mod1 + ".bgr";
	s1.outputFormat  = OutputFormat::kBgr24;

	PipelineStageDesc s2;
	s2.moduleName    = mod2;
	s2.inputShmName  = s1.outputShmName;
	s2.inputPort     = 5600;
	s2.inputTopic    = mod1 + "_bgr_frame";
	s2.outputShmName = "/oe.test." + mod2 + ".jpeg";
	s2.outputFormat  = OutputFormat::kJpeg;

	desc.stages = {s1, s2};
	return desc;
}


// ── Resolve succeeds on valid pipeline ────────────────────────────────────

TEST(PipelineGraphTest, ResolveSucceedsOnValidPipeline)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("test_chain", "denoise", "blur"));

	auto result = graph.resolve();
	EXPECT_TRUE(result.has_value());
	EXPECT_TRUE(graph.isResolved());
}

// ── Empty graph resolves ──────────────────────────────────────────────────

TEST(PipelineGraphTest, EmptyGraphResolves)
{
	PipelineGraph graph;
	auto result = graph.resolve();
	EXPECT_TRUE(result.has_value());
	EXPECT_TRUE(graph.isResolved());
}

// ── Adding a pipeline invalidates resolved state ──────────────────────────

TEST(PipelineGraphTest, AddPipelineInvalidatesResolved)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain1", "a", "b"));
	auto result = graph.resolve();
	ASSERT_TRUE(result.has_value());
	EXPECT_TRUE(graph.isResolved());

	graph.addPipeline(makeTwoStagePipeline("chain2", "c", "d"));
	EXPECT_FALSE(graph.isResolved());
}

// ── Module appears in two pipelines: rejected ─────────────────────────────

TEST(PipelineGraphTest, ModuleInTwoPipelinesRejected)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain1", "denoise", "blur"));
	graph.addPipeline(makeTwoStagePipeline("chain2", "denoise", "beauty"));

	auto result = graph.resolve();
	EXPECT_FALSE(result.has_value());
	EXPECT_NE(result.error().find("denoise"), std::string::npos);
}

// ── Cycle detection: stage reads from later stage's output ────────────────

TEST(PipelineGraphTest, CycleDetected)
{
	PipelineDesc desc;
	desc.name = "cyclic";

	PipelineStageDesc s1;
	s1.moduleName    = "a";
	s1.inputShmName  = "/oe.test.b.bgr";   // reads from b's output
	s1.outputShmName = "/oe.test.a.bgr";

	PipelineStageDesc s2;
	s2.moduleName    = "b";
	s2.inputShmName  = "/oe.test.a.bgr";
	s2.outputShmName = "/oe.test.b.bgr";   // a reads from this (cycle)

	desc.stages = {s1, s2};

	PipelineGraph graph;
	graph.addPipeline(std::move(desc));

	auto result = graph.resolve();
	EXPECT_FALSE(result.has_value());
	EXPECT_NE(result.error().find("cycle"), std::string::npos);
}

// ── SHM name collision across pipelines ───────────────────────────────────

TEST(PipelineGraphTest, ShmCollisionDetected)
{
	PipelineDesc p1;
	p1.name = "chain1";
	PipelineStageDesc s1;
	s1.moduleName    = "a";
	s1.outputShmName = "/oe.shared.name";
	p1.stages = {s1};

	PipelineDesc p2;
	p2.name = "chain2";
	PipelineStageDesc s2;
	s2.moduleName    = "b";
	s2.outputShmName = "/oe.shared.name";  // collision with chain1
	p2.stages = {s2};

	PipelineGraph graph;
	graph.addPipeline(std::move(p1));
	graph.addPipeline(std::move(p2));

	auto result = graph.resolve();
	EXPECT_FALSE(result.has_value());
	EXPECT_NE(result.error().find("collision"), std::string::npos);
}

// ── No collision when SHM names differ ────────────────────────────────────

TEST(PipelineGraphTest, NoCollisionWhenShmNamesDiffer)
{
	PipelineDesc p1;
	p1.name = "chain1";
	PipelineStageDesc s1;
	s1.moduleName    = "a";
	s1.outputShmName = "/oe.chain1.a";
	p1.stages = {s1};

	PipelineDesc p2;
	p2.name = "chain2";
	PipelineStageDesc s2;
	s2.moduleName    = "b";
	s2.outputShmName = "/oe.chain2.b";
	p2.stages = {s2};

	PipelineGraph graph;
	graph.addPipeline(std::move(p1));
	graph.addPipeline(std::move(p2));

	auto result = graph.resolve();
	EXPECT_TRUE(result.has_value());
}

// ── argsForModule generates correct CLI args ──────────────────────────────

TEST(PipelineGraphTest, ArgsForModuleGeneratesCliArgs)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain", "denoise", "blur"));
	ASSERT_TRUE(graph.resolve().has_value());

	// Stage 2 (blur) should get --shm-input, --input-port, --input-topic
	auto args = graph.argsForModule("chain", "blur");
	ASSERT_TRUE(args.has_value());
	EXPECT_EQ(args->moduleName, "blur");

	// Check that --shm-input is present with correct value
	bool foundShmInput = false;
	bool foundInputPort = false;
	bool foundInputTopic = false;
	for (std::size_t i = 0; i + 1 < args->args.size(); ++i) {
		if (args->args[i] == "--shm-input") {
			EXPECT_EQ(args->args[i + 1], "/oe.test.denoise.bgr");
			foundShmInput = true;
		}
		if (args->args[i] == "--input-port") {
			EXPECT_EQ(args->args[i + 1], "5600");
			foundInputPort = true;
		}
		if (args->args[i] == "--input-topic") {
			EXPECT_EQ(args->args[i + 1], "denoise_bgr_frame");
			foundInputTopic = true;
		}
	}
	EXPECT_TRUE(foundShmInput);
	EXPECT_TRUE(foundInputPort);
	EXPECT_TRUE(foundInputTopic);
}

// ── argsForModule for first stage (no input overrides) ────────────────────

TEST(PipelineGraphTest, ArgsForFirstStageMinimal)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain", "denoise", "blur"));
	ASSERT_TRUE(graph.resolve().has_value());

	// Stage 1 (denoise) has no inputShmName, no inputPort, no inputTopic
	auto args = graph.argsForModule("chain", "denoise");
	ASSERT_TRUE(args.has_value());
	EXPECT_EQ(args->moduleName, "denoise");

	// Should have --output-format bgr24 but no --shm-input
	bool foundShmInput = false;
	bool foundOutputFormat = false;
	for (std::size_t i = 0; i + 1 < args->args.size(); ++i) {
		if (args->args[i] == "--shm-input") foundShmInput = true;
		if (args->args[i] == "--output-format") {
			EXPECT_EQ(args->args[i + 1], "bgr24");
			foundOutputFormat = true;
		}
	}
	EXPECT_FALSE(foundShmInput);
	EXPECT_TRUE(foundOutputFormat);
}

// ── argsForModule returns nullopt for unknown module ──────────────────────

TEST(PipelineGraphTest, ArgsForUnknownModuleReturnsNullopt)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain", "denoise", "blur"));
	ASSERT_TRUE(graph.resolve().has_value());

	auto args = graph.argsForModule("chain", "nonexistent");
	EXPECT_FALSE(args.has_value());
}

// ── allArgsForPipeline returns all stages in order ────────────────────────

TEST(PipelineGraphTest, AllArgsForPipelineReturnsAllStages)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("chain", "denoise", "blur"));
	ASSERT_TRUE(graph.resolve().has_value());

	auto allArgs = graph.allArgsForPipeline("chain");
	ASSERT_EQ(allArgs.size(), 2u);
	EXPECT_EQ(allArgs[0].moduleName, "denoise");
	EXPECT_EQ(allArgs[1].moduleName, "blur");
}

// ── pipelineForModule returns correct pipeline ────────────────────────────

TEST(PipelineGraphTest, PipelineForModuleReturnsCorrectPipeline)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("portrait", "denoise", "blur"));
	ASSERT_TRUE(graph.resolve().has_value());

	auto p1 = graph.pipelineForModule("denoise");
	ASSERT_TRUE(p1.has_value());
	EXPECT_EQ(*p1, "portrait");

	auto p2 = graph.pipelineForModule("blur");
	ASSERT_TRUE(p2.has_value());
	EXPECT_EQ(*p2, "portrait");

	auto p3 = graph.pipelineForModule("nonexistent");
	EXPECT_FALSE(p3.has_value());
}

// ── Three-stage pipeline resolves and generates correct args ──────────────

TEST(PipelineGraphTest, ThreeStagePipelineResolves)
{
	PipelineDesc desc;
	desc.name = "enhanced_portrait";

	PipelineStageDesc s1;
	s1.moduleName    = "video_denoise";
	s1.outputShmName = "/oe.cv.denoise.bgr";
	s1.outputFormat  = OutputFormat::kBgr24;

	PipelineStageDesc s2;
	s2.moduleName    = "beauty";
	s2.inputShmName  = "/oe.cv.denoise.bgr";
	s2.inputPort     = 5568;
	s2.inputTopic    = "denoise_bgr_frame";
	s2.outputShmName = "/oe.cv.beauty.bgr";
	s2.outputFormat  = OutputFormat::kBgr24;

	PipelineStageDesc s3;
	s3.moduleName    = "background_blur";
	s3.inputShmName  = "/oe.cv.beauty.bgr";
	s3.inputPort     = 5579;
	s3.inputTopic    = "beauty_bgr_frame";
	s3.outputShmName = "/oe.cv.blur.jpeg";
	s3.outputFormat  = OutputFormat::kJpeg;

	desc.stages = {s1, s2, s3};

	PipelineGraph graph;
	graph.addPipeline(std::move(desc));

	auto result = graph.resolve();
	ASSERT_TRUE(result.has_value());

	// Verify middle stage args
	auto args = graph.argsForModule("enhanced_portrait", "beauty");
	ASSERT_TRUE(args.has_value());

	bool foundShmInput = false;
	for (std::size_t i = 0; i + 1 < args->args.size(); ++i) {
		if (args->args[i] == "--shm-input") {
			EXPECT_EQ(args->args[i + 1], "/oe.cv.denoise.bgr");
			foundShmInput = true;
		}
	}
	EXPECT_TRUE(foundShmInput);

	// Verify topological order in allArgs
	auto allArgs = graph.allArgsForPipeline("enhanced_portrait");
	ASSERT_EQ(allArgs.size(), 3u);
	EXPECT_EQ(allArgs[0].moduleName, "video_denoise");
	EXPECT_EQ(allArgs[1].moduleName, "beauty");
	EXPECT_EQ(allArgs[2].moduleName, "background_blur");
}

// ── Multiple independent pipelines resolve ────────────────────────────────

TEST(PipelineGraphTest, MultipleIndependentPipelinesResolve)
{
	PipelineGraph graph;
	graph.addPipeline(makeTwoStagePipeline("video_chain", "denoise", "blur"));
	graph.addPipeline(makeTwoStagePipeline("audio_chain", "audio_denoise", "tts"));

	auto result = graph.resolve();
	ASSERT_TRUE(result.has_value());

	EXPECT_EQ(graph.pipelines().size(), 2u);

	auto p1 = graph.pipelineForModule("denoise");
	ASSERT_TRUE(p1.has_value());
	EXPECT_EQ(*p1, "video_chain");

	auto p2 = graph.pipelineForModule("tts");
	ASSERT_TRUE(p2.has_value());
	EXPECT_EQ(*p2, "audio_chain");
}

// ── Stages with empty outputShmName are valid (leaf nodes) ────────────────

TEST(PipelineGraphTest, LeafNodeWithEmptyOutputIsValid)
{
	PipelineDesc desc;
	desc.name = "display_only";

	PipelineStageDesc s1;
	s1.moduleName    = "producer";
	s1.outputShmName = "/oe.test.producer.bgr";
	s1.outputFormat  = OutputFormat::kBgr24;

	PipelineStageDesc s2;
	s2.moduleName   = "display";
	s2.inputShmName = "/oe.test.producer.bgr";
	// No output — leaf consumer

	desc.stages = {s1, s2};

	PipelineGraph graph;
	graph.addPipeline(std::move(desc));

	auto result = graph.resolve();
	EXPECT_TRUE(result.has_value());
}

} // namespace
