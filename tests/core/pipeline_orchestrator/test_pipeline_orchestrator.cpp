/* tests/core/pipeline_orchestrator/test_pipeline_orchestrator.cpp */

#include <gtest/gtest.h>
#include "pipeline_orchestrator/pipeline_orchestrator.hpp"

#include <nlohmann/json.hpp>

using namespace core::graph;
using namespace core::pipeline_orchestrator;

class PipelineOrchestratorTest : public ::testing::Test {
protected:
    OrchestratorConfig config_{.shadowMode = false, .vramBudgetMiB = 11264,
                               .switchPlan = {}, .crashHandler = {}};
    PipelineOrchestrator orchestrator_{config_};

    void makeVertex(PipelineGraph& g, const std::string& id,
                    std::size_t vram = 0) {
        g.addVertex({.id = id, .binaryName = "bin_" + id, .vramBudgetMiB = vram});
    }

    void makeShmEdge(PipelineGraph& g, const std::string& from,
                     const std::string& to, const std::string& shmPath) {
        g.addEdge({
            .id = from + "->" + to,
            .fromVertex = from,
            .toVertex = to,
            .transport = TransportType::kShm,
            .attributes = ShmAttributes{.shmPath = shmPath}
        });
    }
};

TEST_F(PipelineOrchestratorTest, test_initialState) {
    EXPECT_EQ(orchestrator_.currentGraph().vertexCount(), 0);
    EXPECT_TRUE(orchestrator_.edgeStates().empty());
    EXPECT_FALSE(orchestrator_.isShadowMode());
}

TEST_F(PipelineOrchestratorTest, test_setCurrentGraph) {
    PipelineGraph graph;
    makeVertex(graph, "cam");
    makeVertex(graph, "blur");
    makeShmEdge(graph, "cam", "blur", "/oe.vid.ingest");

    orchestrator_.setCurrentGraph(std::move(graph));

    EXPECT_EQ(orchestrator_.currentGraph().vertexCount(), 2);
    EXPECT_EQ(orchestrator_.currentGraph().edgeCount(), 1);
    EXPECT_EQ(orchestrator_.edgeStates().size(), 1);
    EXPECT_EQ(orchestrator_.edgeStates().at("cam->blur").current(),
              EdgeState::kIdle);
}

TEST_F(PipelineOrchestratorTest, test_switchToProfile) {
    /* Set up current: cam → blur */
    PipelineGraph currentGraph;
    makeVertex(currentGraph, "cam");
    makeVertex(currentGraph, "blur");
    makeShmEdge(currentGraph, "cam", "blur", "/oe.vid.ingest");
    orchestrator_.setCurrentGraph(std::move(currentGraph));

    /* Build target: cam → conv */
    PipelineGraph target;
    makeVertex(target, "cam");
    makeVertex(target, "conv", 7500);
    makeShmEdge(target, "cam", "conv", "/oe.vid.ingest");

    auto result = orchestrator_.switchToProfile(target);
    ASSERT_TRUE(result.has_value());

    /* Plan should have evictions and spawns */
    EXPECT_EQ(result->verticesToEvict.size(), 1);
    EXPECT_EQ(result->verticesToSpawn.size(), 1);

    /* Current graph should now match target */
    EXPECT_TRUE(orchestrator_.currentGraph().hasVertex("conv"));
    EXPECT_FALSE(orchestrator_.currentGraph().hasVertex("blur"));
}

TEST_F(PipelineOrchestratorTest, test_switchToSameProfile) {
    PipelineGraph graph;
    makeVertex(graph, "cam");
    makeVertex(graph, "blur");
    makeShmEdge(graph, "cam", "blur", "/oe.vid.ingest");
    orchestrator_.setCurrentGraph(std::move(graph));

    /* Switch to identical graph */
    PipelineGraph target;
    makeVertex(target, "cam");
    makeVertex(target, "blur");
    makeShmEdge(target, "cam", "blur", "/oe.vid.ingest");

    auto result = orchestrator_.switchToProfile(target);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), OrchestratorError::kAlreadyAtTarget);
}

TEST_F(PipelineOrchestratorTest, test_switchValidationFails) {
    /* Target with VRAM exceeding budget */
    PipelineGraph target;
    makeVertex(target, "a", 8000);
    makeVertex(target, "b", 8000);
    makeShmEdge(target, "a", "b", "/oe.test");

    auto result = orchestrator_.switchToProfile(target);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), OrchestratorError::kValidationFailed);
}

TEST_F(PipelineOrchestratorTest, test_shadowModePreservesGraph) {
    OrchestratorConfig shadowConfig{.shadowMode = true, .vramBudgetMiB = 11264,
                                    .switchPlan = {}, .crashHandler = {}};
    PipelineOrchestrator shadowOrch(shadowConfig);

    PipelineGraph graph;
    makeVertex(graph, "cam");
    makeVertex(graph, "blur");
    makeShmEdge(graph, "cam", "blur", "/oe.vid.ingest");
    shadowOrch.setCurrentGraph(std::move(graph));

    /* Switch to different target in shadow mode */
    PipelineGraph target;
    makeVertex(target, "cam");
    makeVertex(target, "conv");
    makeShmEdge(target, "cam", "conv", "/oe.vid.ingest");

    auto result = shadowOrch.switchToProfile(target);
    ASSERT_TRUE(result.has_value());

    /* In shadow mode, current graph should NOT change */
    EXPECT_TRUE(shadowOrch.currentGraph().hasVertex("blur"));
    EXPECT_FALSE(shadowOrch.currentGraph().hasVertex("conv"));
}

TEST_F(PipelineOrchestratorTest, test_introspectAsJson) {
    PipelineGraph graph;
    makeVertex(graph, "cam");
    makeVertex(graph, "blur");
    makeShmEdge(graph, "cam", "blur", "/oe.vid.ingest");
    orchestrator_.setCurrentGraph(std::move(graph));

    auto json = orchestrator_.introspectAsJson();
    auto parsed = nlohmann::json::parse(json);

    EXPECT_EQ(parsed["vertex_count"], 2);
    EXPECT_EQ(parsed["edge_count"], 1);
    EXPECT_TRUE(parsed.contains("schema_version"));
}

TEST_F(PipelineOrchestratorTest, test_edgeStateManagement) {
    PipelineGraph graph;
    makeVertex(graph, "cam");
    makeVertex(graph, "blur");
    makeShmEdge(graph, "cam", "blur", "/oe.vid.ingest");
    orchestrator_.setCurrentGraph(std::move(graph));

    /* Advance edge through lifecycle */
    orchestrator_.setEdgeState("cam->blur", EdgeState::kConnecting);
    EXPECT_EQ(orchestrator_.edgeStates().at("cam->blur").current(),
              EdgeState::kConnecting);

    orchestrator_.setEdgeState("cam->blur", EdgeState::kActive);
    EXPECT_EQ(orchestrator_.edgeStates().at("cam->blur").current(),
              EdgeState::kActive);
}

TEST_F(PipelineOrchestratorTest, test_crashHandlerAccess) {
    orchestrator_.crashHandler().recordCrash("cam");
    EXPECT_EQ(orchestrator_.crashHandler().crashCount("cam"), 1);
}
