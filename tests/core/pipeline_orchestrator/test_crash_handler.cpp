/* tests/core/pipeline_orchestrator/test_crash_handler.cpp — Crash recovery */

#include <gtest/gtest.h>
#include "pipeline_orchestrator/crash_handler.hpp"

#include <algorithm>
#include <unordered_set>

using namespace core::graph;
using namespace core::pipeline_orchestrator;

class CrashHandlerTest : public ::testing::Test {
protected:
    PipelineGraph graph_;
    CrashHandler handler_;

    void makeVertex(const std::string& id) {
        graph_.addVertex({.id = id, .binaryName = "bin_" + id});
    }

    void makeShmEdge(const std::string& from, const std::string& to,
                     const std::string& shmPath) {
        graph_.addEdge({
            .id = from + "->" + to,
            .fromVertex = from,
            .toVertex = to,
            .transport = TransportType::kShm,
            .attributes = ShmAttributes{.shmPath = shmPath}
        });
    }

    bool contains(const std::vector<std::string>& vec, const std::string& val) {
        return std::find(vec.begin(), vec.end(), val) != vec.end();
    }
};

TEST_F(CrashHandlerTest, test_sourceVertexCrash) {
    /* cam → blur → bridge */
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("bridge");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");
    makeShmEdge("blur", "bridge", "/oe.cv.blur");

    auto result = handler_.analyzeImpact(graph_, "cam");
    ASSERT_TRUE(result.has_value());

    /* cam crash affects blur and bridge (all descendants) */
    EXPECT_EQ(result->crashedVertex, "cam");
    EXPECT_EQ(result->affectedDescendants.size(), 2);
    EXPECT_TRUE(contains(result->affectedDescendants, "blur"));
    EXPECT_TRUE(contains(result->affectedDescendants, "bridge"));

    /* All edges are disconnected */
    EXPECT_EQ(result->disconnectedEdges.size(), 2);

    /* Both SHM paths need cleanup */
    EXPECT_EQ(result->shmPathsToCleanup.size(), 2);
}

TEST_F(CrashHandlerTest, test_leafVertexCrash) {
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    auto result = handler_.analyzeImpact(graph_, "blur");
    ASSERT_TRUE(result.has_value());

    /* blur has no descendants */
    EXPECT_TRUE(result->affectedDescendants.empty());

    /* Only the incoming edge is disconnected */
    EXPECT_EQ(result->disconnectedEdges.size(), 1);
    EXPECT_EQ(result->disconnectedEdges[0], "cam->blur");
}

TEST_F(CrashHandlerTest, test_unknownVertex) {
    auto result = handler_.analyzeImpact(graph_, "nonexistent");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CrashHandlerError::kVertexNotFound);
}

TEST_F(CrashHandlerTest, test_crashCountTracking) {
    EXPECT_EQ(handler_.crashCount("cam"), 0);
    EXPECT_FALSE(handler_.isDisabled("cam"));

    handler_.recordCrash("cam");
    EXPECT_EQ(handler_.crashCount("cam"), 1);

    handler_.recordCrash("cam");
    EXPECT_EQ(handler_.crashCount("cam"), 2);
}

TEST_F(CrashHandlerTest, test_permanentDisable) {
    CrashHandler handler(CrashHandlerConfig{.maxCrashesBeforeDisable = 3});

    handler.recordCrash("cam");
    handler.recordCrash("cam");
    EXPECT_FALSE(handler.isDisabled("cam"));

    bool disabled = handler.recordCrash("cam");
    EXPECT_TRUE(disabled);
    EXPECT_TRUE(handler.isDisabled("cam"));
}

TEST_F(CrashHandlerTest, test_disabledVertexRejected) {
    CrashHandler handler(CrashHandlerConfig{.maxCrashesBeforeDisable = 1});
    makeVertex("cam");

    handler.recordCrash("cam");
    EXPECT_TRUE(handler.isDisabled("cam"));

    auto result = handler.analyzeImpact(graph_, "cam");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), CrashHandlerError::kPermanentlyDisabled);
}

TEST_F(CrashHandlerTest, test_resetCrashCount) {
    handler_.recordCrash("cam");
    handler_.recordCrash("cam");
    EXPECT_EQ(handler_.crashCount("cam"), 2);

    handler_.resetCrashCount("cam");
    EXPECT_EQ(handler_.crashCount("cam"), 0);
    EXPECT_FALSE(handler_.isDisabled("cam"));
}

TEST_F(CrashHandlerTest, test_applyEdgeTransitions) {
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    std::unordered_map<std::string, EdgeStateMachine> edgeStates;
    edgeStates.emplace("cam->blur", EdgeStateMachine{"cam->blur"});
    (void)edgeStates.at("cam->blur").beginConnecting();
    (void)edgeStates.at("cam->blur").confirmActive();

    EXPECT_EQ(edgeStates.at("cam->blur").current(), EdgeState::kActive);

    CrashImpact impact;
    impact.crashedVertex = "cam";
    impact.disconnectedEdges = {"cam->blur"};

    handler_.applyEdgeTransitions(impact, edgeStates);

    /* forceReset should move edge to kIdle */
    EXPECT_EQ(edgeStates.at("cam->blur").current(), EdgeState::kIdle);
}
