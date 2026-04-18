/* tests/core/pipeline_orchestrator/test_switch_plan.cpp — SwitchPlan build */

#include <gtest/gtest.h>
#include "pipeline_orchestrator/switch_plan.hpp"

#include <algorithm>

using namespace core::graph;
using namespace core::pipeline_orchestrator;

class SwitchPlanTest : public ::testing::Test {
protected:
    PipelineGraph current_;
    PipelineGraph target_;

    void makeVertex(PipelineGraph& g, const std::string& id) {
        g.addVertex({.id = id, .binaryName = "bin_" + id});
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

    bool comesBefore(const std::vector<std::string>& vec,
                     const std::string& a, const std::string& b) {
        auto posA = std::find(vec.begin(), vec.end(), a);
        auto posB = std::find(vec.begin(), vec.end(), b);
        return posA < posB;
    }
};

TEST_F(SwitchPlanTest, test_emptyDiff) {
    GraphDiff diff;
    auto plan = buildSwitchPlan(current_, target_, diff);
    EXPECT_TRUE(plan.isEmpty());
}

TEST_F(SwitchPlanTest, test_fullSwitch) {
    /* current: cam → blur */
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");

    /* target: conv → tts */
    makeVertex(target_, "conv");
    makeVertex(target_, "tts");
    makeShmEdge(target_, "conv", "tts", "/oe.nlp.llm");

    auto diff = computeDiff(current_, target_);
    auto plan = buildSwitchPlan(current_, target_, diff);

    EXPECT_FALSE(plan.isEmpty());

    /* Drain: the old edge */
    EXPECT_EQ(plan.edgesToDrain.size(), 1);

    /* Evict: cam and blur in reverse-topo order (blur before cam) */
    EXPECT_EQ(plan.verticesToEvict.size(), 2);
    EXPECT_TRUE(comesBefore(plan.verticesToEvict, "blur", "cam"));

    /* Unlink: old SHM path */
    EXPECT_EQ(plan.shmPathsToUnlink.size(), 1);

    /* Spawn: conv and tts in topo order (conv before tts) */
    EXPECT_EQ(plan.verticesToSpawn.size(), 2);
    EXPECT_TRUE(comesBefore(plan.verticesToSpawn, "conv", "tts"));

    /* Connect: new edge */
    EXPECT_EQ(plan.edgesToConnect.size(), 1);
}

TEST_F(SwitchPlanTest, test_partialSwitch) {
    /* current: cam → blur → bridge */
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeVertex(current_, "bridge");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");
    makeShmEdge(current_, "blur", "bridge", "/oe.cv.blur");

    /* target: cam → conv → bridge (blur replaced by conv) */
    makeVertex(target_, "cam");
    makeVertex(target_, "conv");
    makeVertex(target_, "bridge");
    makeShmEdge(target_, "cam", "conv", "/oe.vid.ingest");
    makeShmEdge(target_, "conv", "bridge", "/oe.nlp.llm");

    auto diff = computeDiff(current_, target_);
    auto plan = buildSwitchPlan(current_, target_, diff);

    /* Only blur evicted */
    EXPECT_EQ(plan.verticesToEvict.size(), 1);
    EXPECT_EQ(plan.verticesToEvict[0], "blur");

    /* Only conv spawned */
    EXPECT_EQ(plan.verticesToSpawn.size(), 1);
    EXPECT_EQ(plan.verticesToSpawn[0], "conv");
}

TEST_F(SwitchPlanTest, test_configPropagation) {
    SwitchPlanConfig config{.drainTimeoutMs = 10000,
                            .connectRetryCount = 5,
                            .connectRetryIntervalMs = 200};
    GraphDiff diff;
    auto plan = buildSwitchPlan(current_, target_, diff, config);
    EXPECT_EQ(plan.config.drainTimeoutMs, 10000);
    EXPECT_EQ(plan.config.connectRetryCount, 5);
    EXPECT_EQ(plan.config.connectRetryIntervalMs, 200);
}

TEST_F(SwitchPlanTest, test_evictionOrderReverseTopological) {
    /* current: a → b → c → d (linear chain) */
    makeVertex(current_, "a");
    makeVertex(current_, "b");
    makeVertex(current_, "c");
    makeVertex(current_, "d");
    makeShmEdge(current_, "a", "b", "/oe.1");
    makeShmEdge(current_, "b", "c", "/oe.2");
    makeShmEdge(current_, "c", "d", "/oe.3");

    /* target: empty — evict all */
    GraphDiff diff;
    diff.verticesToEvict = {"a", "b", "c", "d"};
    diff.edgesToRemove = {"a->b", "b->c", "c->d"};
    diff.shmPathsToUnlink = {"/oe.1", "/oe.2", "/oe.3"};

    auto plan = buildSwitchPlan(current_, target_, diff);

    /* Eviction must be reverse-topo: d, c, b, a */
    ASSERT_EQ(plan.verticesToEvict.size(), 4);
    EXPECT_TRUE(comesBefore(plan.verticesToEvict, "d", "c"));
    EXPECT_TRUE(comesBefore(plan.verticesToEvict, "c", "b"));
    EXPECT_TRUE(comesBefore(plan.verticesToEvict, "b", "a"));
}
