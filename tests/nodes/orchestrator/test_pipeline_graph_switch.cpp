/* tests/nodes/orchestrator/test_pipeline_graph_switch.cpp
 *
 * Integration tests for the graph-based mode switch pipeline:
 *   GraphBuilder → PipelineOrchestrator::switchToProfile → SwitchPlan
 *
 * Validates the full planning path that executeSwitchPlan() consumes,
 * without needing a live daemon, ZMQ, or GPU.
 */

#include <gtest/gtest.h>

#include "common/oe_logger.hpp"
#include "orchestrator/graph_builder.hpp"
#include "pipeline_orchestrator/pipeline_orchestrator.hpp"


class PipelineGraphSwitchTest : public ::testing::Test {
protected:
    void SetUp() override {
        modules_ = {
            makeDescriptor("video_ingest",       "omniedge_video_ingest",  0,    0),
            makeDescriptor("audio_ingest",       "omniedge_audio_ingest",  0,    5561),
            makeDescriptor("background_blur",    "omniedge_bg_blur",       700,  5563),
            makeDescriptor("websocket_bridge",   "omniedge_ws_bridge",     0,    5570),
            makeDescriptor("conversation_model", "omniedge_conversation",  5500, 5567),
            makeDescriptor("security_camera",    "omniedge_security",      2500, 5569),
            makeDescriptor("sam2",               "omniedge_sam2",          4500, 5565),
            makeDescriptor("beauty",             "omniedge_beauty",        800,  5572),
        };

        /* Non-shadow mode so switchToProfile returns real plans */
        core::pipeline_orchestrator::OrchestratorConfig cfg;
        cfg.shadowMode = false;
        orchestrator_ = std::make_unique<core::pipeline_orchestrator::PipelineOrchestrator>(cfg);
    }

    static ModuleDescriptor makeDescriptor(const std::string& name,
                                           const std::string& binary,
                                           std::size_t vramMb,
                                           int zmqPort) {
        ModuleDescriptor desc;
        desc.name         = name;
        desc.binaryPath   = binary;
        desc.vramBudgetMb = vramMb;
        desc.zmqPubPort   = zmqPort;
        desc.args         = {"--config", "/etc/omniedge/" + name + ".yaml"};
        return desc;
    }

    std::vector<ModuleDescriptor> modules_;
    std::unique_ptr<core::pipeline_orchestrator::PipelineOrchestrator> orchestrator_;
};


TEST_F(PipelineGraphSwitchTest, FirstSwitchSpawnsAllModules)
{
    /* Starting from empty graph, switching to conversation mode should
     * produce a plan that spawns all 5 required modules and evicts none. */
    ModeDefinition mode{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto graphResult = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(graphResult.has_value()) << graphResult.error();

    auto planResult = orchestrator_->switchToProfile(*graphResult);
    ASSERT_TRUE(planResult.has_value());

    const auto& plan = *planResult;
    EXPECT_EQ(plan.verticesToSpawn.size(), 5u);
    EXPECT_TRUE(plan.verticesToEvict.empty());
    EXPECT_TRUE(plan.edgesToDrain.empty());
    EXPECT_TRUE(plan.edgesToConnect.empty());
    EXPECT_TRUE(plan.shmPathsToUnlink.empty());
    EXPECT_FALSE(plan.isEmpty());

    SPDLOG_DEBUG("first switch plan: spawn={}, evict={}",
                 plan.verticesToSpawn.size(), plan.verticesToEvict.size());
}


TEST_F(PipelineGraphSwitchTest, SwitchBetweenModesEvictsAndSpawns)
{
    /* First switch to conversation (5 modules), then to security (6 modules).
     * security adds security_camera but shares the other 5 modules.
     * Diff should show 1 new spawn, 0 evictions (superset). */
    ModeDefinition conversation{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto graph1 = GraphBuilder::buildForMode(conversation, modules_);
    ASSERT_TRUE(graph1.has_value()) << graph1.error();
    auto plan1 = orchestrator_->switchToProfile(*graph1);
    ASSERT_TRUE(plan1.has_value());

    /* Now switch to security — superset of conversation + security_camera */
    ModeDefinition security{
        .name     = "security",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model", "security_camera"},
    };

    auto graph2 = GraphBuilder::buildForMode(security, modules_);
    ASSERT_TRUE(graph2.has_value()) << graph2.error();
    auto plan2 = orchestrator_->switchToProfile(*graph2);
    ASSERT_TRUE(plan2.has_value());

    /* Only security_camera is new; the other 5 are kept */
    EXPECT_EQ(plan2->verticesToSpawn.size(), 1u);
    EXPECT_EQ(plan2->verticesToSpawn[0], "security_camera");
    EXPECT_TRUE(plan2->verticesToEvict.empty());

    SPDLOG_DEBUG("conversation→security: spawn={}, evict={}",
                 plan2->verticesToSpawn.size(), plan2->verticesToEvict.size());
}


TEST_F(PipelineGraphSwitchTest, SwitchToDisjointModeEvictsAll)
{
    /* Switch to conversation, then to a mode that shares no modules.
     * All conversation modules should be evicted, all new ones spawned. */
    ModeDefinition conversation{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto graph1 = GraphBuilder::buildForMode(conversation, modules_);
    ASSERT_TRUE(graph1.has_value());
    auto plan1 = orchestrator_->switchToProfile(*graph1);
    ASSERT_TRUE(plan1.has_value());

    /* sam2 segmentation uses only sam2 — disjoint from conversation */
    ModeDefinition sam2Mode{
        .name     = "sam2_segmentation",
        .required = {"sam2"},
    };

    auto graph2 = GraphBuilder::buildForMode(sam2Mode, modules_);
    ASSERT_TRUE(graph2.has_value());
    auto plan2 = orchestrator_->switchToProfile(*graph2);
    ASSERT_TRUE(plan2.has_value());

    EXPECT_EQ(plan2->verticesToEvict.size(), 5u);
    EXPECT_EQ(plan2->verticesToSpawn.size(), 1u);
    EXPECT_EQ(plan2->verticesToSpawn[0], "sam2");

    SPDLOG_DEBUG("conversation→sam2_segmentation: evict={}, spawn={}",
                 plan2->verticesToEvict.size(), plan2->verticesToSpawn.size());
}


TEST_F(PipelineGraphSwitchTest, SameModeReturnsAlreadyAtTarget)
{
    /* Switching to the same profile twice should return kAlreadyAtTarget. */
    ModeDefinition mode{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto graph1 = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(graph1.has_value());
    auto plan1 = orchestrator_->switchToProfile(*graph1);
    ASSERT_TRUE(plan1.has_value());

    /* Second switch with identical graph */
    auto graph2 = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(graph2.has_value());
    auto plan2 = orchestrator_->switchToProfile(*graph2);

    ASSERT_FALSE(plan2.has_value());
    EXPECT_EQ(plan2.error(),
              core::pipeline_orchestrator::OrchestratorError::kAlreadyAtTarget);

    SPDLOG_DEBUG("same-mode switch correctly returned kAlreadyAtTarget");
}


TEST_F(PipelineGraphSwitchTest, EmptyPlanOnIdenticalSubset)
{
    /* Switch to a superset, then back to a subset that is already loaded.
     * The superset→subset transition should evict extra modules. */
    ModeDefinition security{
        .name     = "security",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model", "security_camera"},
    };

    auto graph1 = GraphBuilder::buildForMode(security, modules_);
    ASSERT_TRUE(graph1.has_value());
    auto plan1 = orchestrator_->switchToProfile(*graph1);
    ASSERT_TRUE(plan1.has_value());

    /* Drop back to conversation — should evict security_camera only */
    ModeDefinition conversation{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto graph2 = GraphBuilder::buildForMode(conversation, modules_);
    ASSERT_TRUE(graph2.has_value());
    auto plan2 = orchestrator_->switchToProfile(*graph2);
    ASSERT_TRUE(plan2.has_value());

    EXPECT_EQ(plan2->verticesToEvict.size(), 1u);
    EXPECT_EQ(plan2->verticesToEvict[0], "security_camera");
    EXPECT_TRUE(plan2->verticesToSpawn.empty());

    SPDLOG_DEBUG("security→conversation: evict={}, spawn={}",
                 plan2->verticesToEvict.size(), plan2->verticesToSpawn.size());
}


TEST_F(PipelineGraphSwitchTest, SwitchPlanConfigPassedThrough)
{
    /* Verify that SwitchPlanConfig values from the orchestrator config
     * are reflected in the generated SwitchPlan. */
    core::pipeline_orchestrator::OrchestratorConfig cfg;
    cfg.shadowMode = false;
    cfg.switchPlan.drainTimeoutMs      = 3000;
    cfg.switchPlan.connectRetryCount   = 5;
    cfg.switchPlan.connectRetryIntervalMs = 250;

    auto customOrchestrator =
        std::make_unique<core::pipeline_orchestrator::PipelineOrchestrator>(cfg);

    ModeDefinition mode{
        .name     = "conversation",
        .required = {"video_ingest", "conversation_model"},
    };

    auto graph = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(graph.has_value());
    auto plan = customOrchestrator->switchToProfile(*graph);
    ASSERT_TRUE(plan.has_value());

    EXPECT_EQ(plan->config.drainTimeoutMs, 3000u);
    EXPECT_EQ(plan->config.connectRetryCount, 5u);
    EXPECT_EQ(plan->config.connectRetryIntervalMs, 250u);

    SPDLOG_DEBUG("plan config: drain={}ms, retries={}, interval={}ms",
                 plan->config.drainTimeoutMs,
                 plan->config.connectRetryCount,
                 plan->config.connectRetryIntervalMs);
}
