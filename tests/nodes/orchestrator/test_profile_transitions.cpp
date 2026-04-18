/* tests/nodes/orchestrator/test_profile_transitions.cpp
 *
 * Comprehensive profile transition tests for the graph-based pipeline
 * orchestrator. Validates ALL 5 processing profiles with production VRAM
 * values, the full 5x5 transition matrix, multi-step chains, cross-profile
 * diff correctness, and benchmarks graph operation latency.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/oe_logger.hpp"
#include "orchestrator/graph_builder.hpp"
#include "graph/graph_diff.hpp"
#include "graph/graph_topology.hpp"
#include "graph/graph_validator.hpp"
#include "graph/pipeline_edge.hpp"
#include "pipeline_orchestrator/pipeline_orchestrator.hpp"


/* ── Fixture ──────────────────────────────────────────────────────────── */

class ProfileTransitionsTest : public ::testing::Test {
protected:
    static constexpr std::array<const char*, 5> kProfileNames = {
        "conversation", "sam2_segmentation", "vision_model", "security", "beauty"
    };

    void SetUp() override {
        /* 11 module descriptors with production VRAM values from omniedge.ini */
        modules_ = {
            makeDesc("video_ingest",       "omniedge_video_ingest",    0,    0),
            makeDesc("audio_ingest",       "omniedge_audio_ingest",    0,    5561),
            makeDesc("background_blur",    "omniedge_bg_blur",         250,  5563),
            makeDesc("websocket_bridge",   "omniedge_ws_bridge",       0,    5570),
            makeDesc("conversation_model", "omniedge_conversation",    7500, 5567),
            makeDesc("security_camera",    "omniedge_security",        250,  5569),
            makeDesc("sam2",               "omniedge_sam2",            800,  5565),
            makeDesc("beauty",             "omniedge_beauty",          250,  5572),
            makeDesc("face_recognition",   "omniedge_face_recog",      350,  5562),
            makeDesc("tts",                "omniedge_tts",             100,  5568),
            makeDesc("audio_denoise",      "omniedge_audio_denoise",   50,   5566),
        };

        /* 5 mode definitions matching omniedge.ini */
        modes_["conversation"] = ModeDefinition{
            .name = "conversation",
            .required = {"video_ingest", "audio_ingest", "background_blur",
                         "websocket_bridge", "conversation_model"},
        };
        modes_["sam2_segmentation"] = ModeDefinition{
            .name = "sam2_segmentation",
            .required = {"sam2", "background_blur",
                         "video_ingest", "websocket_bridge"},
        };
        modes_["vision_model"] = ModeDefinition{
            .name = "vision_model",
            .required = {"conversation_model", "face_recognition",
                         "video_ingest", "websocket_bridge"},
        };
        modes_["security"] = ModeDefinition{
            .name = "security",
            .required = {"security_camera", "background_blur",
                         "video_ingest", "audio_ingest", "websocket_bridge"},
        };
        modes_["beauty"] = ModeDefinition{
            .name = "beauty",
            .required = {"beauty", "background_blur", "face_recognition",
                         "video_ingest", "audio_ingest", "websocket_bridge"},
        };

        /* Pre-build all 5 graphs */
        for (const auto& [name, mode] : modes_) {
            auto result = GraphBuilder::buildForMode(mode, modules_);
            ASSERT_TRUE(result.has_value())
                << "Failed to build graph for mode '" << name << "': " << result.error();
            graphs_[name] = std::move(*result);
        }

        resetOrchestrator();
    }

    void resetOrchestrator() {
        core::pipeline_orchestrator::OrchestratorConfig cfg;
        cfg.shadowMode = false;
        cfg.vramBudgetMiB = 11264;
        orchestrator_ = std::make_unique<
            core::pipeline_orchestrator::PipelineOrchestrator>(cfg);
    }

    static ModuleDescriptor makeDesc(const std::string& name,
                                     const std::string& binary,
                                     std::size_t vramMb, int zmqPort) {
        ModuleDescriptor desc;
        desc.name         = name;
        desc.binaryPath   = binary;
        desc.vramBudgetMb = vramMb;
        desc.zmqPubPort   = zmqPort;
        desc.args         = {"--config", "/etc/omniedge/" + name + ".yaml"};
        return desc;
    }

    [[nodiscard]] std::size_t totalVram(
        const core::graph::PipelineGraph& graph) const {
        std::size_t sum = 0;
        for (const auto& id : graph.allVertexIds()) {
            auto v = graph.getVertex(id);
            if (v.has_value()) sum += (*v)->vramBudgetMiB;
        }
        return sum;
    }

    /* Compute set difference: a - b */
    [[nodiscard]] static std::unordered_set<std::string> setDifference(
        const std::unordered_set<std::string>& a,
        const std::unordered_set<std::string>& b) {
        std::unordered_set<std::string> result;
        for (const auto& item : a) {
            if (b.count(item) == 0) result.insert(item);
        }
        return result;
    }

    /* Compute set intersection: a & b */
    [[nodiscard]] static std::unordered_set<std::string> setIntersection(
        const std::unordered_set<std::string>& a,
        const std::unordered_set<std::string>& b) {
        std::unordered_set<std::string> result;
        for (const auto& item : a) {
            if (b.count(item) > 0) result.insert(item);
        }
        return result;
    }

    std::vector<ModuleDescriptor> modules_;
    std::map<std::string, ModeDefinition> modes_;
    std::map<std::string, core::graph::PipelineGraph> graphs_;
    std::unique_ptr<core::pipeline_orchestrator::PipelineOrchestrator> orchestrator_;
};


/* ── A. Profile Build Validation ─────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, AllProfilesBuildSuccessfully)
{
    ASSERT_EQ(graphs_.size(), 5u);

    for (const auto& [name, graph] : graphs_) {
        SCOPED_TRACE(name);
        const auto& mode = modes_.at(name);

        EXPECT_EQ(graph.vertexCount(), mode.required.size())
            << "mode '" << name << "' expected " << mode.required.size()
            << " vertices, got " << graph.vertexCount();

        for (const auto& moduleName : mode.required) {
            EXPECT_TRUE(graph.hasVertex(moduleName))
                << "mode '" << name << "' missing vertex: " << moduleName;
        }

        EXPECT_EQ(graph.edgeCount(), 0u)
            << "Phase 4.5: no edges expected";

        SPDLOG_DEBUG("profile '{}': {} vertices, VRAM={}MiB",
                     name, graph.vertexCount(), totalVram(graph));
    }
}

TEST_F(ProfileTransitionsTest, AllProfilesUnderVramBudget)
{
    /* Expected VRAM totals per profile (from production config) */
    const std::map<std::string, std::size_t> expectedVram = {
        {"conversation",     7750},   /* 0+0+250+0+7500 */
        {"sam2_segmentation", 1050},  /* 800+250+0+0 */
        {"vision_model",     7850},   /* 7500+350+0+0 */
        {"security",          500},   /* 250+250+0+0+0 */
        {"beauty",            850},   /* 250+250+350+0+0+0 */
    };

    for (const auto& [name, graph] : graphs_) {
        SCOPED_TRACE(name);

        /* Validate VRAM budget via graph validator (NOT validateAll — see plan) */
        auto vramResult = core::graph::validator::validateVramBudget(graph, 11264);
        EXPECT_TRUE(vramResult.has_value())
            << "VRAM budget exceeded for profile '" << name << "'";

        std::size_t actual = totalVram(graph);
        EXPECT_EQ(actual, expectedVram.at(name))
            << "VRAM mismatch for '" << name << "'";
        EXPECT_LE(actual, 11264u)
            << "profile '" << name << "' VRAM " << actual
            << " MiB exceeds 11264 MiB budget";

        SPDLOG_DEBUG("profile '{}': VRAM={}MiB (budget=11264MiB, headroom={}MiB)",
                     name, actual, 11264 - actual);
    }
}


/* ── B. Full Transition Matrix ───────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, FullTransitionMatrix)
{
    int transitions = 0;
    int alreadyAtTarget = 0;

    for (const auto& srcName : kProfileNames) {
        for (const auto& tgtName : kProfileNames) {
            SCOPED_TRACE(std::string(srcName) + " -> " + tgtName);
            resetOrchestrator();

            /* First switch: empty -> src */
            auto plan1 = orchestrator_->switchToProfile(graphs_.at(srcName));
            ASSERT_TRUE(plan1.has_value())
                << "empty -> " << srcName << " failed";

            /* Second switch: src -> tgt */
            auto plan2 = orchestrator_->switchToProfile(graphs_.at(tgtName));

            if (std::string(srcName) == std::string(tgtName)) {
                /* Same profile: expect kAlreadyAtTarget */
                ASSERT_FALSE(plan2.has_value());
                EXPECT_EQ(plan2.error(),
                    core::pipeline_orchestrator::OrchestratorError::kAlreadyAtTarget);
                ++alreadyAtTarget;
            } else {
                /* Different profiles: expect valid plan */
                ASSERT_TRUE(plan2.has_value())
                    << srcName << " -> " << tgtName << " produced error";

                auto expectedEvict = setDifference(
                    modes_.at(srcName).required, modes_.at(tgtName).required);
                auto expectedSpawn = setDifference(
                    modes_.at(tgtName).required, modes_.at(srcName).required);

                /* Verify evict set matches */
                std::unordered_set<std::string> actualEvict(
                    plan2->verticesToEvict.begin(), plan2->verticesToEvict.end());
                EXPECT_EQ(actualEvict, expectedEvict)
                    << srcName << " -> " << tgtName << " eviction mismatch";

                /* Verify spawn set matches */
                std::unordered_set<std::string> actualSpawn(
                    plan2->verticesToSpawn.begin(), plan2->verticesToSpawn.end());
                EXPECT_EQ(actualSpawn, expectedSpawn)
                    << srcName << " -> " << tgtName << " spawn mismatch";

                /* Phase 4.5: edge phases must be empty */
                EXPECT_TRUE(plan2->edgesToDrain.empty());
                EXPECT_TRUE(plan2->edgesToConnect.empty());
                EXPECT_TRUE(plan2->shmPathsToUnlink.empty());
            }
            ++transitions;
        }
    }

    EXPECT_EQ(transitions, 25);
    EXPECT_EQ(alreadyAtTarget, 5);
    SPDLOG_DEBUG("transition matrix: {}/25 tested, {} same-mode no-ops",
                 transitions, alreadyAtTarget);
}


/* ── C. Module Overlap Analysis ──────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, SharedModulesAreKeptNotRespawned)
{
    struct TransitionCase {
        const char* src;
        const char* tgt;
        std::unordered_set<std::string> expectedShared;
        std::unordered_set<std::string> expectedEvict;
        std::unordered_set<std::string> expectedSpawn;
    };

    std::vector<TransitionCase> cases = {
        {"conversation", "security",
         {"video_ingest", "audio_ingest", "background_blur", "websocket_bridge"},
         {"conversation_model"},
         {"security_camera"}},

        {"sam2_segmentation", "beauty",
         {"background_blur", "video_ingest", "websocket_bridge"},
         {"sam2"},
         {"beauty", "face_recognition", "audio_ingest"}},
    };

    for (const auto& tc : cases) {
        SCOPED_TRACE(std::string(tc.src) + " -> " + tc.tgt);
        resetOrchestrator();

        auto plan1 = orchestrator_->switchToProfile(graphs_.at(tc.src));
        ASSERT_TRUE(plan1.has_value());

        auto plan2 = orchestrator_->switchToProfile(graphs_.at(tc.tgt));
        ASSERT_TRUE(plan2.has_value());

        /* Shared modules must NOT appear in evict or spawn */
        for (const auto& shared : tc.expectedShared) {
            EXPECT_EQ(std::count(plan2->verticesToEvict.begin(),
                                 plan2->verticesToEvict.end(), shared), 0)
                << shared << " should not be evicted";
            EXPECT_EQ(std::count(plan2->verticesToSpawn.begin(),
                                 plan2->verticesToSpawn.end(), shared), 0)
                << shared << " should not be respawned";
        }

        /* Verify exact evict and spawn sets */
        std::unordered_set<std::string> actualEvict(
            plan2->verticesToEvict.begin(), plan2->verticesToEvict.end());
        std::unordered_set<std::string> actualSpawn(
            plan2->verticesToSpawn.begin(), plan2->verticesToSpawn.end());
        EXPECT_EQ(actualEvict, tc.expectedEvict);
        EXPECT_EQ(actualSpawn, tc.expectedSpawn);

        SPDLOG_DEBUG("{} -> {}: shared={}, evict={}, spawn={}",
                     tc.src, tc.tgt, tc.expectedShared.size(),
                     plan2->verticesToEvict.size(), plan2->verticesToSpawn.size());
    }
}


/* ── D. Multi-Step Chains ────────────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, MultiStepChainConvSecurityBeautyConv)
{
    /* empty -> conversation: spawn 5 */
    auto plan1 = orchestrator_->switchToProfile(graphs_.at("conversation"));
    ASSERT_TRUE(plan1.has_value());
    EXPECT_EQ(plan1->verticesToSpawn.size(), 5u);
    EXPECT_TRUE(plan1->verticesToEvict.empty());

    /* conversation -> security: evict conversation_model, spawn security_camera */
    auto plan2 = orchestrator_->switchToProfile(graphs_.at("security"));
    ASSERT_TRUE(plan2.has_value());
    EXPECT_EQ(plan2->verticesToEvict.size(), 1u);
    EXPECT_EQ(plan2->verticesToSpawn.size(), 1u);

    /* security -> beauty: evict security_camera, spawn beauty + face_recognition */
    auto plan3 = orchestrator_->switchToProfile(graphs_.at("beauty"));
    ASSERT_TRUE(plan3.has_value());
    EXPECT_EQ(plan3->verticesToEvict.size(), 1u);
    EXPECT_EQ(plan3->verticesToSpawn.size(), 2u);

    /* beauty -> conversation: evict beauty + face_recognition, spawn conversation_model */
    auto plan4 = orchestrator_->switchToProfile(graphs_.at("conversation"));
    ASSERT_TRUE(plan4.has_value());
    EXPECT_EQ(plan4->verticesToEvict.size(), 2u);
    EXPECT_EQ(plan4->verticesToSpawn.size(), 1u);

    /* Verify final graph matches conversation profile */
    const auto& finalGraph = orchestrator_->currentGraph();
    EXPECT_EQ(finalGraph.vertexCount(), 5u);
    for (const auto& name : modes_.at("conversation").required) {
        EXPECT_TRUE(finalGraph.hasVertex(name))
            << "after round-trip, missing: " << name;
    }

    SPDLOG_DEBUG("chain complete: 4 transitions, final graph has {} vertices",
                 finalGraph.vertexCount());
}

TEST_F(ProfileTransitionsTest, MultiStepChainAllProfilesSequential)
{
    for (const auto& name : kProfileNames) {
        SCOPED_TRACE(name);
        auto plan = orchestrator_->switchToProfile(graphs_.at(name));
        ASSERT_TRUE(plan.has_value()) << "failed switching to " << name;

        EXPECT_EQ(orchestrator_->currentGraph().vertexCount(),
                  modes_.at(name).required.size())
            << "vertex count mismatch after switching to " << name;
    }

    SPDLOG_DEBUG("sequential chain through all 5 profiles complete");
}


/* ── E. Cross-Profile Diff Validation ────────────────────────────────── */

TEST_F(ProfileTransitionsTest, CrossProfileDiffMatrix)
{
    int pairs = 0;

    for (const auto& [nameA, graphA] : graphs_) {
        for (const auto& [nameB, graphB] : graphs_) {
            if (nameA == nameB) continue;
            SCOPED_TRACE(nameA + " -> " + nameB);

            auto diff = core::graph::computeDiff(graphA, graphB);

            /* Partition invariant: evict + kept = A's vertices */
            std::unordered_set<std::string> fromA;
            for (const auto& id : graphA.allVertexIds()) fromA.insert(id);

            std::unordered_set<std::string> evictPlusKept;
            for (const auto& id : diff.verticesToEvict) evictPlusKept.insert(id);
            for (const auto& id : diff.verticesKept)    evictPlusKept.insert(id);
            EXPECT_EQ(evictPlusKept, fromA)
                << "evict + kept != A for " << nameA << " -> " << nameB;

            /* Partition invariant: spawn + kept = B's vertices */
            std::unordered_set<std::string> fromB;
            for (const auto& id : graphB.allVertexIds()) fromB.insert(id);

            std::unordered_set<std::string> spawnPlusKept;
            for (const auto& id : diff.verticesToSpawn) spawnPlusKept.insert(id);
            for (const auto& id : diff.verticesKept)    spawnPlusKept.insert(id);
            EXPECT_EQ(spawnPlusKept, fromB)
                << "spawn + kept != B for " << nameA << " -> " << nameB;

            /* kept = intersection(A, B) */
            auto expectedKept = setIntersection(
                modes_.at(nameA).required, modes_.at(nameB).required);
            std::unordered_set<std::string> actualKept(
                diff.verticesKept.begin(), diff.verticesKept.end());
            EXPECT_EQ(actualKept, expectedKept)
                << "kept mismatch for " << nameA << " -> " << nameB;

            /* Phase 4.5: no edge diffs */
            EXPECT_TRUE(diff.edgesToRemove.empty());
            EXPECT_TRUE(diff.edgesToAdd.empty());
            EXPECT_TRUE(diff.edgesKept.empty());
            EXPECT_TRUE(diff.shmPathsToUnlink.empty());

            ++pairs;
        }
    }

    EXPECT_EQ(pairs, 20);  /* 5 * 4 */
    SPDLOG_DEBUG("cross-profile diff matrix: {}/20 pairs validated", pairs);
}


/* ── F. Topo Sort Documentation ──────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, EvictionOrderIsReverseTopological)
{
    /* Build a linear graph A -> B -> C to verify reverse-topo eviction.
     * NOTE: evictPriority-sorted eviction is in executeSwitchPlan
     * (omniedge_daemon.cpp), not in buildSwitchPlan. This test only
     * validates the SwitchPlan layer's reverse-topological ordering. */
    core::graph::PipelineGraph current;
    current.addVertex({.id = "a", .binaryName = "bin_a"});
    current.addVertex({.id = "b", .binaryName = "bin_b"});
    current.addVertex({.id = "c", .binaryName = "bin_c"});
    current.addEdge({.id = "a_to_b", .fromVertex = "a", .toVertex = "b",
                     .transport = core::graph::TransportType::kZmq,
                     .attributes = core::graph::ZmqAttributes{.port = 9000}});
    current.addEdge({.id = "b_to_c", .fromVertex = "b", .toVertex = "c",
                     .transport = core::graph::TransportType::kZmq,
                     .attributes = core::graph::ZmqAttributes{.port = 9001}});

    core::graph::PipelineGraph empty;
    auto diff = core::graph::computeDiff(current, empty);
    auto plan = core::pipeline_orchestrator::buildSwitchPlan(
        current, empty, diff);

    /* Reverse-topo of A->B->C is C, B, A */
    ASSERT_EQ(plan.verticesToEvict.size(), 3u);
    EXPECT_EQ(plan.verticesToEvict[0], "c");
    EXPECT_EQ(plan.verticesToEvict[1], "b");
    EXPECT_EQ(plan.verticesToEvict[2], "a");

    SPDLOG_DEBUG("reverse-topo eviction order: {}, {}, {}",
                 plan.verticesToEvict[0], plan.verticesToEvict[1],
                 plan.verticesToEvict[2]);
}


/* ── G. Benchmarks ───────────────────────────────────────────────────── */

TEST_F(ProfileTransitionsTest, BenchmarkConversationSecurityRoundTrips)
{
    constexpr int kWarmup   = 10;
    constexpr int kMeasured = 100;

    /* Warmup */
    for (int i = 0; i < kWarmup; ++i) {
        resetOrchestrator();
        orchestrator_->switchToProfile(graphs_.at("conversation"));
        orchestrator_->switchToProfile(graphs_.at("security"));
    }

    /* Measured */
    std::vector<double> latenciesMs;
    latenciesMs.reserve(kMeasured);

    for (int i = 0; i < kMeasured; ++i) {
        resetOrchestrator();
        const auto start = std::chrono::steady_clock::now();
        orchestrator_->switchToProfile(graphs_.at("conversation"));
        orchestrator_->switchToProfile(graphs_.at("security"));
        const auto end = std::chrono::steady_clock::now();

        latenciesMs.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::ranges::sort(latenciesMs);
    const double mean = std::accumulate(
        latenciesMs.begin(), latenciesMs.end(), 0.0) / kMeasured;
    const double p99 = latenciesMs[static_cast<std::size_t>(kMeasured * 0.99)];
    const double total = std::accumulate(
        latenciesMs.begin(), latenciesMs.end(), 0.0);

    SPDLOG_INFO("benchmark_round_trips: n={}, mean={:.3f}ms, p99={:.3f}ms, "
                "total={:.1f}ms",
                kMeasured, mean, p99, total);

    RecordProperty("mean_ms", std::to_string(mean));
    RecordProperty("p99_ms", std::to_string(p99));

    /* Sanity: pure in-memory graph ops should be well under 1ms */
    EXPECT_LT(mean, 1.0)
        << "Mean round-trip latency " << mean << "ms exceeds 1ms threshold";
}

TEST_F(ProfileTransitionsTest, BenchmarkDiffAndPlanBuild)
{
    constexpr int kIterations = 1000;

    const auto& conv = graphs_.at("conversation");
    const auto& sec  = graphs_.at("security");

    /* Warmup */
    for (int i = 0; i < 50; ++i) {
        auto diff = core::graph::computeDiff(conv, sec);
        auto plan = core::pipeline_orchestrator::buildSwitchPlan(
            conv, sec, diff);
        (void)plan;
    }

    /* Measured */
    std::vector<double> latenciesUs;
    latenciesUs.reserve(kIterations);

    for (int i = 0; i < kIterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto diff = core::graph::computeDiff(conv, sec);
        auto plan = core::pipeline_orchestrator::buildSwitchPlan(
            conv, sec, diff);
        const auto end = std::chrono::steady_clock::now();
        (void)plan;

        latenciesUs.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::ranges::sort(latenciesUs);
    const double mean = std::accumulate(
        latenciesUs.begin(), latenciesUs.end(), 0.0) / kIterations;
    const double p99 = latenciesUs[static_cast<std::size_t>(kIterations * 0.99)];

    SPDLOG_INFO("benchmark_diff_plan: n={}, mean={:.1f}us, p99={:.1f}us",
                kIterations, mean, p99);

    RecordProperty("diff_plan_mean_us", std::to_string(mean));
    RecordProperty("diff_plan_p99_us", std::to_string(p99));

    /* Sanity: diff + plan build should be well under 100us */
    EXPECT_LT(mean, 100.0)
        << "Mean diff+plan latency " << mean << "us exceeds 100us threshold";
}
