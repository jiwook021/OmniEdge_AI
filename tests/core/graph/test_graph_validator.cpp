/* tests/core/graph/test_graph_validator.cpp — Graph validation suite */

#include <gtest/gtest.h>
#include "graph/graph_validator.hpp"

using namespace core::graph;
using namespace core::graph::validator;

class GraphValidatorTest : public ::testing::Test {
protected:
    PipelineGraph graph_;

    void makeVertex(const std::string& id, std::size_t vram = 0) {
        graph_.addVertex({.id = id, .binaryName = "bin_" + id, .vramBudgetMiB = vram});
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
};

TEST_F(GraphValidatorTest, test_validDag) {
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    EXPECT_TRUE(validateDag(graph_).has_value());
}

TEST_F(GraphValidatorTest, test_cycleRejected) {
    makeVertex("a");
    makeVertex("b");
    makeVertex("c");
    graph_.addEdge({.id = "a->b", .fromVertex = "a", .toVertex = "b"});
    graph_.addEdge({.id = "b->c", .fromVertex = "b", .toVertex = "c"});
    graph_.addEdge({.id = "c->a", .fromVertex = "c", .toVertex = "a"});

    auto result = validateDag(graph_);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ValidationError::kCycleDetected);
}

TEST_F(GraphValidatorTest, test_connectedSources) {
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    EXPECT_TRUE(validateConnectedSources(graph_).has_value());
}

TEST_F(GraphValidatorTest, test_disconnectedVertex) {
    /* Mixed graph: a connected pair plus one isolated vertex.
       Edgeless graphs are valid by design in Phase 4.5, so the orphan
       check only applies when the graph has at least one edge. */
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");
    makeVertex("orphan");

    auto result = validateConnectedSources(graph_);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ValidationError::kDisconnectedVertex);
}

TEST_F(GraphValidatorTest, test_sourceVertexAllowed) {
    /* A source vertex (no incoming, has outgoing) is fine */
    makeVertex("cam");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    EXPECT_TRUE(validateConnectedSources(graph_).has_value());
}

TEST_F(GraphValidatorTest, test_vramBudgetPass) {
    makeVertex("conv", 7500);
    makeVertex("blur", 250);
    makeShmEdge("conv", "blur", "/oe.test");

    EXPECT_TRUE(validateVramBudget(graph_, 11264).has_value());
}

TEST_F(GraphValidatorTest, test_vramBudgetExceeded) {
    makeVertex("conv", 7500);
    makeVertex("sr", 4000);
    makeShmEdge("conv", "sr", "/oe.test");

    auto result = validateVramBudget(graph_, 11264);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ValidationError::kVramBudgetExceeded);
}

TEST_F(GraphValidatorTest, test_shmPathUniqueSameProducer) {
    /* Same producer writing to same path on multiple edges — OK (fan-out) */
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("face");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");
    makeShmEdge("cam", "face", "/oe.vid.ingest");

    EXPECT_TRUE(validateShmPathUniqueness(graph_).has_value());
}

TEST_F(GraphValidatorTest, test_shmPathDuplicateProducer) {
    /* Two different producers write to the same SHM path — BAD */
    makeVertex("cam");
    makeVertex("screen");
    makeVertex("blur");
    makeShmEdge("cam", "blur", "/oe.vid.ingest");
    makeShmEdge("screen", "blur", "/oe.vid.ingest");

    auto result = validateShmPathUniqueness(graph_);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ValidationError::kDuplicateShmProducer);
}

TEST_F(GraphValidatorTest, test_validateAllFirstFailure) {
    /* Create graph with cycle — validateAll should fail on DAG check */
    makeVertex("a");
    makeVertex("b");
    graph_.addEdge({.id = "a->b", .fromVertex = "a", .toVertex = "b"});
    graph_.addEdge({.id = "b->a", .fromVertex = "b", .toVertex = "a"});

    auto result = validateAll(graph_, 11264);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ValidationError::kCycleDetected);
}

TEST_F(GraphValidatorTest, test_validateAllPasses) {
    makeVertex("cam", 0);
    makeVertex("blur", 250);
    makeShmEdge("cam", "blur", "/oe.vid.ingest");

    EXPECT_TRUE(validateAll(graph_, 11264).has_value());
}

TEST_F(GraphValidatorTest, test_emptyGraphValid) {
    /* Empty graph has no cycles, no VRAM, no paths — passes validation */
    EXPECT_TRUE(validateDag(graph_).has_value());
    EXPECT_TRUE(validateVramBudget(graph_, 11264).has_value());
    EXPECT_TRUE(validateShmPathUniqueness(graph_).has_value());
    /* connectedSources on empty graph also passes — no vertices to check */
    EXPECT_TRUE(validateConnectedSources(graph_).has_value());
}
