/* tests/core/graph/test_core_pipeline_graph.cpp — DAG container operations */

#include <gtest/gtest.h>
#include "graph/pipeline_graph.hpp"

using namespace core::graph;

class CorePipelineGraphTest : public ::testing::Test {
protected:
    PipelineGraph graph_;

    void addTwoVertices() {
        graph_.addVertex({.id = "cam", .binaryName = "omniedge_video_ingest"});
        graph_.addVertex({.id = "blur", .binaryName = "omniedge_bg_blur", .vramBudgetMiB = 250});
    }

    void addEdgeBetween() {
        addTwoVertices();
        graph_.addEdge({
            .id = "cam->blur",
            .fromVertex = "cam",
            .toVertex = "blur",
            .transport = TransportType::kShm,
            .attributes = ShmAttributes{.shmPath = "/oe.vid.ingest"}
        });
    }
};

TEST_F(CorePipelineGraphTest, test_emptyGraph) {
    EXPECT_EQ(graph_.vertexCount(), 0);
    EXPECT_EQ(graph_.edgeCount(), 0);
    EXPECT_TRUE(graph_.allVertexIds().empty());
    EXPECT_TRUE(graph_.allEdgeIds().empty());
}

TEST_F(CorePipelineGraphTest, test_addVertex) {
    auto result = graph_.addVertex({.id = "cam", .binaryName = "omniedge_video_ingest"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(graph_.vertexCount(), 1);
    EXPECT_TRUE(graph_.hasVertex("cam"));
    EXPECT_FALSE(graph_.hasVertex("nonexistent"));
}

TEST_F(CorePipelineGraphTest, test_addDuplicateVertex) {
    graph_.addVertex({.id = "cam", .binaryName = "bin1"});
    auto result = graph_.addVertex({.id = "cam", .binaryName = "bin2"});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kDuplicateVertex);
    EXPECT_EQ(graph_.vertexCount(), 1);
}

TEST_F(CorePipelineGraphTest, test_getVertex) {
    graph_.addVertex({.id = "cam", .binaryName = "omniedge_video_ingest", .vramBudgetMiB = 0});
    auto result = graph_.getVertex("cam");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->binaryName, "omniedge_video_ingest");

    auto missing = graph_.getVertex("nonexistent");
    EXPECT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error(), GraphError::kVertexNotFound);
}

TEST_F(CorePipelineGraphTest, test_addEdge) {
    addEdgeBetween();
    EXPECT_EQ(graph_.edgeCount(), 1);
    EXPECT_TRUE(graph_.hasEdge("cam->blur"));
}

TEST_F(CorePipelineGraphTest, test_addEdgeMissingEndpoint) {
    graph_.addVertex({.id = "cam", .binaryName = "bin"});
    auto result = graph_.addEdge({.id = "e1", .fromVertex = "cam", .toVertex = "nonexistent"});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kEndpointVertexMissing);
}

TEST_F(CorePipelineGraphTest, test_addDuplicateEdge) {
    addEdgeBetween();
    auto result = graph_.addEdge({.id = "cam->blur", .fromVertex = "cam", .toVertex = "blur"});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kDuplicateEdge);
}

TEST_F(CorePipelineGraphTest, test_adjacencyQueries) {
    addEdgeBetween();

    auto out = graph_.outgoingEdges("cam");
    ASSERT_EQ(out.size(), 1);
    EXPECT_EQ(out[0], "cam->blur");

    auto in = graph_.incomingEdges("blur");
    ASSERT_EQ(in.size(), 1);
    EXPECT_EQ(in[0], "cam->blur");

    EXPECT_TRUE(graph_.outgoingEdges("blur").empty());
    EXPECT_TRUE(graph_.incomingEdges("cam").empty());
}

TEST_F(CorePipelineGraphTest, test_edgesOnShmPath) {
    addEdgeBetween();
    /* Add another consumer of the same SHM */
    graph_.addVertex({.id = "face", .binaryName = "omniedge_face_recog"});
    graph_.addEdge({
        .id = "cam->face",
        .fromVertex = "cam",
        .toVertex = "face",
        .transport = TransportType::kShm,
        .attributes = ShmAttributes{.shmPath = "/oe.vid.ingest"}
    });

    auto edges = graph_.edgesOnShmPath("/oe.vid.ingest");
    EXPECT_EQ(edges.size(), 2);

    EXPECT_TRUE(graph_.edgesOnShmPath("/nonexistent").empty());
}

TEST_F(CorePipelineGraphTest, test_removeEdge) {
    addEdgeBetween();
    auto result = graph_.removeEdge("cam->blur");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(graph_.edgeCount(), 0);
    EXPECT_TRUE(graph_.outgoingEdges("cam").empty());
    EXPECT_TRUE(graph_.incomingEdges("blur").empty());
}

TEST_F(CorePipelineGraphTest, test_removeEdgeNotFound) {
    auto result = graph_.removeEdge("nonexistent");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kEdgeNotFound);
}

TEST_F(CorePipelineGraphTest, test_removeVertexWithEdges) {
    addEdgeBetween();
    auto result = graph_.removeVertex("cam");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kVertexHasEdges);
}

TEST_F(CorePipelineGraphTest, test_removeVertexClean) {
    addEdgeBetween();
    graph_.removeEdge("cam->blur");
    auto result = graph_.removeVertex("blur");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(graph_.vertexCount(), 1);
    EXPECT_FALSE(graph_.hasVertex("blur"));
}

TEST_F(CorePipelineGraphTest, test_removeVertexNotFound) {
    auto result = graph_.removeVertex("nonexistent");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), GraphError::kVertexNotFound);
}
