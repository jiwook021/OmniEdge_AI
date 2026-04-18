/* tests/core/graph/test_graph_topology.cpp — Topological sort and reachability */

#include <gtest/gtest.h>
#include "graph/graph_topology.hpp"

#include <algorithm>
#include <unordered_set>

using namespace core::graph;
using namespace core::graph::topology;

class GraphTopologyTest : public ::testing::Test {
protected:
    PipelineGraph graph_;

    void makeVertex(const std::string& id) {
        graph_.addVertex({.id = id, .binaryName = "bin_" + id});
    }

    void makeEdge(const std::string& from, const std::string& to) {
        graph_.addEdge({
            .id = from + "->" + to,
            .fromVertex = from,
            .toVertex = to,
            .transport = TransportType::kShm,
            .attributes = ShmAttributes{.shmPath = "/oe." + from}
        });
    }

    /* Check that 'a' comes before 'b' in the sorted order */
    bool comesBefore(const std::vector<std::string>& order,
                     const std::string& a, const std::string& b) {
        auto posA = std::find(order.begin(), order.end(), a);
        auto posB = std::find(order.begin(), order.end(), b);
        return posA < posB;
    }
};

TEST_F(GraphTopologyTest, test_emptyGraph) {
    auto result = topologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST_F(GraphTopologyTest, test_singleVertex) {
    makeVertex("cam");
    auto result = topologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ((*result)[0], "cam");
}

TEST_F(GraphTopologyTest, test_linearChain) {
    /* cam → blur → bridge */
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("bridge");
    makeEdge("cam", "blur");
    makeEdge("blur", "bridge");

    auto result = topologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3);
    EXPECT_TRUE(comesBefore(*result, "cam", "blur"));
    EXPECT_TRUE(comesBefore(*result, "blur", "bridge"));
}

TEST_F(GraphTopologyTest, test_diamondDag) {
    /*     cam
          /   \
       blur   face
          \   /
          bridge   */
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("face");
    makeVertex("bridge");
    makeEdge("cam", "blur");
    makeEdge("cam", "face");
    makeEdge("blur", "bridge");
    makeEdge("face", "bridge");

    auto result = topologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 4);
    EXPECT_TRUE(comesBefore(*result, "cam", "blur"));
    EXPECT_TRUE(comesBefore(*result, "cam", "face"));
    EXPECT_TRUE(comesBefore(*result, "blur", "bridge"));
    EXPECT_TRUE(comesBefore(*result, "face", "bridge"));
}

TEST_F(GraphTopologyTest, test_cycleDetected) {
    makeVertex("a");
    makeVertex("b");
    makeVertex("c");
    makeEdge("a", "b");
    makeEdge("b", "c");
    makeEdge("c", "a");

    auto result = topologicalSort(graph_);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), TopologyError::kCycleDetected);
}

TEST_F(GraphTopologyTest, test_reverseTopologicalSort) {
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("bridge");
    makeEdge("cam", "blur");
    makeEdge("blur", "bridge");

    auto result = reverseTopologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3);
    /* Reverse: bridge before blur before cam */
    EXPECT_TRUE(comesBefore(*result, "bridge", "blur"));
    EXPECT_TRUE(comesBefore(*result, "blur", "cam"));
}

TEST_F(GraphTopologyTest, test_ancestors) {
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("bridge");
    makeEdge("cam", "blur");
    makeEdge("blur", "bridge");

    auto anc = ancestors(graph_, "bridge");
    std::unordered_set<std::string> ancSet(anc.begin(), anc.end());
    EXPECT_EQ(ancSet.size(), 2);
    EXPECT_TRUE(ancSet.count("cam"));
    EXPECT_TRUE(ancSet.count("blur"));

    EXPECT_TRUE(ancestors(graph_, "cam").empty());
}

TEST_F(GraphTopologyTest, test_descendants) {
    makeVertex("cam");
    makeVertex("blur");
    makeVertex("bridge");
    makeEdge("cam", "blur");
    makeEdge("blur", "bridge");

    auto desc = descendants(graph_, "cam");
    std::unordered_set<std::string> descSet(desc.begin(), desc.end());
    EXPECT_EQ(descSet.size(), 2);
    EXPECT_TRUE(descSet.count("blur"));
    EXPECT_TRUE(descSet.count("bridge"));

    EXPECT_TRUE(descendants(graph_, "bridge").empty());
}

TEST_F(GraphTopologyTest, test_disconnectedVertices) {
    /* Two isolated vertices — valid DAG (no edges, no cycles) */
    makeVertex("a");
    makeVertex("b");
    auto result = topologicalSort(graph_);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2);
}
