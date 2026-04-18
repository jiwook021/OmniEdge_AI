/* tests/core/graph/test_graph_diff.cpp — Diff computation between graphs */

#include <gtest/gtest.h>
#include "graph/graph_diff.hpp"

#include <algorithm>
#include <unordered_set>

using namespace core::graph;

class GraphDiffTest : public ::testing::Test {
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

    bool contains(const std::vector<std::string>& vec, const std::string& val) {
        return std::find(vec.begin(), vec.end(), val) != vec.end();
    }
};

TEST_F(GraphDiffTest, test_identicalGraphs) {
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");

    makeVertex(target_, "cam");
    makeVertex(target_, "blur");
    makeShmEdge(target_, "cam", "blur", "/oe.vid.ingest");

    auto diff = computeDiff(current_, target_);

    EXPECT_TRUE(diff.verticesToEvict.empty());
    EXPECT_TRUE(diff.verticesToSpawn.empty());
    EXPECT_EQ(diff.verticesKept.size(), 2);
    EXPECT_TRUE(diff.edgesToRemove.empty());
    EXPECT_TRUE(diff.edgesToAdd.empty());
    EXPECT_EQ(diff.edgesKept.size(), 1);
    EXPECT_TRUE(diff.shmPathsToUnlink.empty());
}

TEST_F(GraphDiffTest, test_disjointGraphs) {
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");

    makeVertex(target_, "conv");
    makeVertex(target_, "tts");
    makeShmEdge(target_, "conv", "tts", "/oe.nlp.llm");

    auto diff = computeDiff(current_, target_);

    EXPECT_EQ(diff.verticesToEvict.size(), 2);
    EXPECT_EQ(diff.verticesToSpawn.size(), 2);
    EXPECT_TRUE(diff.verticesKept.empty());

    EXPECT_EQ(diff.edgesToRemove.size(), 1);
    EXPECT_EQ(diff.edgesToAdd.size(), 1);
    EXPECT_TRUE(diff.edgesKept.empty());

    /* SHM path from current should be unlinked since target doesn't use it */
    EXPECT_EQ(diff.shmPathsToUnlink.size(), 1);
    EXPECT_TRUE(contains(diff.shmPathsToUnlink, "/oe.vid.ingest"));
}

TEST_F(GraphDiffTest, test_partialOverlap) {
    /* current: cam → blur → bridge */
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeVertex(current_, "bridge");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");
    makeShmEdge(current_, "blur", "bridge", "/oe.cv.blur");

    /* target: cam → conv → bridge  (blur replaced by conv) */
    makeVertex(target_, "cam");
    makeVertex(target_, "conv");
    makeVertex(target_, "bridge");
    makeShmEdge(target_, "cam", "conv", "/oe.vid.ingest");
    makeShmEdge(target_, "conv", "bridge", "/oe.nlp.llm");

    auto diff = computeDiff(current_, target_);

    EXPECT_TRUE(contains(diff.verticesToEvict, "blur"));
    EXPECT_TRUE(contains(diff.verticesToSpawn, "conv"));
    EXPECT_TRUE(contains(diff.verticesKept, "cam"));
    EXPECT_TRUE(contains(diff.verticesKept, "bridge"));

    /* /oe.cv.blur is removed; /oe.vid.ingest is kept in target */
    EXPECT_TRUE(contains(diff.shmPathsToUnlink, "/oe.cv.blur"));
    EXPECT_FALSE(contains(diff.shmPathsToUnlink, "/oe.vid.ingest"));
}

TEST_F(GraphDiffTest, test_emptyGraphs) {
    auto diff = computeDiff(current_, target_);
    EXPECT_TRUE(diff.verticesToEvict.empty());
    EXPECT_TRUE(diff.verticesToSpawn.empty());
    EXPECT_TRUE(diff.shmPathsToUnlink.empty());
}

TEST_F(GraphDiffTest, test_shmFanoutPreservation) {
    /* current: cam fans out to blur and face via same SHM */
    makeVertex(current_, "cam");
    makeVertex(current_, "blur");
    makeVertex(current_, "face");
    makeShmEdge(current_, "cam", "blur", "/oe.vid.ingest");
    makeShmEdge(current_, "cam", "face", "/oe.vid.ingest");

    /* target: only cam→blur remains (face removed) */
    makeVertex(target_, "cam");
    makeVertex(target_, "blur");
    makeShmEdge(target_, "cam", "blur", "/oe.vid.ingest");

    auto diff = computeDiff(current_, target_);

    /* SHM path NOT unlinked because target still has an edge using it */
    EXPECT_TRUE(diff.shmPathsToUnlink.empty());
    EXPECT_TRUE(contains(diff.verticesToEvict, "face"));
}
