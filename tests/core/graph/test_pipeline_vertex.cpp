/* tests/core/graph/test_pipeline_vertex.cpp — PipelineVertex construction and equality */

#include <gtest/gtest.h>
#include "graph/pipeline_vertex.hpp"

using core::graph::PipelineVertex;

TEST(PipelineVertexTest, test_construction) {
    PipelineVertex v{
        .id = "blur",
        .binaryName = "omniedge_bg_blur",
        .displayLabel = "BackgroundBlur",
        .vramBudgetMiB = 250,
        .spawnArgs = {"--config", "omniedge.ini"},
        .zmqPubPort = 5567
    };

    EXPECT_EQ(v.id, "blur");
    EXPECT_EQ(v.binaryName, "omniedge_bg_blur");
    EXPECT_EQ(v.displayLabel, "BackgroundBlur");
    EXPECT_EQ(v.vramBudgetMiB, 250);
    ASSERT_EQ(v.spawnArgs.size(), 2);
    EXPECT_EQ(v.spawnArgs[0], "--config");
    ASSERT_TRUE(v.zmqPubPort.has_value());
    EXPECT_EQ(v.zmqPubPort.value(), 5567);
}

TEST(PipelineVertexTest, test_defaultZmqPort) {
    PipelineVertex v{.id = "cam", .binaryName = "omniedge_video_ingest"};
    EXPECT_FALSE(v.zmqPubPort.has_value());
    EXPECT_EQ(v.vramBudgetMiB, 0);
}

TEST(PipelineVertexTest, test_equality) {
    PipelineVertex a{.id = "conv", .binaryName = "omniedge_conversation"};
    PipelineVertex b{.id = "conv", .binaryName = "different_binary"};
    PipelineVertex c{.id = "tts", .binaryName = "omniedge_conversation"};

    EXPECT_EQ(a, b);   /* same ID = equal */
    EXPECT_NE(a, c);   /* different ID = not equal */
}
