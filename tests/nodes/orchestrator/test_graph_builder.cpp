/* tests/nodes/orchestrator/test_graph_builder.cpp
 *
 * Unit tests for GraphBuilder — converts ModeDefinition + ModuleDescriptors
 * into a core::graph::PipelineGraph (vertices only, no edges).
 */

#include <gtest/gtest.h>

#include "common/oe_logger.hpp"
#include "orchestrator/graph_builder.hpp"


class GraphBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        /* Build a representative set of module descriptors matching
         * the real omniedge_config.yaml module definitions. */
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
};

TEST_F(GraphBuilderTest, BuildConversationMode)
{
    ModeDefinition mode{
        .name     = "conversation",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "conversation_model"},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(result.has_value()) << result.error();

    const auto& graph = *result;
    EXPECT_EQ(graph.vertexCount(), 5);
    EXPECT_EQ(graph.edgeCount(), 0);

    /* Verify all required modules are present as vertices */
    for (const auto& name : mode.required) {
        auto v = graph.getVertex(name);
        ASSERT_TRUE(v.has_value()) << "missing vertex: " << name;
        EXPECT_EQ((*v)->id, name);
    }

    SPDLOG_DEBUG("conversation mode: {} vertices", graph.vertexCount());
}

TEST_F(GraphBuilderTest, BuildSecurityMode)
{
    ModeDefinition mode{
        .name     = "security",
        .required = {"video_ingest", "audio_ingest", "background_blur",
                     "websocket_bridge", "security_camera", "conversation_model"},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->vertexCount(), 6);
}

TEST_F(GraphBuilderTest, UnknownModuleReturnsError)
{
    ModeDefinition mode{
        .name     = "broken",
        .required = {"video_ingest", "nonexistent_module"},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("nonexistent_module"), std::string::npos);

    SPDLOG_DEBUG("expected error: {}", result.error());
}

TEST_F(GraphBuilderTest, EmptyModeProducesEmptyGraph)
{
    ModeDefinition mode{
        .name     = "empty",
        .required = {},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->vertexCount(), 0);
    EXPECT_EQ(result->edgeCount(), 0);
}

TEST_F(GraphBuilderTest, VertexFieldsMatchModuleDescriptor)
{
    ModeDefinition mode{
        .name     = "single",
        .required = {"conversation_model"},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(result.has_value()) << result.error();

    auto vertexResult = result->getVertex("conversation_model");
    ASSERT_TRUE(vertexResult.has_value());
    const auto* vertex = *vertexResult;

    EXPECT_EQ(vertex->id, "conversation_model");
    EXPECT_EQ(vertex->binaryName, "omniedge_conversation");
    EXPECT_EQ(vertex->displayLabel, "conversation_model");
    EXPECT_EQ(vertex->vramBudgetMiB, 5500);
    EXPECT_TRUE(vertex->zmqPubPort.has_value());
    EXPECT_EQ(*vertex->zmqPubPort, 5567);

    /* Verify spawnArgs are forwarded */
    ASSERT_EQ(vertex->spawnArgs.size(), 2);
    EXPECT_EQ(vertex->spawnArgs[0], "--config");

    SPDLOG_DEBUG("vertex '{}': binary={}, vram={}MiB, port={}",
                 vertex->id, vertex->binaryName,
                 vertex->vramBudgetMiB, *vertex->zmqPubPort);
}

TEST_F(GraphBuilderTest, ZeroZmqPortMapsToNullopt)
{
    ModeDefinition mode{
        .name     = "ingest_only",
        .required = {"video_ingest"},
    };

    auto result = GraphBuilder::buildForMode(mode, modules_);
    ASSERT_TRUE(result.has_value()) << result.error();

    auto vertexResult = result->getVertex("video_ingest");
    ASSERT_TRUE(vertexResult.has_value());
    EXPECT_FALSE((*vertexResult)->zmqPubPort.has_value());
}
