/* tests/core/graph/test_graph_serialization.cpp — JSON round-trip */

#include <gtest/gtest.h>
#include "common/constants/conversation_constants.hpp"
#include "graph/graph_serialization.hpp"

#include <nlohmann/json.hpp>

using namespace core::graph;
using namespace core::graph::serialization;

class GraphSerializationTest : public ::testing::Test {
protected:
    PipelineGraph graph_;
    std::unordered_map<std::string, EdgeState> edgeStates_;
};

TEST_F(GraphSerializationTest, test_emptyGraph) {
    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);

    EXPECT_EQ(parsed["schema_version"], defaults::kSchemaVersion);
    EXPECT_EQ(parsed["vertex_count"], 0);
    EXPECT_EQ(parsed["edge_count"], 0);
    EXPECT_TRUE(parsed["vertices"].is_array());
    EXPECT_TRUE(parsed["edges"].is_array());
}

TEST_F(GraphSerializationTest, test_shmEdgeSerialization) {
    graph_.addVertex({.id = "cam", .binaryName = "omniedge_video_ingest", .displayLabel = "VideoIngest"});
    graph_.addVertex({.id = "blur", .binaryName = "omniedge_bg_blur", .displayLabel = "BackgroundBlur", .vramBudgetMiB = 250});
    graph_.addEdge({
        .id = "cam->blur",
        .fromVertex = "cam",
        .toVertex = "blur",
        .transport = TransportType::kShm,
        .attributes = ShmAttributes{.shmPath = "/oe.vid.ingest", .slotCount = 4, .slotSizeBytes = 6220800}
    });
    edgeStates_["cam->blur"] = EdgeState::kActive;

    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);

    EXPECT_EQ(parsed["vertex_count"], 2);
    EXPECT_EQ(parsed["edge_count"], 1);

    /* Check edge */
    const auto& edges = parsed["edges"];
    ASSERT_EQ(edges.size(), 1);
    EXPECT_EQ(edges[0]["id"], "cam->blur");
    EXPECT_EQ(edges[0]["transport"], "shm");
    EXPECT_EQ(edges[0]["state"], "Active");
    EXPECT_EQ(edges[0]["attributes"]["shm_path"], "/oe.vid.ingest");
}

TEST_F(GraphSerializationTest, test_zmqEdgeSerialization) {
    graph_.addVertex({.id = "conv", .binaryName = "omniedge_conversation"});
    graph_.addVertex({.id = "bridge", .binaryName = "omniedge_ws_bridge"});
    graph_.addEdge({
        .id = "conv->bridge",
        .fromVertex = "conv",
        .toVertex = "bridge",
        .transport = TransportType::kZmq,
        .attributes = ZmqAttributes{.port = 5572, .topic = std::string(kZmqTopicConversationResponse), .conflate = false}
    });
    edgeStates_["conv->bridge"] = EdgeState::kConnecting;

    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);

    const auto& edges = parsed["edges"];
    ASSERT_EQ(edges.size(), 1);
    EXPECT_EQ(edges[0]["transport"], "zmq");
    EXPECT_EQ(edges[0]["state"], "Connecting");
    EXPECT_EQ(edges[0]["attributes"]["port"], 5572);
    EXPECT_EQ(edges[0]["attributes"]["topic"], std::string(kZmqTopicConversationResponse));
}

TEST_F(GraphSerializationTest, test_websocketEdgeSerialization) {
    graph_.addVertex({.id = "bridge", .binaryName = "omniedge_ws_bridge"});
    graph_.addVertex({.id = "frontend", .binaryName = "browser"});
    graph_.addEdge({
        .id = "bridge->frontend",
        .fromVertex = "bridge",
        .toVertex = "frontend",
        .transport = TransportType::kWebsocket,
        .attributes = WebsocketAttributes{.channel = "/video", .contentType = ContentType::kBinaryJpeg}
    });

    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);

    const auto& edges = parsed["edges"];
    EXPECT_EQ(edges[0]["transport"], "websocket");
    EXPECT_EQ(edges[0]["state"], "unknown");  /* no state registered */
    EXPECT_EQ(edges[0]["attributes"]["channel"], "/video");
}

TEST_F(GraphSerializationTest, test_vertexFields) {
    graph_.addVertex({
        .id = "blur",
        .binaryName = "omniedge_bg_blur",
        .displayLabel = "BackgroundBlur",
        .vramBudgetMiB = 250,
        .spawnArgs = {"--config", "omniedge.ini"},
        .zmqPubPort = 5567
    });

    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);

    const auto& verts = parsed["vertices"];
    ASSERT_EQ(verts.size(), 1);
    EXPECT_EQ(verts[0]["id"], "blur");
    EXPECT_EQ(verts[0]["binary_name"], "omniedge_bg_blur");
    EXPECT_EQ(verts[0]["display_label"], "BackgroundBlur");
    EXPECT_EQ(verts[0]["vram_budget_mib"], 250);
    EXPECT_EQ(verts[0]["zmq_pub_port"], 5567);
    EXPECT_EQ(verts[0]["spawn_args"].size(), 2);
}

TEST_F(GraphSerializationTest, test_schemaVersion) {
    auto json = serializeToJson(graph_, edgeStates_);
    auto parsed = nlohmann::json::parse(json);
    EXPECT_EQ(parsed["schema_version"], 1);
}
