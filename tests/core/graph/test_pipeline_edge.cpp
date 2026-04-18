/* tests/core/graph/test_pipeline_edge.cpp — PipelineEdge + transport types */

#include <gtest/gtest.h>
#include "common/constants/conversation_constants.hpp"
#include "graph/pipeline_edge.hpp"

using namespace core::graph;

TEST(PipelineEdgeTest, test_shmEdge) {
    PipelineEdge edge{
        .id = "cam->blur",
        .fromVertex = "cam",
        .toVertex = "blur",
        .transport = TransportType::kShm,
        .attributes = ShmAttributes{
            .shmPath = "/oe.vid.ingest",
            .bufferType = BufferType::kCircularBuffer,
            .slotCount = 4,
            .slotSizeBytes = 6220800,
            .alignmentBytes = 64
        }
    };

    EXPECT_EQ(edge.transport, TransportType::kShm);
    EXPECT_EQ(shmPathOf(edge), "/oe.vid.ingest");
}

TEST(PipelineEdgeTest, test_zmqEdge) {
    PipelineEdge edge{
        .id = "conv->bridge",
        .fromVertex = "conv",
        .toVertex = "bridge",
        .transport = TransportType::kZmq,
        .attributes = ZmqAttributes{
            .port = 5572,
            .topic = std::string(kZmqTopicConversationResponse),
            .conflate = false
        }
    };

    EXPECT_EQ(edge.transport, TransportType::kZmq);
    EXPECT_EQ(shmPathOf(edge), "");  /* not SHM */
    const auto& attrs = std::get<ZmqAttributes>(edge.attributes);
    EXPECT_EQ(attrs.port, 5572);
    EXPECT_EQ(attrs.topic, std::string(kZmqTopicConversationResponse));
}

TEST(PipelineEdgeTest, test_websocketEdge) {
    PipelineEdge edge{
        .id = "bridge->frontend_video",
        .fromVertex = "bridge",
        .toVertex = "frontend",
        .transport = TransportType::kWebsocket,
        .attributes = WebsocketAttributes{
            .channel = "/video",
            .contentType = ContentType::kBinaryJpeg
        }
    };

    EXPECT_EQ(edge.transport, TransportType::kWebsocket);
    EXPECT_EQ(shmPathOf(edge), "");
    const auto& attrs = std::get<WebsocketAttributes>(edge.attributes);
    EXPECT_EQ(attrs.channel, "/video");
}

TEST(PipelineEdgeTest, test_equality) {
    PipelineEdge a{.id = "e1", .fromVertex = "a", .toVertex = "b"};
    PipelineEdge b{.id = "e1", .fromVertex = "c", .toVertex = "d"};
    PipelineEdge c{.id = "e2", .fromVertex = "a", .toVertex = "b"};

    EXPECT_EQ(a, b);   /* same ID */
    EXPECT_NE(a, c);   /* different ID */
}
