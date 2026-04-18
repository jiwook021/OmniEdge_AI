#include <gtest/gtest.h>

#include "ingest/video_ingest_node.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/runtime_defaults.hpp"

#include <chrono>
#include <cstring>
#include <thread>

// ---------------------------------------------------------------------------
// VideoIngestNode tests — CPU-only (no GPU, no GStreamer TCP source required).
// Tests exercise the SHM layout, circular buffer logic, and config validation
// using the ShmCircularBuffer<ShmVideoHeader> API.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Helper: build a test config with offset ports to avoid production clash
// ---------------------------------------------------------------------------
[[maybe_unused]]
static VideoIngestNode::Config makeTestConfig()
{
	VideoIngestNode::Config cfg;
	cfg.pubPort         = 25555;   // +20000 offset
	cfg.wsBridgeSubPort = 25570;
	cfg.frameWidth      = 320;
	cfg.frameHeight     = 240;
	cfg.v4l2Device      = "/dev/video0";
	cfg.moduleName      = "video_ingest_test";
	return cfg;
}

// ---------------------------------------------------------------------------
// SHM circular buffer write/read correctness
//
// Simulate the producer's circular buffer write and verify consumer sees
// the right slot by applying the same protocol used in video_ingest_node.cpp.
// ---------------------------------------------------------------------------

TEST(VideoIngestShmTest, CircularBufferWriteProtocol)
{
	const uint32_t W = 4, H = 4;
	const std::size_t frameBytes = W * H * 3u;
	constexpr uint32_t kSlots = 4;

	const std::string shmName = "/oe.test.vi_circ";
	ShmCircularBuffer<ShmVideoHeader> producer(shmName, kSlots, frameBytes, true);

	auto* hdr = producer.header();
	hdr->width  = W;
	hdr->height = H;

	// Write frame A into slot 0
	auto [ptrA, idxA] = producer.acquireWriteSlot();
	EXPECT_EQ(idxA, 0u);
	std::memset(ptrA, 0xAA, frameBytes);
	producer.commitWrite();

	// Write frame B into slot 1
	auto [ptrB, idxB] = producer.acquireWriteSlot();
	EXPECT_EQ(idxB, 1u);
	std::memset(ptrB, 0xBB, frameBytes);
	producer.commitWrite();

	// Consumer opens the segment and reads latest
	ShmCircularBuffer<ShmVideoHeader> consumer(shmName, kSlots, frameBytes, false);
	auto latest = consumer.readLatestSlot();
	EXPECT_TRUE(latest.valid);
	EXPECT_EQ(latest.slot, 1u);  // Latest is slot 1
	EXPECT_EQ(latest.data[0], 0xBBu);
}

// ---------------------------------------------------------------------------
// Gradient pattern test — real-ish BGR frame data
// ---------------------------------------------------------------------------

TEST(VideoIngestShmTest, GradientFrameRoundTrip)
{
	constexpr uint32_t W = 16, H = 16;
	const std::size_t frameBytes = W * H * 3u;
	constexpr uint32_t kSlots = 4;

	const std::string shmName = "/oe.test.vi_grad";
	ShmCircularBuffer<ShmVideoHeader> producer(shmName, kSlots, frameBytes, true);

	// Generate a horizontal gradient BGR frame
	std::vector<uint8_t> frame(frameBytes);
	for (uint32_t y = 0; y < H; ++y) {
		for (uint32_t x = 0; x < W; ++x) {
			const std::size_t offset = (y * W + x) * 3;
			frame[offset + 0] = static_cast<uint8_t>(x * 16);      // B
			frame[offset + 1] = static_cast<uint8_t>(y * 16);      // G
			frame[offset + 2] = static_cast<uint8_t>((x + y) * 8); // R
		}
	}

	auto [ptr, idx] = producer.acquireWriteSlot();
	std::memcpy(ptr, frame.data(), frameBytes);
	producer.commitWrite();

	// Read back and verify pixel by pixel
	ShmCircularBuffer<ShmVideoHeader> consumer(shmName, kSlots, frameBytes, false);
	auto result = consumer.acquireReadSlot();
	ASSERT_TRUE(result.valid);

	for (uint32_t y = 0; y < H; ++y) {
		for (uint32_t x = 0; x < W; ++x) {
			const std::size_t offset = (y * W + x) * 3;
			ASSERT_EQ(result.data[offset + 0], static_cast<uint8_t>(x * 16))
				<< "B mismatch at (" << x << "," << y << ")";
			ASSERT_EQ(result.data[offset + 1], static_cast<uint8_t>(y * 16))
				<< "G mismatch at (" << x << "," << y << ")";
			ASSERT_EQ(result.data[offset + 2], static_cast<uint8_t>((x + y) * 8))
				<< "R mismatch at (" << x << "," << y << ")";
		}
	}
}

// ---------------------------------------------------------------------------
// Wrap-around and stale-read detection
// ---------------------------------------------------------------------------

TEST(VideoIngestShmTest, FrameOverwriteDetection)
{
	constexpr uint32_t W = 4, H = 4;
	const std::size_t frameBytes = W * H * 3u;
	constexpr uint32_t kSlots = 4;

	const std::string shmName = "/oe.test.vi_stale";
	ShmCircularBuffer<ShmVideoHeader> producer(shmName, kSlots, frameBytes, true);
	ShmCircularBuffer<ShmVideoHeader> consumer(shmName, kSlots, frameBytes, false);

	// Write kSlots + 1 frames without consumer reading → writer laps reader
	for (uint32_t i = 0; i <= kSlots; ++i) {
		auto [ptr, idx] = producer.acquireWriteSlot();
		std::memset(ptr, static_cast<uint8_t>(i + 1), frameBytes);
		producer.commitWrite();
	}

	// Consumer should detect writer has lapped
	EXPECT_TRUE(consumer.writerHasLappedReader());
	auto result = consumer.acquireReadSlot();
	EXPECT_FALSE(result.valid);
}

// ---------------------------------------------------------------------------
// ShmCircularReader (lightweight consumer used by shm_frame_reader.hpp)
// ---------------------------------------------------------------------------

TEST(VideoIngestShmTest, CircularReaderLatestFrame)
{
	constexpr uint32_t W = 8, H = 8;
	const std::size_t frameBytes = W * H * 3u;
	constexpr uint32_t kSlots = 4;

	const std::string shmName = "/oe.test.vi_reader";
	auto totalSize = ShmCircularBuffer<ShmVideoHeader>::segmentSize(kSlots, frameBytes);
	ShmCircularBuffer<ShmVideoHeader> producer(shmName, kSlots, frameBytes, true);

	producer.header()->width = W;
	producer.header()->height = H;

	// Write 3 frames
	for (uint32_t i = 0; i < 3; ++i) {
		auto [ptr, idx] = producer.acquireWriteSlot();
		std::memset(ptr, static_cast<uint8_t>(0x10 * (i + 1)), frameBytes);
		producer.commitWrite();
	}

	// Use raw ShmMapping + ShmCircularReader (simulates shm_frame_reader.hpp usage)
	ShmMapping raw(shmName, totalSize, false);
	ShmCircularReader reader(raw, sizeof(ShmVideoHeader));

	auto latestIdx = reader.latestSlotIndex();
	ASSERT_TRUE(latestIdx.has_value());
	EXPECT_EQ(*latestIdx, 2u);  // 3 writes → latest at index 2

	const uint8_t* frameData = reader.slotData(*latestIdx);
	EXPECT_EQ(frameData[0], 0x30u);  // 3rd frame pattern
}

