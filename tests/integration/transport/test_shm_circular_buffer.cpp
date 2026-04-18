// ---------------------------------------------------------------------------
// test_shm_circular_buffer.cpp — ShmCircularBuffer<Header> unit tests
//
// Tests: create/destroy, header access, producer→consumer visibility,
//        circular wrap-around, stale-read detection (writer laps reader),
//        move semantics, multi-slot round-trip, segment size calculation,
//        latest-slot reader.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/oe_logger.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// Minimal test header.
struct TestHeader {
	uint32_t seqNumber = 0;
	uint32_t width     = 320;
	uint32_t height    = 240;
	uint32_t _pad      = 0;
};
static_assert(sizeof(TestHeader) == 16);

constexpr uint32_t    kSlotCount    = 4;
constexpr std::size_t kSlotSize     = 1024;  // 1 KB per slot
constexpr const char* kTestShmName  = "/oe.test.circ";

// ── Create/Destroy ─────────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, CreateAndDestroyProducer)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);
	EXPECT_EQ(buf.slotCount(), kSlotCount);
	EXPECT_EQ(buf.slotByteSize(), kSlotSize);
	EXPECT_EQ(buf.writePos(), 0u);
	EXPECT_EQ(buf.readPos(), 0u);
	EXPECT_NE(buf.header(), nullptr);
	EXPECT_EQ(buf.name(), kTestShmName);

	const auto expectedSize = ShmCircularBuffer<TestHeader>::segmentSize(kSlotCount, kSlotSize);
	EXPECT_EQ(buf.totalSize(), expectedSize);
}

// ── Header Access ──────────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, HeaderAccessWorks)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);
	auto* header = buf.header();

	header->width     = 640;
	header->height    = 480;
	header->seqNumber = 42;

	EXPECT_EQ(header->width, 640u);
	EXPECT_EQ(header->height, 480u);
	EXPECT_EQ(buf.header()->seqNumber, 42u);
}

// ── Producer Write Visible to Consumer ─────────────────────────────────────

TEST(ShmCircularBufferTest, ProducerWriteVisibleToConsumer)
{
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);

	// Write known pattern to first slot
	auto [writePtr, writeIdx] = producer.acquireWriteSlot();
	EXPECT_EQ(writeIdx, 0u);
	std::memset(writePtr, 0xAB, kSlotSize);
	producer.commitWrite();

	EXPECT_EQ(producer.writePos(), 1u);

	// Consumer opens the same segment
	ShmCircularBuffer<TestHeader> consumer(kTestShmName, kSlotCount, kSlotSize, false);
	auto result = consumer.acquireReadSlot();
	EXPECT_TRUE(result.valid);
	EXPECT_EQ(result.slot, 0u);
	EXPECT_EQ(result.data[0], 0xABu);
	EXPECT_EQ(result.data[kSlotSize - 1], 0xABu);
	consumer.advanceRead();

	EXPECT_EQ(consumer.readPos(), 1u);
}

// ── Circular Wrap-Around ───────────────────────────────────────────────────

TEST(ShmCircularBufferTest, CircularAdvancementWrapsCorrectly)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);

	// Write slotCount + 2 slots — verifies wrap-around
	const uint32_t totalWrites = kSlotCount + 2;
	for (uint32_t i = 0; i < totalWrites; ++i) {
		auto [ptr, idx] = buf.acquireWriteSlot();
		EXPECT_EQ(idx, i % kSlotCount);
		// Fill with a unique byte per write
		std::memset(ptr, static_cast<uint8_t>(i + 1), kSlotSize);
		buf.commitWrite();
	}

	EXPECT_EQ(buf.writePos(), totalWrites);

	// The last written slot should be at index (totalWrites-1) % slotCount
	auto latest = buf.readLatestSlot();
	EXPECT_TRUE(latest.valid);
	EXPECT_EQ(latest.slot, (totalWrites - 1) % kSlotCount);
	// Last write used byte value totalWrites (6)
	EXPECT_EQ(latest.data[0], static_cast<uint8_t>(totalWrites));
}

// ── Stale Read Detection (Writer Laps Reader) ─────────────────────────────

TEST(ShmCircularBufferTest, StaleReadDetection)
{
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);
	ShmCircularBuffer<TestHeader> consumer(kTestShmName, kSlotCount, kSlotSize, false);

	// Writer writes slotCount + 1 times without consumer reading
	for (uint32_t i = 0; i <= kSlotCount; ++i) {
		auto [ptr, idx] = producer.acquireWriteSlot();
		std::memset(ptr, static_cast<uint8_t>(i), kSlotSize);
		producer.commitWrite();
	}

	// Consumer should detect that writer has lapped
	EXPECT_TRUE(consumer.writerHasLappedReader());

	auto result = consumer.acquireReadSlot();
	EXPECT_FALSE(result.valid);  // Lapped → invalid
}

// ── No Data Available ──────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, EmptyBufferReturnsInvalid)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);

	auto result = buf.acquireReadSlot();
	EXPECT_FALSE(result.valid);
	EXPECT_EQ(result.data, nullptr);
}

// ── Move Semantics ─────────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, MoveConstructorTransfersOwnership)
{
	ShmCircularBuffer<TestHeader> original(kTestShmName, kSlotCount, kSlotSize, true);
	original.header()->seqNumber = 99;

	auto [ptr, idx] = original.acquireWriteSlot();
	std::memset(ptr, 0xCC, kSlotSize);
	original.commitWrite();

	ShmCircularBuffer<TestHeader> moved(std::move(original));
	EXPECT_EQ(moved.header()->seqNumber, 99u);
	EXPECT_EQ(moved.writePos(), 1u);
	EXPECT_EQ(moved.slotCount(), kSlotCount);
}

// ── Multi-Slot Round-Trip ──────────────────────────────────────────────────

TEST(ShmCircularBufferTest, MultiSlotRoundTrip)
{
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);
	ShmCircularBuffer<TestHeader> consumer(kTestShmName, kSlotCount, kSlotSize, false);

	// Write 3 different patterns
	for (uint32_t i = 0; i < 3; ++i) {
		auto [ptr, idx] = producer.acquireWriteSlot();
		std::memset(ptr, static_cast<uint8_t>(0x10 + i), kSlotSize);
		producer.commitWrite();
	}

	// Consumer reads all 3 sequentially
	for (uint32_t i = 0; i < 3; ++i) {
		auto result = consumer.acquireReadSlot();
		ASSERT_TRUE(result.valid) << "slot " << i;
		EXPECT_EQ(result.slot, i);
		EXPECT_EQ(result.data[0], static_cast<uint8_t>(0x10 + i))
			<< "pattern mismatch at slot " << i;
		consumer.advanceRead();
	}

	// No more data
	auto empty = consumer.acquireReadSlot();
	EXPECT_FALSE(empty.valid);
}

// ── Segment Size Calculation ───────────────────────────────────────────────

TEST(ShmCircularBufferTest, SegmentSizeCalculation)
{
	const auto computed = ShmCircularBuffer<TestHeader>::segmentSize(kSlotCount, kSlotSize);

	// Header padded to 64B + control 64B + slots
	const auto headerPadded = alignToCacheLine(sizeof(TestHeader));
	const auto expected = headerPadded + sizeof(ShmCircularControl)
		+ static_cast<std::size_t>(kSlotCount) * kSlotSize;
	EXPECT_EQ(computed, expected);
}

// ── Latest Slot Reader ─────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, LatestSlotReturnsNewest)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);

	// Nothing written yet
	auto empty = buf.readLatestSlot();
	EXPECT_FALSE(empty.valid);

	// Write 3 slots
	for (uint32_t i = 0; i < 3; ++i) {
		auto [ptr, idx] = buf.acquireWriteSlot();
		std::memset(ptr, static_cast<uint8_t>(i + 1), kSlotSize);
		buf.commitWrite();
	}

	auto latest = buf.readLatestSlot();
	EXPECT_TRUE(latest.valid);
	EXPECT_EQ(latest.slot, 2u);      // 3rd write → slot index 2
	EXPECT_EQ(latest.data[0], 3u);   // Pattern byte = 3
}

// ── ShmCircularReader (Lightweight Consumer) ───────────────────────────────

TEST(ShmCircularReaderTest, ReadsFromRawMapping)
{
	// Create a circular buffer producer
	const auto totalSize = ShmCircularBuffer<TestHeader>::segmentSize(kSlotCount, kSlotSize);
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);

	// Write a pattern
	auto [ptr, idx] = producer.acquireWriteSlot();
	std::memset(ptr, 0xDE, kSlotSize);
	producer.commitWrite();

	// Open with raw ShmMapping (simulates consumer that doesn't know template params)
	ShmMapping raw(kTestShmName, totalSize, false);
	ShmCircularReader reader(raw, sizeof(TestHeader));

	EXPECT_EQ(reader.slotCount(), kSlotCount);
	EXPECT_EQ(reader.slotByteSize(), kSlotSize);

	auto latestIdx = reader.latestSlotIndex();
	ASSERT_TRUE(latestIdx.has_value());
	EXPECT_EQ(*latestIdx, 0u);

	const uint8_t* data = reader.slotData(*latestIdx);
	EXPECT_EQ(data[0], 0xDEu);
	EXPECT_EQ(data[kSlotSize - 1], 0xDEu);
}

TEST(ShmCircularReaderTest, StaleReadDetection)
{
	const auto totalSize = ShmCircularBuffer<TestHeader>::segmentSize(kSlotCount, kSlotSize);
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);

	auto [ptr, idx] = producer.acquireWriteSlot();
	std::memset(ptr, 0xAA, kSlotSize);
	producer.commitWrite();

	ShmMapping raw(kTestShmName, totalSize, false);
	ShmCircularReader reader(raw, sizeof(TestHeader));

	uint64_t captured = reader.writePos();
	EXPECT_FALSE(reader.isStale(captured));

	// Producer writes again — now the captured pos is stale
	auto [ptr2, idx2] = producer.acquireWriteSlot();
	std::memset(ptr2, 0xBB, kSlotSize);
	producer.commitWrite();

	EXPECT_TRUE(reader.isStale(captured));
}

// ── Write/Read Position Monotonicity ───────────────────────────────────────

TEST(ShmCircularBufferTest, PositionMonotonicity)
{
	ShmCircularBuffer<TestHeader> buf(kTestShmName, kSlotCount, kSlotSize, true);

	uint64_t prevWrite = buf.writePos();
	uint64_t prevRead  = buf.readPos();

	for (uint32_t i = 0; i < 10; ++i) {
		[[maybe_unused]] auto [ptr, idx] = buf.acquireWriteSlot();
		buf.commitWrite();

		EXPECT_GT(buf.writePos(), prevWrite);
		prevWrite = buf.writePos();
	}

	// Read side
	ShmCircularBuffer<TestHeader> consumer(kTestShmName, kSlotCount, kSlotSize, false);
	for (uint32_t i = 0; i < 10; ++i) {
		auto result = consumer.acquireReadSlot();
		if (result.valid) {
			consumer.advanceRead();
			EXPECT_GT(consumer.readPos(), prevRead);
			prevRead = consumer.readPos();
		}
	}
}

// ── Available Slots ────────────────────────────────────────────────────────

TEST(ShmCircularBufferTest, AvailableSlotsTracksCorrectly)
{
	ShmCircularBuffer<TestHeader> producer(kTestShmName, kSlotCount, kSlotSize, true);
	ShmCircularBuffer<TestHeader> consumer(kTestShmName, kSlotCount, kSlotSize, false);

	EXPECT_EQ(producer.availableSlots(), 0u);

	// Write 3 slots
	for (uint32_t i = 0; i < 3; ++i) {
		[[maybe_unused]] auto [ptr, idx] = producer.acquireWriteSlot();
		producer.commitWrite();
	}
	EXPECT_EQ(consumer.availableSlots(), 3u);

	// Read 1
	[[maybe_unused]] auto readResult = consumer.acquireReadSlot();
	consumer.advanceRead();
	EXPECT_EQ(consumer.availableSlots(), 2u);
}

} // namespace
