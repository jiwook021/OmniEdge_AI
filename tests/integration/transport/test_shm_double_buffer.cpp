// ---------------------------------------------------------------------------
// test_shm_double_buffer.cpp — ShmDoubleBuffer<Header> unit tests
//
// Tests: create/destroy, producer→consumer visibility, stale-read guard,
//        move semantics, uint32_t underflow guard.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "shm/shm_double_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/oe_logger.hpp"

#include <cstring>
#include <string>

namespace {

/// Minimal test header (fits after the 64-byte control region).
struct TestHeader {
	uint32_t seqNumber = 0;
	uint32_t width     = 320;
	uint32_t height    = 240;
	uint32_t _pad      = 0;
};
static_assert(sizeof(TestHeader) == 16);

constexpr std::size_t kSlotSize    = 320 * 240;  // One "frame" per slot
constexpr std::size_t kTotalSize   = kDoubleBufferControlBytes
                                   + sizeof(TestHeader)
                                   + 2 * kSlotSize;  // Two slots
constexpr std::size_t kDataOffset  = kDoubleBufferControlBytes + sizeof(TestHeader);

constexpr const char* kTestShmName = "/oe.test.dbuf";

// ── Create/Destroy ─────────────────────────────────────────────────────────

TEST(ShmDoubleBufferTest, CreateAndDestroyProducer)
{
	ShmDoubleBuffer<TestHeader> buf(kTestShmName, kTotalSize, true);
	EXPECT_EQ(buf.totalSize(), kTotalSize);
	EXPECT_EQ(buf.name(), kTestShmName);
	EXPECT_NE(buf.header(), nullptr);
	EXPECT_EQ(buf.writeIndex(), 0u);
	OE_LOG_DEBUG("CreateAndDestroyProducer: total_size={}", buf.totalSize());
}

TEST(ShmDoubleBufferTest, HeaderAccessWorks)
{
	ShmDoubleBuffer<TestHeader> buf(kTestShmName, kTotalSize, true);
	auto* header = buf.header();

	// SHM memory is zero-initialized by the OS (not C++ default-initialized).
	// Producer must write header fields explicitly.
	header->width  = 320;
	header->height = 240;
	EXPECT_EQ(header->width,  320u);
	EXPECT_EQ(header->height, 240u);

	header->seqNumber = 42;
	EXPECT_EQ(buf.header()->seqNumber, 42u);
	OE_LOG_DEBUG("HeaderAccessWorks: seqNumber={}", header->seqNumber);
}

// ── Producer → Consumer Data Visibility ────────────────────────────────────

TEST(ShmDoubleBufferTest, ProducerWriteVisibleToConsumer)
{
	// Producer creates the segment
	ShmDoubleBuffer<TestHeader> producer(kTestShmName, kTotalSize, true);

	// Write known pattern to write slot
	auto [writePtr, writeSlot] = producer.writeSlotData(kSlotSize, kDataOffset);
	EXPECT_EQ(writeSlot, 1u);  // Initial writeIndex=0, so write slot is 1
	std::memset(writePtr, 0xAB, kSlotSize);
	producer.flipSlot();  // Make slot 1 visible

	OE_LOG_DEBUG("Producer wrote 0xAB to slot {}, flipped", writeSlot);

	// Consumer opens the same segment
	ShmDoubleBuffer<TestHeader> consumer(kTestShmName, kTotalSize, false);
	auto readResult = consumer.readSlotData(kSlotSize, kDataOffset);
	EXPECT_TRUE(readResult.valid);
	EXPECT_EQ(readResult.data[0], 0xAB);
	EXPECT_EQ(readResult.data[kSlotSize - 1], 0xAB);

	OE_LOG_DEBUG("Consumer read slot {}: first_byte=0x{:02X}, valid={}",
	           readResult.slot, readResult.data[0], readResult.valid);
}

// ── Double-Buffer Flip ─────────────────────────────────────────────────────

TEST(ShmDoubleBufferTest, FlipAlternatesSlots)
{
	ShmDoubleBuffer<TestHeader> buf(kTestShmName, kTotalSize, true);

	uint32_t idx0 = buf.writeIndex();
	EXPECT_EQ(idx0, 0u);

	buf.flipSlot();
	uint32_t idx1 = buf.writeIndex();
	EXPECT_EQ(idx1, 1u);

	buf.flipSlot();
	uint32_t idx2 = buf.writeIndex();
	EXPECT_EQ(idx2, 0u);

	OE_LOG_DEBUG("FlipAlternatesSlots: {} -> {} -> {}", idx0, idx1, idx2);
}

// ── Move Semantics ─────────────────────────────────────────────────────────

TEST(ShmDoubleBufferTest, MoveConstructorTransfersOwnership)
{
	ShmDoubleBuffer<TestHeader> original(kTestShmName, kTotalSize, true);
	original.header()->seqNumber = 99;

	ShmDoubleBuffer<TestHeader> moved(std::move(original));
	EXPECT_EQ(moved.header()->seqNumber, 99u);
	EXPECT_EQ(moved.totalSize(), kTotalSize);

	OE_LOG_DEBUG("MoveConstructor: seqNumber={}", moved.header()->seqNumber);
}

// ── WriteIndex Modulo Safety (uint32_t overflow) ───────────────────────────

TEST(ShmDoubleBufferTest, WriteIndexModuloHandlesOverflow)
{
	ShmDoubleBuffer<TestHeader> buf(kTestShmName, kTotalSize, true);

	// Manually set the control index to a large value near uint32_t max
	// to verify that writeIndex() correctly handles wrapping via % 2.
	auto* controlRegion = reinterpret_cast<std::atomic<uint32_t>*>(buf.rawBytes());
	controlRegion->store(0xFFFF'FFFE, std::memory_order_release);

	EXPECT_EQ(buf.writeIndex(), 0u);  // 0xFFFFFFFE % 2 = 0

	buf.flipSlot();  // 0xFFFFFFFE + 1 = 0xFFFFFFFF
	EXPECT_EQ(buf.writeIndex(), 1u);  // 0xFFFFFFFF % 2 = 1

	buf.flipSlot();  // 0xFFFFFFFF + 1 = 0x00000000 (wraps)
	EXPECT_EQ(buf.writeIndex(), 0u);  // 0x00000000 % 2 = 0

	OE_LOG_DEBUG("WriteIndex overflow test passed: modulo safety verified");
}

} // namespace
