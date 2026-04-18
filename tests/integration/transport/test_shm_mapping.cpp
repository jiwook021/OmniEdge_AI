#include <gtest/gtest.h>

#include "shm/shm_mapping.hpp"

#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// ShmMapping tests — CPU-only, no CUDA or GPU required.
// All SHM names are prefixed with /oe.test. to avoid clashing with production
// segments. Tests use unique names per test to prevent cross-test contamination.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ShmMapping lifecycle
// ---------------------------------------------------------------------------

TEST(ShmMappingTest, ProducerCreatesAndMapsSuccessfully)
{
	const std::string name = "/oe.test.shm_create";
	constexpr std::size_t size = 4096;

	EXPECT_NO_THROW({
		ShmMapping producer(name, size, /*create=*/true);
		EXPECT_NE(producer.data(), nullptr);
		EXPECT_EQ(producer.size(), size);
		EXPECT_EQ(producer.name(), name);
	});
	// Destructor calls shm_unlink — segment is gone after this scope
}

TEST(ShmMappingTest, ConsumerOpensExistingSegment)
{
	const std::string name = "/oe.test.shm_consumer";
	constexpr std::size_t size = 4096;

	// Producer must outlive the consumer for this test
	ShmMapping producer(name, size, /*create=*/true);

	EXPECT_NO_THROW({
		ShmMapping consumer(name, size, /*create=*/false);
		EXPECT_NE(consumer.data(), nullptr);
		EXPECT_EQ(consumer.size(), size);
	});
}

TEST(ShmMappingTest, ProducerWriteVisibleToConsumer)
{
	const std::string name = "/oe.test.shm_rw";
	constexpr std::size_t size = 64;

	ShmMapping producer(name, size, /*create=*/true);
	ShmMapping consumer(name, size, /*create=*/false);

	// Write a sentinel value via the producer
	constexpr uint8_t kSentinel = 0xAB;
	std::memset(producer.data(), kSentinel, size);

	// Consumer must see the same bytes — both map MAP_SHARED
	const auto* consumerBytes = static_cast<const uint8_t*>(consumer.data());
	for (std::size_t i = 0; i < size; ++i) {
		ASSERT_EQ(consumerBytes[i], kSentinel)
			<< "Mismatch at byte " << i;
	}
}

TEST(ShmMappingTest, MoveConstructorTransfersOwnership)
{
	const std::string name = "/oe.test.shm_move";
	constexpr std::size_t size = 1024;

	ShmMapping original(name, size, /*create=*/true);
	void* const originalPtr = original.data();

	ShmMapping moved(std::move(original));

	// Moved-from object should be in a valid but empty state
	EXPECT_EQ(original.data(), nullptr);
	EXPECT_EQ(original.size(), 0u);

	// Moved-to object retains the mapping
	EXPECT_EQ(moved.data(), originalPtr);
	EXPECT_EQ(moved.size(), size);
	EXPECT_EQ(moved.name(), name);
}

TEST(ShmMappingTest, OpenNonExistentSegmentThrows)
{
	EXPECT_THROW(
		ShmMapping("/oe.test.does_not_exist_xyz", 4096, /*create=*/false),
		std::runtime_error);
}

TEST(ShmMappingTest, ShmVideoHeaderRoundTrip)
{
	const std::string name = "/oe.test.shm_header";
	const std::size_t size = ShmVideoHeader::segmentSize(320, 240);

	ShmMapping producer(name, size, /*create=*/true);
	ShmMapping consumer(name, size, /*create=*/false);

	// Producer writes header
	auto* videoHeader = static_cast<ShmVideoHeader*>(producer.data());
	videoHeader->width        = 320;
	videoHeader->height       = 240;
	videoHeader->bytesPerPixel = 3;
	videoHeader->seqNumber    = 42;
	videoHeader->timestampNs  = 123456789ULL;

	// Consumer reads back the same header
	const auto* consumerHeader = static_cast<const ShmVideoHeader*>(consumer.data());
	EXPECT_EQ(consumerHeader->width,        320u);
	EXPECT_EQ(consumerHeader->height,       240u);
	EXPECT_EQ(consumerHeader->bytesPerPixel, 3u);
	EXPECT_EQ(consumerHeader->seqNumber,    42u);
	EXPECT_EQ(consumerHeader->timestampNs,  123456789ULL);
}


// ---------------------------------------------------------------------------
// Huge pages + mlock (best-effort memory optimizations)
// ---------------------------------------------------------------------------

TEST(ShmMappingTest, MadviseAndMlockSucceedOnNormalSegment)
{
	// A 4 MB segment exceeds kHugePageThreshold — both madvise and mlock
	// should be attempted.  This test verifies the constructor does not
	// throw or crash when the optimizations run.
	const std::string name = "/oe.test.shm_hugepage";
	constexpr std::size_t size = 4 * 1024 * 1024;  // 4 MiB

	EXPECT_NO_THROW({
		ShmMapping producer(name, size, /*create=*/true);
		EXPECT_NE(producer.data(), nullptr);
		EXPECT_EQ(producer.size(), size);
	});
}

TEST(ShmMappingTest, SmallSegmentSkipsHugePages)
{
	// A 4 KB segment is below kHugePageThreshold.  madvise(MADV_HUGEPAGE)
	// should NOT be called, but mlock should still be attempted.
	// This test ensures no crash or throw for small segments.
	const std::string name = "/oe.test.shm_small_no_hugepage";
	constexpr std::size_t size = 4096;

	EXPECT_NO_THROW({
		ShmMapping producer(name, size, /*create=*/true);
		EXPECT_NE(producer.data(), nullptr);
	});
}

TEST(ShmMappingTest, MlockFailureIsNonFatal)
{
	// Even if RLIMIT_MEMLOCK is too low for a large segment, the
	// constructor must still succeed.  On CI with default rlimits,
	// a 25 MB mlock may fail — but the mapping must still be usable.
	const std::string name = "/oe.test.shm_mlock_graceful";
	constexpr std::size_t size = 25 * 1024 * 1024;  // 25 MiB

	EXPECT_NO_THROW({
		ShmMapping producer(name, size, /*create=*/true);
		EXPECT_NE(producer.data(), nullptr);
		EXPECT_EQ(producer.size(), size);

		// Verify the mapping is still functional despite potential mlock failure
		auto* bytes = producer.bytes();
		bytes[0] = 0xAB;
		bytes[size - 1] = 0xCD;
		EXPECT_EQ(bytes[0], 0xAB);
		EXPECT_EQ(bytes[size - 1], 0xCD);
	});
}

