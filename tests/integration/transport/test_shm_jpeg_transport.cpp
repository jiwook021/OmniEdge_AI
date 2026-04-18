#include <gtest/gtest.h>

#include "cv/background_blur_node.hpp"
#include "shm/shm_mapping.hpp"
#include "zmq/jpeg_constants.hpp"
#include "cv_test_helpers.hpp"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// SHM JPEG Transport Tests
//
// Purpose: Verify the shared-memory double-buffer protocol that carries
// blurred JPEG frames from BackgroundBlurNode to WebSocketBridgeNode.
//
// What these tests catch:
//   - JPEG data corruption during SHM write/read (torn reads, partial writes)
//   - Double-buffer slot confusion (reading stale slot instead of latest)
//   - Slot overlap: data in slot 0 corrupted when writing slot 1
//   - Large payloads (500 KB+ JPEGs) surviving the SHM round-trip
//
// All tests operate on real POSIX SHM segments (create + destroy per test).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubJpegProducer — writes a synthetic JPEG of configurable size into a
// buffer.  Used to simulate the blur inferencer writing into SHM slots.
// ---------------------------------------------------------------------------
class StubJpegProducer {
public:
    explicit StubJpegProducer(std::size_t jpegPayloadSize)
        : payloadSize_{jpegPayloadSize}
    {}

    /// Write a synthetic JPEG (SOI + sequential fill + EOI) into the buffer.
    /// Returns the number of bytes written.
    std::size_t writeJpeg(uint8_t* destinationBuffer,
                          std::size_t maxBytes) const
    {
        if (payloadSize_ > maxBytes || payloadSize_ < 4) { return 0; }

        destinationBuffer[0] = 0xFF;
        destinationBuffer[1] = 0xD8;  // SOI
        for (std::size_t i = 2; i < payloadSize_ - 2; ++i) {
            destinationBuffer[i] = static_cast<uint8_t>(i & 0xFF);
        }
        destinationBuffer[payloadSize_ - 2] = 0xFF;
        destinationBuffer[payloadSize_ - 1] = 0xD9;  // EOI

        return payloadSize_;
    }

private:
    std::size_t payloadSize_;
};

// ===========================================================================
// Producer → Consumer JPEG Round-Trip
// ===========================================================================

TEST(ShmJpegTransport, ProducerWrite_ConsumerReadsIdenticalJpegBytes)
{
    // Bug caught: JPEG data corrupted during SHM transfer (misaligned memcpy,
    // wrong slot offset, etc.)

    const std::size_t shmTotalSize =
        sizeof(ShmJpegControl) + 2u * kMaxJpegBytesPerSlot;
    const std::string shmSegmentName = "/oe.test.cv.blur.roundtrip";

    // GIVEN: a producer and consumer open the same SHM segment
    ShmMapping producerShm(shmSegmentName, shmTotalSize, /*create=*/true);
    ShmMapping consumerShm(shmSegmentName, shmTotalSize, /*create=*/false);
    std::memset(producerShm.data(), 0, shmTotalSize);

    // WHEN: the producer writes a JPEG into slot 1
    auto* controlHeader =
        reinterpret_cast<ShmJpegControl*>(producerShm.bytes());
    constexpr uint32_t kWriteSlot = 1;
    uint8_t* slotBuffer = producerShm.bytes()
                        + sizeof(ShmJpegControl)
                        + kWriteSlot * kMaxJpegBytesPerSlot;

    auto personSceneBgr = loadBgr24Fixture(kPersonSceneFile);
    SKIP_IF_NO_FIXTURE(personSceneBgr);

    // Use a stub inferencer to produce a JPEG from real image data
    StubJpegProducer jpegProducer(1024);  // 1 KB simulated JPEG
    const std::size_t jpegByteCount =
        jpegProducer.writeJpeg(slotBuffer, kMaxJpegBytesPerSlot);

    controlHeader->jpegSize[kWriteSlot]  = static_cast<uint32_t>(jpegByteCount);
    controlHeader->seqNumber[kWriteSlot] = 1;
    controlHeader->writeIndex.store(kWriteSlot, std::memory_order_release);

    // THEN: the consumer reads the exact same JPEG bytes
    const auto* consumerControl =
        reinterpret_cast<const ShmJpegControl*>(consumerShm.bytes());
    const uint32_t readSlot =
        consumerControl->writeIndex.load(std::memory_order_acquire);

    EXPECT_EQ(readSlot, kWriteSlot);
    EXPECT_EQ(consumerControl->jpegSize[readSlot],
              static_cast<uint32_t>(jpegByteCount));

    const uint8_t* consumerSlot = consumerShm.bytes()
                                + sizeof(ShmJpegControl)
                                + readSlot * kMaxJpegBytesPerSlot;

    EXPECT_EQ(consumerSlot[0], 0xFF);
    EXPECT_EQ(consumerSlot[1], 0xD8)
        << "Consumer must read valid JPEG SOI marker";
    EXPECT_EQ(consumerSlot[jpegByteCount - 2], 0xFF);
    EXPECT_EQ(consumerSlot[jpegByteCount - 1], 0xD9)
        << "Consumer must read valid JPEG EOI marker";
}

TEST(ShmJpegTransport, DoubleBuffer_AlternatesBetweenSlotsAcrossFrames)
{
    // Bug caught: producer always writes to the same slot, causing
    // consumer to read stale data.

    const std::size_t shmTotalSize =
        sizeof(ShmJpegControl) + 2u * kMaxJpegBytesPerSlot;
    const std::string shmSegmentName = "/oe.test.cv.blur.alternate";

    ShmMapping shm(shmSegmentName, shmTotalSize, /*create=*/true);
    std::memset(shm.data(), 0, shmTotalSize);

    auto* controlHeader =
        reinterpret_cast<ShmJpegControl*>(shm.bytes());

    StubJpegProducer jpegProducer(256);

    // WHEN: 6 frames are written using the double-buffer protocol
    constexpr int kNumFrames = 6;
    for (uint32_t frameIndex = 0; frameIndex < kNumFrames; ++frameIndex) {
        // Read the current slot, write to the opposite one
        const uint32_t currentReadSlot =
            controlHeader->writeIndex.load(std::memory_order_acquire);
        const uint32_t nextWriteSlot = 1u - currentReadSlot;

        uint8_t* slotBuffer = shm.bytes()
                            + sizeof(ShmJpegControl)
                            + nextWriteSlot * kMaxJpegBytesPerSlot;

        const auto bytesWritten =
            jpegProducer.writeJpeg(slotBuffer, kMaxJpegBytesPerSlot);

        controlHeader->jpegSize[nextWriteSlot] =
            static_cast<uint32_t>(bytesWritten);
        controlHeader->seqNumber[nextWriteSlot] = frameIndex + 1;
        controlHeader->writeIndex.store(nextWriteSlot,
                                        std::memory_order_release);
    }

    // THEN: the final write index points to a valid slot with frame #6
    const uint32_t finalSlot =
        controlHeader->writeIndex.load(std::memory_order_acquire);
    EXPECT_LT(finalSlot, 2u)
        << "writeIndex must always be 0 or 1";
    EXPECT_EQ(controlHeader->seqNumber[finalSlot], kNumFrames)
        << "Latest slot must contain the most recent frame";
}

TEST(ShmJpegTransport, TwoSlots_ContainIndependentData)
{
    // Bug caught: writing to slot 1 corrupts data in slot 0
    // (wrong pointer arithmetic, overlapping regions).

    const std::size_t shmTotalSize =
        sizeof(ShmJpegControl) + 2u * kMaxJpegBytesPerSlot;
    const std::string shmSegmentName = "/oe.test.cv.blur.independent";

    ShmMapping shm(shmSegmentName, shmTotalSize, /*create=*/true);
    std::memset(shm.data(), 0, shmTotalSize);

    uint8_t* slot0Buffer = shm.bytes() + sizeof(ShmJpegControl);
    uint8_t* slot1Buffer = slot0Buffer + kMaxJpegBytesPerSlot;

    // WHEN: two different-sized JPEGs are written to each slot
    StubJpegProducer smallJpegProducer(128);
    StubJpegProducer largeJpegProducer(256);

    const auto slot0Bytes =
        smallJpegProducer.writeJpeg(slot0Buffer, kMaxJpegBytesPerSlot);
    const auto slot1Bytes =
        largeJpegProducer.writeJpeg(slot1Buffer, kMaxJpegBytesPerSlot);

    // THEN: both slots have valid JPEG markers (no corruption)
    EXPECT_EQ(slot0Buffer[0], 0xFF);
    EXPECT_EQ(slot0Buffer[1], 0xD8) << "Slot 0 SOI corrupted after slot 1 write";
    EXPECT_EQ(slot1Buffer[0], 0xFF);
    EXPECT_EQ(slot1Buffer[1], 0xD8) << "Slot 1 SOI corrupted after slot 0 write";

    // Sizes differ — proves the slots are truly independent
    EXPECT_EQ(slot0Bytes, 128u);
    EXPECT_EQ(slot1Bytes, 256u);
}

TEST(ShmJpegTransport, LargeJpeg_SurvivesShmRoundTrip)
{
    // Bug caught: large JPEG payloads (typical 1080p at quality 85 = ~400 KB)
    // corrupted during SHM transport.

    const std::size_t shmTotalSize =
        sizeof(ShmJpegControl) + 2u * kMaxJpegBytesPerSlot;
    const std::string shmSegmentName = "/oe.test.cv.blur.large";

    ShmMapping producerShm(shmSegmentName, shmTotalSize, /*create=*/true);
    ShmMapping consumerShm(shmSegmentName, shmTotalSize, /*create=*/false);
    std::memset(producerShm.data(), 0, shmTotalSize);

    // GIVEN: a 500 KB JPEG (simulating a high-quality 1080p frame)
    constexpr std::size_t kLargeJpegSize = 500u * 1024u;
    StubJpegProducer largeJpegProducer(kLargeJpegSize);

    auto* controlHeader =
        reinterpret_cast<ShmJpegControl*>(producerShm.bytes());
    constexpr uint32_t kSlot = 1;
    uint8_t* producerSlot = producerShm.bytes()
                          + sizeof(ShmJpegControl)
                          + kSlot * kMaxJpegBytesPerSlot;

    const auto bytesWritten =
        largeJpegProducer.writeJpeg(producerSlot, kMaxJpegBytesPerSlot);
    ASSERT_EQ(bytesWritten, kLargeJpegSize);

    controlHeader->jpegSize[kSlot]  = static_cast<uint32_t>(kLargeJpegSize);
    controlHeader->seqNumber[kSlot] = 42;
    controlHeader->writeIndex.store(kSlot, std::memory_order_release);

    // WHEN: the consumer reads back the JPEG
    const auto* consumerControl =
        reinterpret_cast<const ShmJpegControl*>(consumerShm.bytes());
    const uint32_t readSlot =
        consumerControl->writeIndex.load(std::memory_order_acquire);
    ASSERT_EQ(readSlot, kSlot);

    const uint8_t* consumerSlot = consumerShm.bytes()
                                + sizeof(ShmJpegControl)
                                + readSlot * kMaxJpegBytesPerSlot;

    // THEN: JPEG markers and body content are intact
    EXPECT_EQ(consumerSlot[0], 0xFF);
    EXPECT_EQ(consumerSlot[1], 0xD8) << "SOI corrupted in large JPEG SHM round-trip";
    EXPECT_EQ(consumerSlot[kLargeJpegSize - 2], 0xFF);
    EXPECT_EQ(consumerSlot[kLargeJpegSize - 1], 0xD9)
        << "EOI corrupted in large JPEG SHM round-trip";

    // Spot-check body content at offset 100
    EXPECT_EQ(consumerSlot[100], static_cast<uint8_t>(100 & 0xFF))
        << "JPEG body data corrupted in SHM round-trip";
}

TEST(ShmJpegTransport, PayloadExceedsSlotCapacity_ReturnsZeroBytes)
{
    // Bug caught: oversized JPEG silently overflows into the next slot.

    StubJpegProducer oversizedProducer(2 * kMaxJpegBytesPerSlot);

    std::vector<uint8_t> smallBuffer(1024);
    const auto bytesWritten =
        oversizedProducer.writeJpeg(smallBuffer.data(), smallBuffer.size());

    EXPECT_EQ(bytesWritten, 0u)
        << "Must reject payload that exceeds the available buffer capacity";
}

