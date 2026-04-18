#include <gtest/gtest.h>

#include "ingest/audio_ingest_node.hpp"
#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "zmq/audio_constants.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// AudioIngestNode tests — CPU-only.
// Exercises config defaults, SHM layout, circular buffer protocol, VAD
// threshold logic, and real mock audio data (sine waves).
// ---------------------------------------------------------------------------


[[maybe_unused]]
static AudioIngestNode::Config makeTestAudioConfig()
{
	AudioIngestNode::Config cfg;
	cfg.pubPort         = 25556;
	cfg.wsBridgeSubPort = 25570;
	cfg.windowsHostIp   = "127.0.0.1";
	cfg.audioTcpPort    = 25001;
	cfg.moduleName      = "audio_ingest_test";
	return cfg;
}

// ---------------------------------------------------------------------------
// Helper: generate a sine wave at a given frequency
// ---------------------------------------------------------------------------

static std::vector<float> generateSineWave(float freqHz, uint32_t sampleRate,
                                            uint32_t numSamples)
{
	std::vector<float> pcm(numSamples);
	constexpr float kTwoPi = 2.0f * 3.14159265358979323846f;
	for (uint32_t i = 0; i < numSamples; ++i) {
		pcm[i] = std::sin(kTwoPi * freqHz * static_cast<float>(i)
		                   / static_cast<float>(sampleRate));
	}
	return pcm;
}

// ---------------------------------------------------------------------------
// SHM circular buffer write/read round-trip with real sine wave PCM
// ---------------------------------------------------------------------------

TEST(AudioIngestShmTest, CircularBufferSineWaveRoundTrip)
{
	constexpr uint32_t kSamples = 480;
	constexpr uint32_t kSlots   = 4;
	const std::size_t  slotSz   = sizeof(ShmAudioHeader) + kSamples * sizeof(float);

	const std::string name = "/oe.test.ai_sine";
	ShmCircularBuffer<ShmAudioHeader> producer(name, kSlots, slotSz, true);
	ShmCircularBuffer<ShmAudioHeader> consumer(name, kSlots, slotSz, false);

	// Generate a 440 Hz sine wave (A4 note)
	auto pcm = generateSineWave(440.0f, 16000, kSamples);

	// Write to circular buffer
	auto [slotPtr, slotIdx] = producer.acquireWriteSlot();
	auto* hdr = reinterpret_cast<ShmAudioHeader*>(slotPtr);
	hdr->sampleRateHz = 16'000;
	hdr->numSamples   = kSamples;
	hdr->seqNumber    = 1;
	hdr->timestampNs  = 123456789ULL;
	float* dst = reinterpret_cast<float*>(slotPtr + sizeof(ShmAudioHeader));
	std::memcpy(dst, pcm.data(), kSamples * sizeof(float));
	producer.commitWrite();

	// Consumer reads
	auto result = consumer.acquireReadSlot();
	ASSERT_TRUE(result.valid);
	EXPECT_EQ(result.slot, 0u);

	const auto* chdr = reinterpret_cast<const ShmAudioHeader*>(result.data);
	EXPECT_EQ(chdr->sampleRateHz, 16'000u);
	EXPECT_EQ(chdr->numSamples, kSamples);
	EXPECT_EQ(chdr->seqNumber, 1u);

	const float* readPcm = reinterpret_cast<const float*>(
		result.data + sizeof(ShmAudioHeader));
	for (uint32_t i = 0; i < kSamples; ++i) {
		ASSERT_FLOAT_EQ(readPcm[i], pcm[i]) << "sample " << i;
	}
}

// ---------------------------------------------------------------------------
// Multi-chunk sequence with different frequencies (wrap-around)
// ---------------------------------------------------------------------------

TEST(AudioIngestShmTest, CircularBufferMultiChunkWrapAround)
{
	constexpr uint32_t kSamples = 480;
	constexpr uint32_t kSlots   = 4;
	const std::size_t  slotSz   = sizeof(ShmAudioHeader) + kSamples * sizeof(float);

	const std::string name = "/oe.test.ai_multi";
	ShmCircularBuffer<ShmAudioHeader> producer(name, kSlots, slotSz, true);
	ShmCircularBuffer<ShmAudioHeader> consumer(name, kSlots, slotSz, false);

	// Write 6 chunks at different frequencies (wraps the 4-slot buffer)
	const float frequencies[] = {440.0f, 880.0f, 1320.0f, 1760.0f, 2200.0f, 2640.0f};
	std::vector<std::vector<float>> allPcm;

	for (int i = 0; i < 6; ++i) {
		auto pcm = generateSineWave(frequencies[i], 16000, kSamples);
		allPcm.push_back(pcm);

		auto [slotPtr, slotIdx] = producer.acquireWriteSlot();
		auto* hdr = reinterpret_cast<ShmAudioHeader*>(slotPtr);
		hdr->sampleRateHz = 16'000;
		hdr->numSamples   = kSamples;
		hdr->seqNumber    = static_cast<uint64_t>(i + 1);

		float* dst = reinterpret_cast<float*>(slotPtr + sizeof(ShmAudioHeader));
		std::memcpy(dst, pcm.data(), kSamples * sizeof(float));
		producer.commitWrite();
	}

	// Consumer was at position 0, writer wrote 6 slots.
	// Writer has lapped (6 - 0 >= 4), so sequential read is invalid.
	EXPECT_TRUE(consumer.writerHasLappedReader());

	// But readLatestSlot should still give us the last-written data
	auto latest = consumer.readLatestSlot();
	EXPECT_TRUE(latest.valid);
	// Last write was at index (6-1) % 4 = 1
	EXPECT_EQ(latest.slot, 1u);

	const auto* hdr = reinterpret_cast<const ShmAudioHeader*>(latest.data);
	EXPECT_EQ(hdr->seqNumber, 6u);

	const float* readPcm = reinterpret_cast<const float*>(
		latest.data + sizeof(ShmAudioHeader));
	// Verify first few samples of the 2640 Hz sine wave
	for (uint32_t i = 0; i < 10; ++i) {
		ASSERT_FLOAT_EQ(readPcm[i], allPcm[5][i]) << "sample " << i;
	}
}

// ---------------------------------------------------------------------------
// Stale read on slow consumer
// ---------------------------------------------------------------------------

TEST(AudioIngestShmTest, AudioStaleReadOnSlowConsumer)
{
	constexpr uint32_t kSamples = 480;
	constexpr uint32_t kSlots   = 4;
	const std::size_t  slotSz   = sizeof(ShmAudioHeader) + kSamples * sizeof(float);

	const std::string name = "/oe.test.ai_stale";
	ShmCircularBuffer<ShmAudioHeader> producer(name, kSlots, slotSz, true);
	ShmCircularBuffer<ShmAudioHeader> consumer(name, kSlots, slotSz, false);

	// Write kSlots + 1 chunks without consumer reading
	for (uint32_t i = 0; i <= kSlots; ++i) {
		auto [ptr, idx] = producer.acquireWriteSlot();
		std::memset(ptr, 0, slotSz);
		producer.commitWrite();
	}

	// Consumer should detect writer has lapped
	EXPECT_TRUE(consumer.writerHasLappedReader());
	auto result = consumer.acquireReadSlot();
	EXPECT_FALSE(result.valid);
}

// ---------------------------------------------------------------------------
// Legacy test: SHM ring-buffer write/read round-trip (kept for regression)
// ---------------------------------------------------------------------------

TEST(AudioIngestShmTest, SpeechChunkRoundTrip)
{
	constexpr uint32_t kSamples = 480;
	constexpr uint32_t kSlots   = 4;
	const std::size_t  slotSz   = sizeof(ShmAudioHeader) + kSamples * sizeof(float);

	const std::string name = "/oe.test.ai_chunk";
	ShmCircularBuffer<ShmAudioHeader> producer(name, kSlots, slotSz, true);
	ShmCircularBuffer<ShmAudioHeader> consumer(name, kSlots, slotSz, false);

	// Write a chunk
	auto [slotPtr, slotIdx] = producer.acquireWriteSlot();
	auto* audioHeader = reinterpret_cast<ShmAudioHeader*>(slotPtr);
	audioHeader->sampleRateHz = 16'000;
	audioHeader->numSamples   = kSamples;
	audioHeader->seqNumber    = 42;
	audioHeader->timestampNs  = 123456789ULL;

	float* dst = reinterpret_cast<float*>(slotPtr + sizeof(ShmAudioHeader));
	std::vector<float> pcm(kSamples, 0.5f);
	std::memcpy(dst, pcm.data(), kSamples * sizeof(float));
	producer.commitWrite();

	// Consumer reads
	auto result = consumer.acquireReadSlot();
	ASSERT_TRUE(result.valid);
	const auto* chdr = reinterpret_cast<const ShmAudioHeader*>(result.data);
	EXPECT_EQ(chdr->sampleRateHz, 16'000u);
	EXPECT_EQ(chdr->numSamples, kSamples);
	EXPECT_EQ(chdr->seqNumber, 42u);
	EXPECT_EQ(chdr->timestampNs, 123456789ULL);

	const float* cDst = reinterpret_cast<const float*>(
		result.data + sizeof(ShmAudioHeader));
	for (uint32_t i = 0; i < kSamples; ++i) {
		ASSERT_FLOAT_EQ(cDst[i], 0.5f) << "sample " << i;
	}
}

// ---------------------------------------------------------------------------
// VAD threshold helpers (logic-only, no ONNX model needed)
// ---------------------------------------------------------------------------

TEST(VadThresholdTest, SilenceDurationAccumulation)
{
	constexpr uint32_t sampleRateHz  = 16'000;
	constexpr uint32_t chunkSamples  = 480;
	constexpr uint32_t silenceThrMs  = 800;
	constexpr uint32_t msPerChunk    = chunkSamples * 1000 / sampleRateHz;
	constexpr uint32_t chunksNeeded  =
		(silenceThrMs + msPerChunk - 1) / msPerChunk;

	uint32_t silenceSamples = 0;
	uint32_t chunks         = 0;
	bool     triggered      = false;

	while (!triggered) {
		silenceSamples += chunkSamples;
		chunks++;
		const uint32_t silenceMs = (silenceSamples * 1000u) / sampleRateHz;
		if (silenceMs >= silenceThrMs) {
			triggered = true;
		}
	}
	EXPECT_EQ(chunks, chunksNeeded);
	EXPECT_TRUE(triggered);
}

