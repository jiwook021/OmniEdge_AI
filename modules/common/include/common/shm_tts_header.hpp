#pragma once

#include <cstddef>
#include <cstdint>

#include "zmq/audio_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — ShmTtsHeader
//
// POD header at byte offset 0 of the /oe.aud.tts shared-memory segment.
// Shared between TTSNode (producer) and WebSocketBridgeNode (consumer).
//
// Layout: [ShmTtsHeader (24 bytes)][float32 PCM samples ...]
// ---------------------------------------------------------------------------

/**
 * @brief POD header at the start of the TTS audio SHM segment.
 *
 * All fields are fixed-width; no padding inserted between them.
 */
struct ShmTtsHeader {
	uint32_t sampleRateHz = kTtsOutputSampleRateHz;  ///< Always 24 kHz for Kokoro output
	uint32_t numSamples   = 0;       ///< Number of float32 samples following
	uint64_t seqNumber    = 0;       ///< Monotonic synthesis sequence number
	uint64_t timestampNs  = 0;       ///< Steady-clock nanoseconds at write

	/** @brief Byte offset where float32 sample data begins. */
	[[nodiscard]] static constexpr std::size_t dataOffset() noexcept
	{
		return sizeof(ShmTtsHeader);
	}

	/** @brief Total SHM bytes needed for n float32 samples. */
	[[nodiscard]] static constexpr std::size_t segmentSize(
		std::size_t nSamples) noexcept
	{
		return sizeof(ShmTtsHeader) + nSamples * sizeof(float);
	}
};

static_assert(sizeof(ShmTtsHeader) == 24,
	"ShmTtsHeader must be exactly 24 bytes");
static_assert(sizeof(ShmTtsHeader) % 8 == 0,
	"ShmTtsHeader must be 8-byte aligned");
