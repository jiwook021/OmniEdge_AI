#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — SHM BGR24 Frame Writer
//
// Counterpart to shm_frame_reader.hpp. Provides a utility for writing
// BGR24 frames into a ShmCircularBuffer<ShmVideoHeader> output segment.
//
// Used by CV nodes in BGR24 output mode (pipeline chaining).
// ---------------------------------------------------------------------------

#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/constants/video_constants.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>


/// Write a BGR24 frame to a ShmCircularBuffer<ShmVideoHeader> output segment.
///
/// @param buf       The circular buffer (producer-side, create=true).
/// @param bgrData   Pointer to BGR24 host bytes (width * height * 3).
/// @param width     Frame width in pixels.
/// @param height    Frame height in pixels.
/// @param seqNumber Monotonic frame counter.
/// @return          The slot index written to (for diagnostics).
inline uint32_t writeBgrFrame(ShmCircularBuffer<ShmVideoHeader>& buf,
                              const uint8_t* bgrData,
                              uint32_t width,
                              uint32_t height,
                              uint64_t seqNumber)
{
	auto [slotPtr, slotIdx] = buf.acquireWriteSlot();

	const std::size_t frameBytes =
		static_cast<std::size_t>(width) * height * kBgr24BytesPerPixel;
	std::memcpy(slotPtr, bgrData, frameBytes);

	// Update the header
	auto* hdr = buf.header();
	hdr->width        = width;
	hdr->height       = height;
	hdr->bytesPerPixel = kBgr24BytesPerPixel;
	hdr->seqNumber    = seqNumber;
	hdr->timestampNs  = static_cast<uint64_t>(
		std::chrono::steady_clock::now().time_since_epoch().count());

	buf.commitWrite();
	return slotIdx;
}
