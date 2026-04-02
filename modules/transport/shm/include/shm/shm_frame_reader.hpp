#pragma once

// Shared SHM BGR frame reader utilities.
//
// The video ingest SHM segment uses a ShmCircularBuffer layout:
//   [ShmVideoHeader (64 B)][ShmCircularControl (64 B)][slot 0 BGR24]...[slot N-1]
// The control region contains writePos/readPos for lock-free SPSC access.

#include "shm/shm_circular_buffer.hpp"
#include "shm/shm_mapping.hpp"
#include "common/constants/video_constants.hpp"

#include <cstdint>


struct BgrFrameView {
    const uint8_t* data = nullptr;
    uint32_t width  = 0;
    uint32_t height = 0;
    uint32_t slotIndex = 0;   // stale-read guard: compare before/after inference
};

inline BgrFrameView readLatestBgrFrame(const ShmMapping& shm)
{
    const auto* videoHeader = reinterpret_cast<const ShmVideoHeader*>(shm.bytes());
    if (videoHeader->width == 0 || videoHeader->height == 0) return {};

    ShmCircularReader reader(shm, sizeof(ShmVideoHeader));
    const uint32_t slot = reader.latestSlotIndex();
    const uint8_t* frameData = reader.slotData(slot);

    return {frameData, videoHeader->width, videoHeader->height, slot};
}

/// Returns the latest slot index — used as a stale-read guard after inference.
inline uint32_t currentShmSlotIndex(const ShmMapping& shm)
{
    ShmCircularReader reader(shm, sizeof(ShmVideoHeader));
    return reader.latestSlotIndex();
}

