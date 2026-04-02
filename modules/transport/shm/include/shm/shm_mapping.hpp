#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "common/constants/video_constants.hpp"
#include "zmq/audio_constants.hpp"

/// Minimum segment size for requesting transparent huge pages via
/// madvise(MADV_HUGEPAGE).  Segments smaller than this (e.g. the 8 KB
/// audio ring buffer) are left on regular 4 KB pages.  2 MiB matches
/// the x86-64 huge page size.
inline constexpr std::size_t kHugePageThreshold = 2 * 1024 * 1024;  // 2 MiB

// ---------------------------------------------------------------------------
// OmniEdge_AI — POSIX Shared Memory RAII Wrapper
//
// Design constraints:
//   1. Layout must be FLAT — no raw pointers inside the mapped region.
//   2. Producer vs. consumer mode:
//        create = true  -> shm_open(O_CREAT|O_RDWR) + ftruncate + mmap
//        create = false -> shm_open(O_RDWR) + mmap
//   3. Mapped addresses are process-specific — never store absolute pointers.
//   4. SHM names follow /oe.<layer>.<module>[.<qualifier>]
// ---------------------------------------------------------------------------


class ShmMapping {
public:
    ShmMapping(std::string_view name, std::size_t size, bool create);
    ~ShmMapping();

    ShmMapping(const ShmMapping&)            = delete;
    ShmMapping& operator=(const ShmMapping&) = delete;

    ShmMapping(ShmMapping&&) noexcept;
    ShmMapping& operator=(ShmMapping&&) noexcept;

    [[nodiscard]] void*       data()    const noexcept { return mappedRegion_;  }
    [[nodiscard]] uint8_t*    bytes()   const noexcept {
        return static_cast<uint8_t*>(mappedRegion_);
    }
    [[nodiscard]] std::size_t size()    const noexcept { return mappedSize_; }
    [[nodiscard]] const std::string& name() const noexcept { return segmentName_; }

private:
    void reset() noexcept;

    void*       mappedRegion_ = nullptr;
    std::size_t mappedSize_   = 0;
    std::string segmentName_;
    bool        isCreator_    = false;
};

// ---------------------------------------------------------------------------
// SHM POD Headers
// ---------------------------------------------------------------------------

struct ShmVideoHeader {
    uint32_t width         = 0;
    uint32_t height        = 0;
    uint32_t bytesPerPixel = kBgr24BytesPerPixel;
    uint32_t _pad0         = 0;
    uint64_t seqNumber     = 0;       ///< Monotonic frame sequence number
    uint64_t timestampNs   = 0;       ///< Capture timestamp (CLOCK_MONOTONIC ns)
    uint64_t _reserved[4]  = {};      ///< Reserved for future use (maintains 64-byte layout)

    [[nodiscard]] static constexpr std::size_t dataOffset() noexcept
    {
        return sizeof(ShmVideoHeader);
    }

    [[nodiscard]] static constexpr std::size_t segmentSize(
        uint32_t frameWidth, uint32_t frameHeight) noexcept
    {
        return sizeof(ShmVideoHeader) +
               static_cast<std::size_t>(frameWidth) * frameHeight * kBgr24BytesPerPixel;
    }
};

static_assert(sizeof(ShmVideoHeader) == 64,
    "ShmVideoHeader must be exactly 64 bytes");
static_assert(sizeof(ShmVideoHeader) % 8 == 0,
    "ShmVideoHeader must be 8-byte aligned");

struct ShmAudioHeader {
    uint32_t sampleRateHz = kSttInputSampleRateHz;
    uint32_t numSamples   = 0;
    uint64_t seqNumber    = 0;
    uint64_t timestampNs  = 0;

    [[nodiscard]] static constexpr std::size_t dataOffset() noexcept
    {
        return sizeof(ShmAudioHeader);
    }

    [[nodiscard]] static constexpr std::size_t segmentSize(
        uint32_t nSamples) noexcept
    {
        return sizeof(ShmAudioHeader) +
               static_cast<std::size_t>(nSamples) * sizeof(float);
    }
};

static_assert(sizeof(ShmAudioHeader) == 24,
    "ShmAudioHeader must be exactly 24 bytes");
static_assert(sizeof(ShmAudioHeader) % 8 == 0,
    "ShmAudioHeader must be 8-byte aligned");

struct ShmJpegControl {
    std::atomic<uint32_t> writeIndex{0};
    std::atomic<uint32_t> jpegSize[2]{};
    std::atomic<uint32_t> seqNumber[2]{};
    uint32_t              _pad[11]{};
};

static_assert(sizeof(ShmJpegControl) == 64,
    "ShmJpegControl must be exactly 64 bytes (one cache line)");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
    "SHM atomics must be lock-free for cross-process correctness");

// ---------------------------------------------------------------------------
// Per-slot metadata for ShmCircularBuffer — lives at the start of each
// slot's data region.  Keeps the global header at a fixed size regardless
// of the slot count.
// ---------------------------------------------------------------------------

struct ShmSlotMetadata {
    uint64_t seqNumber{0};
    uint64_t timestampNs{0};
};

static_assert(sizeof(ShmSlotMetadata) == 16,
    "ShmSlotMetadata must be exactly 16 bytes");

