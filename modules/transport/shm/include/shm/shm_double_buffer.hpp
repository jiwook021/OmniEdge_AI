#pragma once

#include "shm/shm_mapping.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

// ---------------------------------------------------------------------------
// OmniEdge_AI — ShmDoubleBuffer<Header>
//
// Typed RAII wrapper over ShmMapping that encapsulates double-buffer
// slot management with atomic writeIndex, stale-read guards, and
// typed header access.
//
// Usage (producer — create=true):
//   ShmDoubleBuffer<ShmVideoHeader> buf("/oe.vid.ingest", totalSize, true);
//   auto [header, data] = buf.writeSlot();
//   // fill data...
//   header->seqNumber = seq;
//   buf.flipSlot();
//
// Usage (consumer — create=false):
//   ShmDoubleBuffer<ShmVideoHeader> buf("/oe.vid.ingest", totalSize, false);
//   auto [header, data, slot] = buf.readSlot();
//   // use data (copy quickly — producer may flip soon)
//
// The control region is the first 64 bytes of the SHM segment. It
// contains one std::atomic<uint32_t> writeIndex at offset 0.
// ---------------------------------------------------------------------------


inline constexpr std::size_t kDoubleBufferControlBytes = 64;

/**
 * @brief Double-buffered POSIX SHM wrapper with typed header access.
 *
 * @tparam Header  POD header type (e.g. ShmVideoHeader, ShmJpegControl).
 *                 Must be trivially copyable and ≤ kDoubleBufferControlBytes
 *                 or at a known offset after the control region.
 */
template <typename Header>
class ShmDoubleBuffer {
public:
    /**
     * @brief Construct a double-buffered SHM segment.
     *
     * @param name    POSIX SHM name (e.g. "/oe.vid.ingest").
     * @param size    Total segment size in bytes (control + header + 2 × slot).
     * @param create  True for producer (creates segment), false for consumer.
     */
    ShmDoubleBuffer(std::string_view name, std::size_t size, bool create)
        : mapping_(name, size, create)
    {
        auto* base = mapping_.bytes();

        // Control region at offset 0: atomic<uint32_t> writeIndex.
        controlIndex_ = reinterpret_cast<std::atomic<uint32_t>*>(base);

        // Header immediately after the control region.
        header_ = reinterpret_cast<Header*>(base + kDoubleBufferControlBytes);
    }

    ~ShmDoubleBuffer() = default;

    ShmDoubleBuffer(ShmDoubleBuffer&& other) noexcept
        : mapping_(std::move(other.mapping_))
        , controlIndex_(other.controlIndex_)
        , header_(other.header_)
    {
        other.controlIndex_ = nullptr;
        other.header_       = nullptr;
    }

    ShmDoubleBuffer& operator=(ShmDoubleBuffer&& other) noexcept
    {
        if (this != &other) {
            mapping_      = std::move(other.mapping_);
            controlIndex_ = other.controlIndex_;
            header_       = other.header_;
            other.controlIndex_ = nullptr;
            other.header_       = nullptr;
        }
        return *this;
    }

    ShmDoubleBuffer(const ShmDoubleBuffer&)            = delete;
    ShmDoubleBuffer& operator=(const ShmDoubleBuffer&) = delete;

    // ── Accessors ──────────────────────────────────────────────────────────

    /** @brief Typed header pointer (after control region). */
    [[nodiscard]] Header*       header()       noexcept { return header_; }
    [[nodiscard]] const Header* header() const noexcept { return header_; }

    /** @brief Current write index (0 or 1). */
    [[nodiscard]] uint32_t writeIndex() const noexcept
    {
        return controlIndex_->load(std::memory_order_acquire) % 2;
    }

    /** @brief Total SHM segment size. */
    [[nodiscard]] std::size_t totalSize() const noexcept
    {
        return mapping_.size();
    }

    /** @brief SHM segment name. */
    [[nodiscard]] const std::string& name() const noexcept
    {
        return mapping_.name();
    }

    /** @brief Raw pointer to the entire mapped region. */
    [[nodiscard]] uint8_t* rawBytes() noexcept { return mapping_.bytes(); }
    [[nodiscard]] const uint8_t* rawBytes() const noexcept { return mapping_.bytes(); }

    // ── Double-Buffer Protocol ─────────────────────────────────────────────

    /**
     * @brief Get the data region for the write slot (producer side).
     *
     * The producer writes to the slot that is NOT currently being read.
     * After writing, call flipSlot() to make the new data visible.
     *
     * @param slotSize  Size of one slot's data region in bytes.
     * @param dataOffset  Offset from segment start to the data region.
     * @return Pointer to the write slot's data, and the slot index.
     */
    [[nodiscard]] std::pair<uint8_t*, uint32_t>
    writeSlotData(std::size_t slotSize, std::size_t dataOffset) noexcept
    {
        const uint32_t readSlot  = writeIndex();
        const uint32_t writeSlot = 1 - readSlot;
        auto* slotPtr = mapping_.bytes() + dataOffset + writeSlot * slotSize;
        return {slotPtr, writeSlot};
    }

    /**
     * @brief Get the data region for the read slot (consumer side).
     *
     * Implements the stale-read guard: reads writeIndex, accesses data,
     * then re-reads writeIndex. If it changed, the read may be stale.
     *
     * @param slotSize  Size of one slot's data region in bytes.
     * @param dataOffset  Offset from segment start to the data region.
     * @return Pointer to the read slot's data, slot index, and whether the read is valid.
     */
    struct ReadResult {
        const uint8_t* data;
        uint32_t       slot;
        bool           valid;   ///< false if producer flipped during read
    };

    [[nodiscard]] ReadResult
    readSlotData(std::size_t slotSize, std::size_t dataOffset) const noexcept
    {
        const uint32_t slotBefore = writeIndex();
        const auto* slotPtr = mapping_.bytes() + dataOffset + slotBefore * slotSize;

        // The stale-read guard: re-read writeIndex after accessing the pointer.
        // Caller should copy data quickly, then check valid.
        // For zero-copy reads, the caller can re-check after copying.
        const uint32_t slotAfter = writeIndex();

        return {slotPtr, slotBefore, slotBefore == slotAfter};
    }

    /**
     * @brief Flip the write index to make the newly written slot visible.
     *
     * Must be called by the producer after writing data to the write slot.
     */
    void flipSlot() noexcept
    {
        const uint32_t current = controlIndex_->load(std::memory_order_acquire);
        controlIndex_->store(current + 1, std::memory_order_release);
    }

private:
    ShmMapping                 mapping_;
    std::atomic<uint32_t>*     controlIndex_ = nullptr;
    Header*                    header_        = nullptr;
};

