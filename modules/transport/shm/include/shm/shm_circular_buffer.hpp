#pragma once

#include "shm/shm_mapping.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

// ---------------------------------------------------------------------------
// OmniEdge_AI — ShmCircularBuffer<Header>
//
// Lock-free SPSC (single-producer, single-consumer) circular buffer over
// POSIX shared memory.  
//
// SHM Layout (flat, POD, no pointers stored in segment):
//
//   Offset 0:             Header region (sizeof(Header) padded to 64-byte
//                         cache-line boundary)
//   Offset H:             ShmCircularControl (128 bytes, 2 cache lines)
//     - alignas(64) atomic<uint64_t>  writePos  (cache line 0, producer-only)
//     - alignas(64) atomic<uint64_t>  readPos   (cache line 1, consumer-only)
//     - uint32_t          slotCount             (cache line 1)
//     - uint32_t          slotByteSize          (cache line 1)
//   Offset H + 128:       slot[0] .. slot[slotCount-1]
//
//   where H = ((sizeof(Header) + 63) / 64) * 64
//
// Slot index is derived from position modulo slotCount.  writePos and
// readPos are uint64_t so they never overflow in practice, making the
// "writer lapped reader" check trivial: writePos - readPos >= slotCount.
//
// Usage (producer — create=true):
//   ShmCircularBuffer<ShmVideoHeader> buf("/oe.vid.ingest", 4, frameBytes, true);
//   auto [ptr, idx] = buf.acquireWriteSlot();
//   memcpy(ptr, frameData, frameBytes);
//   buf.commitWrite();
//
// Usage (consumer — create=false):
//   ShmCircularBuffer<ShmVideoHeader> buf("/oe.vid.ingest", 4, frameBytes, false);
//   auto result = buf.acquireReadSlot();
//   if (result.valid) { memcpy(dst, result.data, frameBytes); buf.advanceRead(); }
//
// Lightweight consumer (no need to know slot count at compile time):
//   ShmCircularReader reader(existingMapping, sizeof(ShmVideoHeader));
//   auto slot = reader.latestSlotIndex();  // std::optional<uint32_t>
//   if (slot) { const uint8_t* data = reader.slotData(*slot); }
// ---------------------------------------------------------------------------


// ShmSlotMetadata is defined in shm/shm_mapping.hpp (included above).

// ---------------------------------------------------------------------------
// ShmCircularControl — 128-byte control region (two cache lines to avoid false sharing)
// ---------------------------------------------------------------------------

struct ShmCircularControl {
	alignas(64) std::atomic<uint64_t> writePos{0};   // cache line 0 — producer only
	alignas(64) std::atomic<uint64_t> readPos{0};    // cache line 1 — consumer only
	uint32_t              slotCount{0};               //   (same line as readPos — both consumer-read)
	uint32_t              slotByteSize{0};
};

static_assert(sizeof(ShmCircularControl) == 128,
	"ShmCircularControl must be exactly 128 bytes (two cache lines)");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
	"SHM atomics must be lock-free for cross-process correctness");

// ---------------------------------------------------------------------------
// Helper: round up to 64-byte cache-line boundary
// ---------------------------------------------------------------------------

inline constexpr std::size_t alignToCacheLine(std::size_t n) noexcept
{
	return ((n + 63u) / 64u) * 64u;
}

// ---------------------------------------------------------------------------
// ShmCircularBuffer<Header>
// ---------------------------------------------------------------------------

template <typename Header>
class ShmCircularBuffer {
public:
	/// Padded header size (rounded up to cache-line boundary).
	static constexpr std::size_t kHeaderPaddedSize = alignToCacheLine(sizeof(Header));

	/// Offset from segment start to the control region.
	static constexpr std::size_t kControlOffset = kHeaderPaddedSize;

	/// Offset from segment start to the first slot.
	static constexpr std::size_t kSlotsOffset = kHeaderPaddedSize + sizeof(ShmCircularControl);

	/**
	 * @brief Compute total SHM segment size for given parameters.
	 */
	[[nodiscard]] static constexpr std::size_t segmentSize(
		uint32_t slotCount, std::size_t slotByteSize) noexcept
	{
		return kSlotsOffset + static_cast<std::size_t>(slotCount) * slotByteSize;
	}

	/**
	 * @brief Construct a circular-buffered SHM segment.
	 *
	 * @param name          POSIX SHM name (e.g. "/oe.vid.ingest").
	 * @param slotCount     Number of slots (must be >= 2).
	 * @param slotByteSize  Size of one slot's data region in bytes.
	 * @param create        True for producer (creates segment), false for consumer.
	 */
	explicit ShmCircularBuffer(std::string_view name, uint32_t slotCount,
	                          std::size_t slotByteSize, bool create)
		: mapping_(name, segmentSize(slotCount, slotByteSize), create)
		, slotCount_(slotCount)
		, slotByteSize_(slotByteSize)
	{
		if (slotCount < 2) {
			throw std::invalid_argument("ShmCircularBuffer requires at least 2 slots");
		}
		auto* base = mapping_.bytes();

		header_   = reinterpret_cast<Header*>(base);
		control_  = reinterpret_cast<ShmCircularControl*>(base + kControlOffset);
		slotBase_ = base + kSlotsOffset;

		if (create) {
			// Placement-new to value-initialise the control region (atomics + POD).
			new (control_) ShmCircularControl{};
			control_->slotCount    = slotCount;
			control_->slotByteSize = static_cast<uint32_t>(slotByteSize);
		} else {
			// Consumer validates layout metadata matches expectations
			if (control_->slotCount != slotCount) {
				throw std::invalid_argument("consumer/producer slot count mismatch");
			}
			if (control_->slotByteSize != static_cast<uint32_t>(slotByteSize)) {
				throw std::invalid_argument("consumer/producer slot byte size mismatch");
			}
		}
	}

	~ShmCircularBuffer() = default;

	ShmCircularBuffer(ShmCircularBuffer&& other) noexcept
		: mapping_(std::move(other.mapping_))
		, control_(other.control_)
		, header_(other.header_)
		, slotBase_(other.slotBase_)
		, slotCount_(other.slotCount_)
		, slotByteSize_(other.slotByteSize_)
	{
		other.control_  = nullptr;
		other.header_   = nullptr;
		other.slotBase_ = nullptr;
	}

	ShmCircularBuffer& operator=(ShmCircularBuffer&& other) noexcept
	{
		if (this != &other) {
			mapping_      = std::move(other.mapping_);
			control_      = other.control_;
			header_       = other.header_;
			slotBase_     = other.slotBase_;
			slotCount_    = other.slotCount_;
			slotByteSize_ = other.slotByteSize_;
			other.control_  = nullptr;
			other.header_   = nullptr;
			other.slotBase_ = nullptr;
		}
		return *this;
	}

	ShmCircularBuffer(const ShmCircularBuffer&)            = delete;
	ShmCircularBuffer& operator=(const ShmCircularBuffer&) = delete;

	// ── Accessors ──────────────────────────────────────────────────────────

	[[nodiscard]] Header*       header()       noexcept { return header_; }
	[[nodiscard]] const Header* header() const noexcept { return header_; }

	[[nodiscard]] uint32_t    slotCount()    const noexcept { return slotCount_; }
	[[nodiscard]] std::size_t slotByteSize() const noexcept { return slotByteSize_; }
	[[nodiscard]] std::size_t totalSize()    const noexcept { return mapping_.size(); }

	[[nodiscard]] const std::string& name() const noexcept { return mapping_.name(); }

	[[nodiscard]] uint64_t writePos() const noexcept
	{
		return control_->writePos.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint64_t readPos() const noexcept
	{
		return control_->readPos.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint8_t* rawBytes() noexcept { return mapping_.bytes(); }
	[[nodiscard]] const uint8_t* rawBytes() const noexcept { return mapping_.bytes(); }

	// ── Producer Side ──────────────────────────────────────────────────────

	/**
	 * @brief Acquire the next write slot.
	 *
	 * Returns a pointer to the slot's data region and the slot index.
	 * The producer writes payload here, then calls commitWrite().
	 * The slot is NOT visible to the consumer until commitWrite().
	 */
	[[nodiscard]] std::pair<uint8_t*, uint32_t> acquireWriteSlot() noexcept
	{
		const uint64_t wp = control_->writePos.load(std::memory_order_relaxed);
		const uint32_t idx = static_cast<uint32_t>(wp % slotCount_);
		return {slotBase_ + static_cast<std::size_t>(idx) * slotByteSize_, idx};
	}

	/**
	 * @brief Commit the write — makes the slot visible to the consumer.
	 *
	 * Increments writePos with release semantics so the consumer sees
	 * the written data when it loads writePos with acquire.
	 */
	void commitWrite() noexcept
	{
		control_->writePos.fetch_add(1, std::memory_order_release);
	}

	// ── Consumer Side ──────────────────────────────────────────────────────

	struct ReadResult {
		const uint8_t* data;
		uint32_t       slot;
		uint64_t       sequence;   ///< writePos at time of read
		bool           valid;      ///< false if writer has lapped reader
	};

	/**
	 * @brief Acquire the next read slot.
	 *
	 * Returns a pointer to the slot data, the slot index, the sequence
	 * (writePos value), and whether the read is valid.
	 *
	 * If the writer has lapped the reader, the reader is fast-forwarded
	 * to the oldest valid slot (wp - slotCount + 1) so the consumer is
	 * never permanently stuck.
	 * If no data is available (writePos == readPos), returns {nullptr, 0, 0, false}.
	 */
	[[nodiscard]] ReadResult acquireReadSlot() noexcept
	{
		uint64_t rp = control_->readPos.load(std::memory_order_relaxed);
		const uint64_t wp = control_->writePos.load(std::memory_order_acquire);

		// No data available
		if (wp <= rp) {
			return {nullptr, 0, wp, false};
		}

		// Writer has lapped reader — fast-forward and signal the drop
		if ((wp - rp) >= slotCount_) {
			rp = wp - slotCount_ + 1;
			control_->readPos.store(rp, std::memory_order_release);
			return {nullptr, 0, wp, false};
		}

		const uint32_t idx = static_cast<uint32_t>(rp % slotCount_);
		const auto* slotPtr = slotBase_ + static_cast<std::size_t>(idx) * slotByteSize_;

		return {slotPtr, idx, wp, true};
	}

	/**
	 * @brief Advance the read position after consuming a slot.
	 *
	 * Must be called after the consumer has finished reading the slot.
	 */
	void advanceRead() noexcept
	{
		control_->readPos.fetch_add(1, std::memory_order_release);
	}

	// ── Latest Slot (for "latest frame" consumers) ─────────────────────────

	/**
	 * @brief Get the most recently written slot (for latest-frame consumers).
	 *
	 * Unlike acquireReadSlot() which reads sequentially, this returns the
	 * most recently committed slot.  Useful for video consumers that always
	 * want the freshest frame and don't need to process every frame.
	 *
	 * @return Pointer to the latest slot data and its index.
	 */
	struct LatestResult {
		const uint8_t* data;
		uint32_t       slot;
		uint64_t       sequence;
		bool           valid;      ///< false if no data has been written yet
	};

	[[nodiscard]] LatestResult readLatestSlot() const noexcept
	{
		const uint64_t wp = control_->writePos.load(std::memory_order_acquire);
		if (wp == 0) {
			return {nullptr, 0, 0, false};
		}

		const uint64_t latestPos = wp - 1;
		const uint32_t idx = static_cast<uint32_t>(latestPos % slotCount_);
		const auto* slotPtr = slotBase_ + static_cast<std::size_t>(idx) * slotByteSize_;

		// After computing the slot, re-read writePos to detect torn reads.
		// If writePos advanced by more than (slotCount - 1), the data may be corrupt.
		const uint64_t wpAfter = control_->writePos.load(std::memory_order_acquire);
		if (wpAfter != wp) {
			return {slotPtr, idx, wp, false};
		}

		return {slotPtr, idx, wp, true};
	}

	// ── Diagnostics ────────────────────────────────────────────────────────

	[[nodiscard]] bool writerHasLappedReader() const noexcept
	{
		const uint64_t rp = control_->readPos.load(std::memory_order_acquire);
		const uint64_t wp = control_->writePos.load(std::memory_order_acquire);
		return (wp - rp) >= slotCount_;
	}

	[[nodiscard]] uint64_t availableSlots() const noexcept
	{
		const uint64_t rp = control_->readPos.load(std::memory_order_acquire);
		const uint64_t wp = control_->writePos.load(std::memory_order_acquire);
		return (wp > rp) ? (wp - rp) : 0;
	}

private:
	ShmMapping              mapping_;
	ShmCircularControl*     control_   = nullptr;
	Header*                 header_    = nullptr;
	uint8_t*                slotBase_  = nullptr;
	uint32_t                slotCount_ = 0;
	std::size_t             slotByteSize_ = 0;
};

// ---------------------------------------------------------------------------
// ShmCircularReader — lightweight consumer-side reader
//
// Reads a ShmCircularBuffer segment via a raw ShmMapping reference.
// Does not own the mapping.  Useful for consumers that don't know the
// template parameters at compile time or that receive the mapping from
// elsewhere (e.g. readLatestBgrFrame).
// ---------------------------------------------------------------------------

class ShmCircularReader {
public:
	/**
	 * @brief Construct a reader over an existing ShmMapping.
	 *
	 * @param mapping    Reference to a mapped SHM segment.
	 * @param headerSize Raw sizeof(Header) — will be padded to 64B internally.
	 */
	explicit ShmCircularReader(const ShmMapping& mapping, std::size_t headerSize)
		: base_(mapping.bytes())
		, headerPaddedSize_(alignToCacheLine(headerSize))
	{
		control_ = reinterpret_cast<const ShmCircularControl*>(
			base_ + headerPaddedSize_);
	}

	/**
	 * @brief Slot index of the most recently written slot (latest frame).
	 *
	 * Returns the slot that the producer most recently committed.
	 * Returns std::nullopt if nothing has been written yet.
	 */
	[[nodiscard]] std::optional<uint32_t> latestSlotIndex() const noexcept
	{
		const uint64_t wp = control_->writePos.load(std::memory_order_acquire);
		if (wp == 0) return std::nullopt;
		return static_cast<uint32_t>((wp - 1) % control_->slotCount);
	}

	/**
	 * @brief Pointer to a specific slot's data region.
	 */
	[[nodiscard]] const uint8_t* slotData(uint32_t slotIndex) const noexcept
	{
		const std::size_t slotsOffset = headerPaddedSize_ + sizeof(ShmCircularControl);
		return base_ + slotsOffset
			+ static_cast<std::size_t>(slotIndex) * control_->slotByteSize;
	}

	/**
	 * @brief writePos value for stale-read detection.
	 */
	[[nodiscard]] uint64_t writePos() const noexcept
	{
		return control_->writePos.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint32_t slotCount() const noexcept
	{
		return control_->slotCount;
	}

	[[nodiscard]] std::size_t slotByteSize() const noexcept
	{
		return control_->slotByteSize;
	}

	/**
	 * @brief Check if a read captured at `capturedWritePos` is stale.
	 *
	 * Call writePos() before reading, copy data, then call this
	 * with the previously captured value.
	 */
	[[nodiscard]] bool isStale(uint64_t capturedWritePos) const noexcept
	{
		return control_->writePos.load(std::memory_order_acquire) != capturedWritePos;
	}

private:
	const uint8_t*            base_;
	const ShmCircularControl* control_;
	std::size_t               headerPaddedSize_;
};

