#include "shm/shm_mapping.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <cerrno>
#include <iostream>
#include <cstring>
#include <format>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


ShmMapping::ShmMapping(std::string_view name, std::size_t size, bool create)
    : mappedSize_(size), segmentName_(name), isCreator_(create)
{
    if (size == 0) throw std::invalid_argument("[ShmMapping] size must be > 0");
    OE_LOG_DEBUG("shm_open_start: name={}, size={}, create={}", segmentName_, mappedSize_, create);

    const int flags = create
        ? (O_CREAT | O_RDWR | O_TRUNC)
        : O_RDWR;

    const int fileDescriptor = ::shm_open(segmentName_.c_str(), flags, 0600);
    if (fileDescriptor < 0) {
        throw std::runtime_error(
            std::format("[ShmMapping] shm_open('{}', create={}) failed: {}",
                        segmentName_, create, std::strerror(errno)));
    }

    if (create) {
        if (::ftruncate(fileDescriptor, static_cast<off_t>(mappedSize_)) < 0) {
            const int savedErrno = errno;
            ::close(fileDescriptor);
            throw std::runtime_error(
                std::format("[ShmMapping] ftruncate('{}', {} bytes) failed: {}",
                            segmentName_, mappedSize_, std::strerror(savedErrno)));
        }
    } else {
        // Consumer mode: verify the segment is at least as large as requested
        // to prevent SIGBUS from mapping beyond the actual segment size.
        struct stat segmentStat{};
        if (::fstat(fileDescriptor, &segmentStat) < 0) {
            const int savedErrno = errno;
            ::close(fileDescriptor);
            throw std::runtime_error(
                std::format("[ShmMapping] fstat('{}') failed: {}",
                            segmentName_, std::strerror(savedErrno)));
        }
        const auto actualSize = static_cast<std::size_t>(segmentStat.st_size);
        if (actualSize < mappedSize_) {
            ::close(fileDescriptor);
            throw std::runtime_error(
                std::format("[ShmMapping] segment '{}' too small: actual={} < requested={}",
                            segmentName_, actualSize, mappedSize_));
        }
    }

    void* mapped = ::mmap(nullptr, mappedSize_, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fileDescriptor, 0);
    if (mapped == MAP_FAILED) {
        const int savedErrno = errno;
        ::close(fileDescriptor);
        if (create) { ::shm_unlink(segmentName_.c_str()); }
        throw std::runtime_error(
            std::format("[ShmMapping] mmap('{}', {} bytes) failed: {}",
                        segmentName_, mappedSize_, std::strerror(savedErrno)));
    }

    ::close(fileDescriptor);

    // ── Best-effort memory optimizations ───────────────────────────────
    // Both are non-fatal: the system works without THP or mlock.

#ifdef MADV_HUGEPAGE
    if (mappedSize_ >= kHugePageThreshold) {
        if (::madvise(mapped, mappedSize_, MADV_HUGEPAGE) != 0) {
            OE_LOG_WARN("madvise(MADV_HUGEPAGE) failed for '{}': {} (non-fatal)",
                        segmentName_, std::strerror(errno));
        }
    }
#endif

    if (::mlock(mapped, mappedSize_) != 0) {
        OE_LOG_WARN("mlock failed for '{}' ({} bytes): {} "
                    "(non-fatal, raise RLIMIT_MEMLOCK for pinned SHM)",
                    segmentName_, mappedSize_, std::strerror(errno));
    }

    mappedRegion_ = mapped;

    OE_LOG_DEBUG("shm_mapped: name={}, size={}, is_creator={}", segmentName_, mappedSize_, isCreator_);
}

ShmMapping::~ShmMapping()
{
    reset();
}

ShmMapping::ShmMapping(ShmMapping&& other) noexcept
    : mappedRegion_(other.mappedRegion_), mappedSize_(other.mappedSize_),
      segmentName_(std::move(other.segmentName_)), isCreator_(other.isCreator_)
{
    try { OE_LOG_DEBUG("shm_move_constructed: name={}, size={}", segmentName_, mappedSize_); }
    catch (...) { std::cerr << "shm_mapping_log_failed (move-ctor): " << segmentName_ << '\n'; }
    other.mappedRegion_ = nullptr;
    other.mappedSize_   = 0;
    other.isCreator_    = false;
}

ShmMapping& ShmMapping::operator=(ShmMapping&& other) noexcept
{
    if (this != &other) {
        reset();
        mappedRegion_ = other.mappedRegion_;
        mappedSize_   = other.mappedSize_;
        segmentName_  = std::move(other.segmentName_);
        isCreator_    = other.isCreator_;
        other.mappedRegion_ = nullptr;
        other.mappedSize_   = 0;
        other.isCreator_    = false;
    }
    return *this;
}

void ShmMapping::reset() noexcept
{
    if (mappedRegion_ != nullptr) {
        try { OE_LOG_DEBUG("shm_unmapping: name={}, size={}", segmentName_, mappedSize_); }
        catch (...) { std::cerr << "shm_mapping_log_failed (unmapping): " << segmentName_ << '\n'; }
        ::munmap(mappedRegion_, mappedSize_);
        mappedRegion_ = nullptr;
        mappedSize_   = 0;
    }
    if (isCreator_ && !segmentName_.empty()) {
        try { OE_LOG_DEBUG("shm_unlinking: name={}", segmentName_); }
        catch (...) { std::cerr << "shm_mapping_log_failed (unlinking): " << segmentName_ << '\n'; }
        ::shm_unlink(segmentName_.c_str());
        isCreator_ = false;
    }
}

