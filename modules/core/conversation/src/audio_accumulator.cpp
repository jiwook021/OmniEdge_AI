#include "conversation/audio_accumulator.hpp"

#include <cstring>

#include <nlohmann/json.hpp>

#include "common/oe_logger.hpp"
#include "common/oe_shm_helpers.hpp"
#include "shm/shm_mapping.hpp"
#include "zmq/audio_constants.hpp"
#include "zmq/jpeg_constants.hpp"  // kAudioShmDataOffset


// Total SHM segment size (same formula as whisper_constants.hpp, inlined
// here to avoid a conversation → STT module dependency).
static constexpr std::size_t kAudioShmSegmentByteSize =
    kAudioShmDataOffset + kRingBufferSlotCount * kAudioShmSlotByteSize;


AudioAccumulator::AudioAccumulator(const Config& config)
    : config_(config)
{
    buffer_.reserve(config_.maxAccumulationSamples);
}

AudioAccumulator::~AudioAccumulator() = default;

tl::expected<void, std::string> AudioAccumulator::initialize()
{
    auto result = oe::shm::openConsumerWithRetry(
        config_.shmName, kAudioShmSegmentByteSize, "conversation_stt");
    if (!result) return tl::unexpected(result.error());
    shm_ = std::move(*result);
    return {};
}

bool AudioAccumulator::appendFromShm(const nlohmann::json& chunkMetadata)
{
    const int shmSlotIndex = chunkMetadata.value("slot",    -1);
    const int sampleCount  = chunkMetadata.value("samples",  0);

    if (shmSlotIndex < 0 || sampleCount <= 0) {
        OE_LOG_WARN("audio_acc_invalid_chunk: slot={}, samples={}", shmSlotIndex, sampleCount);
        return false;
    }
    if (static_cast<std::size_t>(shmSlotIndex) >= kRingBufferSlotCount) {
        OE_LOG_WARN("audio_acc_slot_oob: slot={}", shmSlotIndex);
        return false;
    }
    if (static_cast<uint32_t>(sampleCount) > kVadChunkSampleCount) {
        OE_LOG_WARN("audio_acc_chunk_too_large: samples={}, max={}", sampleCount, kVadChunkSampleCount);
        return false;
    }

    // Hard cap: prevent unbounded growth if caller doesn't check isFull()
    if (buffer_.size() >= config_.maxAccumulationSamples) {
        OE_LOG_WARN("audio_acc_buffer_full: size={}, max={}",
                    buffer_.size(), config_.maxAccumulationSamples);
        return false;
    }

    // Locate the slot in the SHM ring buffer
    const uint8_t* slotBase = shm_->bytes()
                               + kAudioShmDataOffset
                               + static_cast<std::size_t>(shmSlotIndex) * kAudioShmSlotByteSize;

    // Stale-read guard: compare sequence number before and after copy
    const auto* header = reinterpret_cast<const ShmAudioHeader*>(slotBase);
    const uint64_t seqBefore = header->seqNumber;

    const float* pcm = reinterpret_cast<const float*>(slotBase + sizeof(ShmAudioHeader));

    const std::size_t prevSize = buffer_.size();
    buffer_.resize(prevSize + static_cast<std::size_t>(sampleCount));
    std::memcpy(buffer_.data() + prevSize,
                pcm,
                static_cast<std::size_t>(sampleCount) * sizeof(float));

    const uint64_t seqAfter = header->seqNumber;
    if (seqAfter != seqBefore) {
        // Producer overwrote this slot during our read — discard
        buffer_.resize(prevSize);
        OE_LOG_WARN("audio_acc_stale_shm: slot={}, seq_before={}, seq_after={}",
                   shmSlotIndex, seqBefore, seqAfter);
        return false;
    }

    return true;
}

std::size_t AudioAccumulator::flush(std::vector<float>& out)
{
    const std::size_t n = buffer_.size();
    out = std::move(buffer_);
    buffer_.clear();
    buffer_.reserve(config_.maxAccumulationSamples);
    return n;
}

bool AudioAccumulator::isFull() const noexcept
{
    return buffer_.size() >= config_.maxAccumulationSamples;
}

bool AudioAccumulator::empty() const noexcept
{
    return buffer_.empty();
}

void AudioAccumulator::clear() noexcept
{
    buffer_.clear();
}
