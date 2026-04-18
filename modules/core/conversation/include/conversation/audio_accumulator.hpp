#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <tl/expected.hpp>


class ShmMapping;

/**
 * @brief Accumulates speech-gated PCM audio from shared memory.
 *
 * Reads audio_chunk metadata (slot index, sample count) from ZMQ messages,
 * reads the corresponding PCM data from the SHM ring buffer /oe.aud.ingest,
 * and appends it to an internal rolling buffer.
 *
 * Uses a stale-read guard (seqNumber before/after memcpy) to discard data
 * when the producer overwrites the slot during the read.
 *
 * Thread safety: all methods must be called from the same thread
 * (the ConversationNode poll loop).
 */
class AudioAccumulator {
public:
    struct Config {
        std::string shmName{"/oe.aud.ingest"};
        uint32_t    maxAccumulationSamples{480'000};  ///< 30 s at 16 kHz
    };

    explicit AudioAccumulator(const Config& config);
    ~AudioAccumulator();

    AudioAccumulator(const AudioAccumulator&)            = delete;
    AudioAccumulator& operator=(const AudioAccumulator&) = delete;

    /// Open the SHM segment as a consumer (retries up to 10 times).
    [[nodiscard]] tl::expected<void, std::string> initialize();

    /// Process an audio_chunk ZMQ message: read PCM from SHM slot, append to buffer.
    /// @return true if data was successfully appended.
    bool appendFromShm(const nlohmann::json& chunkMetadata);

    /// Move accumulated audio into @p out, clearing the internal buffer.
    /// @return Number of samples flushed.
    std::size_t flush(std::vector<float>& out);

    /// True when buffer has reached maxAccumulationSamples.
    [[nodiscard]] bool isFull() const noexcept;

    /// True when no audio has been accumulated.
    [[nodiscard]] bool empty() const noexcept;

    /// Discard all accumulated audio.
    void clear() noexcept;

private:
    Config                        config_;
    std::unique_ptr<ShmMapping>   shm_;
    std::vector<float>            buffer_;
};
