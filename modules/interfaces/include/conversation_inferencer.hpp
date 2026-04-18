#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

#include <tl/expected.hpp>

#include "common/runtime_defaults.hpp"


/**
 * @brief User-controllable generation parameters forwarded from the frontend.
 *
 * Parsed from the `generation_params` field of the `conversation_prompt`
 * ZMQ message.  If fields are missing, defaults from omniedge.ini [generation]
 * section are used (falling back to compile-time constants).  Values are
 * clamped to valid ranges before use.
 */
struct GenerationParams {
    float temperature = kGenerationDefaultTemperature;  ///< 0.0 = deterministic, 2.0 = maximum creativity
    float topP        = kGenerationDefaultTopP;         ///< Nucleus sampling threshold (0.0 - 1.0)
    int   maxTokens   = kGenerationDefaultMaxTokens;    ///< Maximum tokens to generate (64 - 8192)
    std::string imagePath;                               ///< Optional: path to uploaded image for vision-capable models

    /// Clamp all fields to valid ranges.
    constexpr void clamp() noexcept
    {
        if (temperature < 0.0f) temperature = 0.0f;
        if (temperature > 2.0f) temperature = 2.0f;
        if (topP < 0.0f) topP = 0.0f;
        if (topP > 1.0f) topP = 1.0f;
        if (maxTokens < kGenerationMinTokens) maxTokens = kGenerationMinTokens;
        if (maxTokens > kGenerationMaxTokens) maxTokens = kGenerationMaxTokens;
    }
};

/**
 * @brief Per-token streaming callback for conversation inference.
 *
 * Called once per generated token during streaming generation.
 *
 * @param token             Decoded text for this token (may be empty for BOS/EOS).
 * @param sentenceBoundary  True when this token starts a new sentence
 *                          (previous token ended with '.', '?', or '!').
 * @param done              True on the final call -- generation is complete.
 */
/**
 * @brief Pure virtual interface for multimodal conversation inferencers.
 *
 * Strategy pattern: ConversationNode depends on this interface only.
 * Production code supplies a concrete inferencer (currently GemmaInferencer
 * for Gemma-4 E2B/E4B); tests inject MockConversationInferencer.
 *
 * Capability flags advertise what the current inferencer can do natively:
 *   - supportsNativeStt()    — accepts raw PCM audio, no Whisper needed
 *   - supportsNativeTts()    — emits audio tokens, no TTS sidecar needed
 *   - supportsNativeVision() — accepts image/video frames alongside text
 *
 * Thread safety: all methods are called from the ConversationNode run()
 * thread except cancel(), which may be called from any thread.
 */
class ConversationInferencer {
public:
    virtual ~ConversationInferencer() = default;

    /**
     * @brief Whether this inferencer can transcribe audio input natively.
     *
     * True for Gemma-4 (accepts raw PCM via native audio encoder).
     */
    [[nodiscard]] virtual bool supportsNativeStt() const noexcept = 0;

    /**
     * @brief Whether this inferencer produces native audio output.
     *
     * False for Gemma-4 (text-only output — requires KokoroTTSNode sidecar).
     */
    [[nodiscard]] virtual bool supportsNativeTts() const noexcept = 0;

    /**
     * @brief Whether this inferencer accepts video/image frames as input.
     *
     * True for Gemma-4 (accepts image tokens alongside text/audio).
     * When true, generateWithVideo() can be called with a video frame
     * to enable video conversation mode.
     */
    [[nodiscard]] virtual bool supportsNativeVision() const noexcept = 0;

    /**
     * @brief Load model weights and tokenizer from disk.
     *
     * @param modelDir  Path to the model weights directory.
     * @return void on success, or an error message on failure.
     */
    [[nodiscard]] virtual tl::expected<void, std::string> loadModel(
        const std::string& modelDir) = 0;

    /**
     * @brief Stream-generate tokens from a prompt with generation parameters.
     *
     * Calls @p callback once per generated token.  The final call has
     * done=true.  Must not be called concurrently.
     *
     * cancel() may be called from another thread while generate() is running.
     *
     * @param prompt    Fully formatted prompt (assembled by daemon).
     * @param params    Generation parameters (temperature, topP, maxTokens).
     * @param callback  Invoked per token on the calling thread.
     * @return void on success, error string on engine/CUDA failure.
     */
    [[nodiscard]] virtual tl::expected<void, std::string> generate(
        const std::string& prompt,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) = 0;

    /**
     * @brief Stream-generate tokens from a prompt with an attached video frame.
     *
     * Only valid when supportsNativeVision() returns true.  The video frame
     * is raw BGR24 data read from the video ingest SHM circular buffer.
     * The inferencer is responsible for any format conversion (BGR→RGB,
     * resize, JPEG encode) required by the underlying model.
     *
     * Default implementation ignores the frame and delegates to generate().
     *
     * @param prompt     Fully formatted prompt (assembled by daemon).
     * @param videoFrame Raw BGR24 frame data from /oe.vid.ingest SHM.
     * @param frameWidth  Frame width in pixels.
     * @param frameHeight Frame height in pixels.
     * @param params     Generation parameters (temperature, topP, maxTokens).
     * @param callback   Invoked per token on the calling thread.
     * @return void on success, error string on engine/CUDA failure.
     */
    [[nodiscard]] virtual tl::expected<void, std::string> generateWithVideo(
        const std::string& prompt,
        std::span<const uint8_t> /*videoFrame*/,
        uint32_t /*frameWidth*/,
        uint32_t /*frameHeight*/,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback)
    {
        // Default: ignore video frame, fall back to text-only generation.
        return generate(prompt, params, callback);
    }

    /**
     * @brief Transcribe raw PCM audio to text using the model's native STT.
     *
     * Only valid when supportsNativeStt() returns true.  Accepts mono float32
     * PCM at the specified sample rate.
     *
     * @param pcmAudio     Contiguous mono float32 samples.
     * @param sampleRateHz Sample rate (typically 16000).
     * @return Transcribed text on success, or an error message on failure.
     */
    [[nodiscard]] virtual tl::expected<std::string, std::string> transcribe(
        std::span<const float> pcmAudio,
        uint32_t sampleRateHz) = 0;

    /**
     * @brief Set a callback invoked periodically during generate() so the
     * caller can publish heartbeats while the poll loop is blocked.
     *
     * Default: no-op.  Override in inferencers that block for extended periods.
     */
    virtual void setHeartbeatCallback(std::function<void()> /*cb*/) {}

    /**
     * @brief Cancel an in-progress generate() call.
     *
     * Thread-safe.  Returns immediately; generation ceases at the next token
     * boundary.  A no-op if no generation is running.
     */
    virtual void cancel() noexcept = 0;

    /**
     * @brief Unload model weights and release all GPU memory.
     *
     * Idempotent -- safe to call multiple times or when loadModel() never ran.
     */
    virtual void unloadModel() noexcept = 0;

    /**
     * @brief Estimated VRAM used by this inferencer in bytes.
     *
     * Returns 0 before loadModel() succeeds.
     */
    [[nodiscard]] virtual size_t currentVramUsageBytes() const noexcept = 0;

    /** @brief Inferencer identifier string, e.g. "gemma-e4b". */
    [[nodiscard]] virtual std::string name() const = 0;
};

