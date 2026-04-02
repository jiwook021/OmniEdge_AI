#pragma once

#include <cstddef>
#include <functional>
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

    /// Clamp all fields to valid ranges.
    void clamp() noexcept
    {
        if (temperature < 0.0f) temperature = 0.0f;
        if (temperature > 2.0f) temperature = 2.0f;
        if (topP < 0.0f) topP = 0.0f;
        if (topP > 1.0f) topP = 1.0f;
        if (maxTokens < 64)   maxTokens = 64;
        if (maxTokens > 8192) maxTokens = 8192;
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
 * Production code supplies model-specific implementations (QwenOmniInferencer,
 * GemmaE4BInferencer); tests inject MockConversationInferencer.
 *
 * Supports three models:
 *   - Qwen2.5-Omni-7B  (native audio I/O, no TTS sidecar)
 *   - Qwen2.5-Omni-3B  (native audio I/O, no TTS sidecar)
 *   - Gemma 4 E4B       (text-only output, requires TTS sidecar)
 *
 * Thread safety: all methods are called from the ConversationNode run()
 * thread except cancel(), which may be called from any thread.
 */
class ConversationInferencer {
public:
    virtual ~ConversationInferencer() = default;

    /**
     * @brief Whether this inferencer produces native audio output.
     *
     * True for Qwen2.5-Omni (unified STT+LLM+TTS).
     * False for Gemma 4 E4B (text only -- needs KokoroTTSNode sidecar).
     */
    [[nodiscard]] virtual bool supportsNativeAudio() const noexcept = 0;

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

    /** @brief Inferencer identifier string, e.g. "qwen-omni-7b". */
    [[nodiscard]] virtual std::string name() const = 0;
};

