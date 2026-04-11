#pragma once

// EXPERIMENTAL -- not validated with real models

#include "conversation_inferencer.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>


/**
 * @brief Qwen2.5-Omni inferencer for unified STT+LLM+TTS conversation.
 *
 * Supports both the 7B and 3B variants.  Both accept audio input
 * (supportsNativeStt() == true) and produce native audio output
 * (supportsNativeTts() == true) -- no TTS sidecar required.
 *
 * The variant is selected by the model directory passed to loadModel().
 *
 * VRAM footprint (standard tier):
 *   - Qwen2.5-Omni-7B: ~3.5 GB (4-bit GPTQ/AWQ)
 *   - Qwen2.5-Omni-3B: ~2.0 GB
 *
 * Thread safety:
 *   - loadModel(), generate(), unloadModel() -- single thread only
 *     (the ConversationNode run() thread).
 *   - cancel() -- may be called from any thread (uses std::atomic).
 *
 * Pimpl pattern: model-specific headers confined to .cpp.
 */
class QwenOmniInferencer final : public ConversationInferencer {
public:
    /// @param variantName  "qwen-omni-7b" or "qwen-omni-3b"
    explicit QwenOmniInferencer(std::string variantName);
    ~QwenOmniInferencer() override;

    // Non-copyable, non-movable -- owns GPU resources.
    QwenOmniInferencer(const QwenOmniInferencer&)            = delete;
    QwenOmniInferencer& operator=(const QwenOmniInferencer&) = delete;

    /// Qwen2.5-Omni accepts raw audio input natively.
    [[nodiscard]] bool supportsNativeStt() const noexcept override { return true; }
    /// Qwen2.5-Omni produces native audio output.
    [[nodiscard]] bool supportsNativeTts() const noexcept override { return true; }
    /// Qwen2.5-Omni accepts image/video frames as native input.
    [[nodiscard]] bool supportsNativeVision() const noexcept override { return true; }

    [[nodiscard]] tl::expected<void, std::string> loadModel(
        const std::string& modelDir) override;

    [[nodiscard]] tl::expected<void, std::string> generate(
        const std::string& prompt,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) override;

    [[nodiscard]] tl::expected<void, std::string> generateWithVideo(
        const std::string& prompt,
        std::span<const uint8_t> videoFrame,
        uint32_t frameWidth,
        uint32_t frameHeight,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) override;

    [[nodiscard]] tl::expected<std::string, std::string> transcribe(
        std::span<const float> pcmAudio,
        uint32_t sampleRateHz) override;

    void cancel() noexcept override;
    void unloadModel() noexcept override;

    [[nodiscard]] size_t currentVramUsageBytes() const noexcept override;
    [[nodiscard]] std::string name() const override { return variantName_; }

private:
    std::string variantName_;
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    std::atomic<bool> cancelRequested_{false};
};

/// Factory: create a Qwen2.5-Omni-7B inferencer.
[[nodiscard]] std::unique_ptr<ConversationInferencer> createQwenOmni7bInferencer();

/// Factory: create a Qwen2.5-Omni-3B inferencer.
[[nodiscard]] std::unique_ptr<ConversationInferencer> createQwenOmni3bInferencer();

