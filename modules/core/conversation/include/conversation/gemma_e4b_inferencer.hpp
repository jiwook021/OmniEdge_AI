#pragma once

// EXPERIMENTAL -- not validated with real models

#include "conversation_inferencer.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>


/**
 * @brief Gemma 4 E4B inferencer for multimodal conversation (text-only output).
 *
 * Gemma 4 E4B accepts audio + vision input natively but produces text-only
 * output.  supportsNativeAudio() returns false -- a KokoroTTSNode sidecar
 * is required for speech synthesis.
 *
 * VRAM footprint: ~3.0 GB (expert-based MoE, ~4B parameters)
 *
 * Thread safety:
 *   - loadModel(), generate(), unloadModel() -- single thread only.
 *   - cancel() -- may be called from any thread (uses std::atomic).
 *
 * Pimpl pattern: model-specific headers confined to .cpp.
 */
class GemmaE4BInferencer final : public ConversationInferencer {
public:
    GemmaE4BInferencer();
    ~GemmaE4BInferencer() override;

    // Non-copyable, non-movable -- owns GPU resources.
    GemmaE4BInferencer(const GemmaE4BInferencer&)            = delete;
    GemmaE4BInferencer& operator=(const GemmaE4BInferencer&) = delete;

    /// Gemma 4 E4B does NOT produce native audio -- TTS sidecar required.
    [[nodiscard]] bool supportsNativeAudio() const noexcept override { return false; }

    [[nodiscard]] tl::expected<void, std::string> loadModel(
        const std::string& modelDir) override;

    [[nodiscard]] tl::expected<void, std::string> generate(
        const std::string& prompt,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) override;

    void cancel() noexcept override;
    void unloadModel() noexcept override;

    [[nodiscard]] size_t currentVramUsageBytes() const noexcept override;
    [[nodiscard]] std::string name() const override { return "gemma-e4b"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    std::atomic<bool> cancelRequested_{false};
};

/// Factory: create a Gemma 4 E4B inferencer.
[[nodiscard]] std::unique_ptr<ConversationInferencer> createGemmaE4BInferencer();

