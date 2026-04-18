#pragma once

// EXPERIMENTAL -- not validated on hardware yet (Gemma-4 is gated, requires HF_TOKEN)

#include "conversation_inferencer.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>


/**
 * @brief Google Gemma-4 conversation inferencer (E2B and E4B variants).
 *
 * Loads a HuggingFace `google/gemma-4-*-it` checkpoint via a persistent Python
 * subprocess (model_generate.py). Both variants accept raw PCM audio and image
 * frames as native input but emit text-only output; TTS is handled by the
 * Kokoro sidecar.
 *
 * VRAM footprint (BF16 on RTX PRO 3000 Blackwell — measure before commit):
 *   - Gemma-4 E2B (2B effective): ~2.5 GB weights + activations
 *   - Gemma-4 E4B (4B effective): ~3.0 GB active (15 GB full, sparse MoE)
 *
 * Thread safety:
 *   - loadModel(), generate(), unloadModel() — single thread only
 *     (the ConversationNode run() thread).
 *   - cancel() — may be called from any thread (uses std::atomic).
 *
 * Pimpl pattern: Python-subprocess plumbing is confined to the .cpp.
 */
class GemmaInferencer final : public ConversationInferencer {
public:
    /// Gemma-4 variant selector. Extend by adding enum entries and a
    /// variantTag() case in the .cpp — no other wiring required.
    enum class Variant : std::uint8_t {
        kE2B,   ///< google/gemma-4-E2B-it (2B effective)
        kE4B,   ///< google/gemma-4-E4B-it (4B effective, default)
    };

    explicit GemmaInferencer(Variant variant);
    ~GemmaInferencer() override;

    // Non-copyable, non-movable — owns a Python subprocess + GPU resources.
    GemmaInferencer(const GemmaInferencer&)            = delete;
    GemmaInferencer& operator=(const GemmaInferencer&) = delete;

    /// Gemma-4 accepts raw audio input natively (bypasses Whisper STT).
    [[nodiscard]] bool supportsNativeStt()    const noexcept override { return true;  }
    /// Gemma-4 emits text only — KokoroTTSNode sidecar required.
    [[nodiscard]] bool supportsNativeTts()    const noexcept override { return false; }
    /// Gemma-4 accepts image/video frames as native input.
    [[nodiscard]] bool supportsNativeVision() const noexcept override { return true;  }

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
    [[nodiscard]] std::string name() const override;

    void setHeartbeatCallback(std::function<void()> cb) override { heartbeatCb_ = std::move(cb); }

    /// Short tag for the variant, e.g. "gemma-e4b". Useful for diagnostic
    /// logging and as the ZMQ `model` field on `llm_response` messages.
    [[nodiscard]] static std::string_view variantTag(Variant variant) noexcept;

private:
    Variant                 variant_;
    struct Impl;
    std::unique_ptr<Impl>   pimpl_;
    std::atomic<bool>       cancelRequested_{false};
    std::function<void()>   heartbeatCb_;  ///< Pumps heartbeats during blocking generate()
};

/// Factory. The daemon resolves the default variant from INI
/// `[conversation_model].default_variant` before calling this.
[[nodiscard]] std::unique_ptr<ConversationInferencer>
createGemmaInferencer(GemmaInferencer::Variant variant);
