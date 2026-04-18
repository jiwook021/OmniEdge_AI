// gemma_inferencer_stub.cpp -- CPU-only stub for tests.
//
// PURPOSE: Provides all GemmaInferencer method bodies WITHOUT real model
// headers. Tests use MockConversationInferencer exclusively. This stub lets
// conversation_node.cpp link without requiring model weights, the HF
// transformers Python runtime, or a GPU.
//
// Every method returns an explicit "not available" error so accidental calls
// surface immediately rather than silently returning garbage.

#include "conversation/gemma_inferencer.hpp"

#include <memory>
#include <span>
#include <string>


struct GemmaInferencer::Impl {};

GemmaInferencer::GemmaInferencer(Variant variant)
    : variant_(variant)
    , pimpl_(std::make_unique<Impl>())
{}

GemmaInferencer::~GemmaInferencer() = default;

tl::expected<void, std::string>
GemmaInferencer::loadModel(const std::string& /*modelDir*/)
{
    return tl::unexpected(
        std::string("Gemma not available: built with stub inferencer"));
}

tl::expected<void, std::string>
GemmaInferencer::generate(
    const std::string& /*prompt*/,
    const GenerationParams& /*params*/,
    const std::function<void(std::string_view, bool, bool)>& /*callback*/)
{
    return tl::unexpected(
        std::string("Gemma not available: built with stub inferencer"));
}

tl::expected<void, std::string>
GemmaInferencer::generateWithVideo(
    const std::string& /*prompt*/,
    std::span<const uint8_t> /*videoFrame*/,
    uint32_t /*frameWidth*/,
    uint32_t /*frameHeight*/,
    const GenerationParams& /*params*/,
    const std::function<void(std::string_view, bool, bool)>& /*callback*/)
{
    return tl::unexpected(
        std::string("Gemma not available: built with stub inferencer"));
}

tl::expected<std::string, std::string>
GemmaInferencer::transcribe(
    std::span<const float> /*pcmAudio*/,
    uint32_t /*sampleRateHz*/)
{
    return tl::unexpected(
        std::string("Gemma not available: built with stub inferencer"));
}

void GemmaInferencer::cancel() noexcept
{
    cancelRequested_.store(true, std::memory_order_release);
}

void GemmaInferencer::unloadModel() noexcept {}

size_t GemmaInferencer::currentVramUsageBytes() const noexcept
{
    return 0;
}

std::string GemmaInferencer::name() const
{
    return std::string(variantTag(variant_));
}

std::string_view GemmaInferencer::variantTag(Variant variant) noexcept
{
    switch (variant) {
    case Variant::kE2B: return "gemma-e2b";
    case Variant::kE4B: return "gemma-e4b";
    }
    return "gemma-unknown";
}

std::unique_ptr<ConversationInferencer>
createGemmaInferencer(GemmaInferencer::Variant variant)
{
    return std::make_unique<GemmaInferencer>(variant);
}
