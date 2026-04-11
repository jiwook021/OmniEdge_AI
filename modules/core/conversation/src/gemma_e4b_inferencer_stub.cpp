// gemma_e4b_inferencer_stub.cpp -- CPU-only stub for tests.
//
// PURPOSE: Provides all GemmaE4BInferencer method bodies WITHOUT real model
// headers.  Tests use MockConversationInferencer exclusively.  This stub lets
// conversation_node.cpp link without requiring model weights or GPU runtime.
//
// Every method returns an explicit "not available" error so accidental calls
// surface immediately rather than silently returning garbage.

// EXPERIMENTAL -- not validated with real models

#include "conversation/gemma_e4b_inferencer.hpp"

#include <memory>
#include <span>
#include <string>


struct GemmaE4BInferencer::Impl {};

GemmaE4BInferencer::GemmaE4BInferencer()
    : pimpl_(std::make_unique<Impl>())
{}

GemmaE4BInferencer::~GemmaE4BInferencer() = default;

tl::expected<void, std::string>
GemmaE4BInferencer::loadModel(const std::string& /*modelDir*/)
{
    return tl::unexpected(
        std::string("Gemma-E4B not available: built with stub inferencer"));
}

tl::expected<void, std::string>
GemmaE4BInferencer::generate(
    const std::string& /*prompt*/,
    const GenerationParams& /*params*/,
    const std::function<void(std::string_view, bool, bool)>& /*callback*/)
{
    return tl::unexpected(
        std::string("Gemma-E4B not available: built with stub inferencer"));
}

tl::expected<void, std::string>
GemmaE4BInferencer::generateWithVideo(
    const std::string& /*prompt*/,
    std::span<const uint8_t> /*videoFrame*/,
    uint32_t /*frameWidth*/,
    uint32_t /*frameHeight*/,
    const GenerationParams& /*params*/,
    const std::function<void(std::string_view, bool, bool)>& /*callback*/)
{
    return tl::unexpected(
        std::string("Gemma-E4B not available: built with stub inferencer"));
}

tl::expected<std::string, std::string>
GemmaE4BInferencer::transcribe(
    std::span<const float> /*pcmAudio*/,
    uint32_t /*sampleRateHz*/)
{
    return tl::unexpected(
        std::string("Gemma-E4B not available: built with stub inferencer"));
}

void GemmaE4BInferencer::cancel() noexcept
{
    cancelRequested_.store(true, std::memory_order_release);
}

void GemmaE4BInferencer::unloadModel() noexcept {}

size_t GemmaE4BInferencer::currentVramUsageBytes() const noexcept
{
    return 0;
}

std::unique_ptr<ConversationInferencer> createGemmaE4BInferencer()
{
    return std::make_unique<GemmaE4BInferencer>();
}

