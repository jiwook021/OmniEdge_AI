// qwen_omni_inferencer_stub.cpp -- CPU-only stub for tests.
//
// PURPOSE: Provides all QwenOmniInferencer method bodies WITHOUT real model
// headers.  Tests use MockConversationInferencer exclusively.  This stub lets
// conversation_node.cpp link without requiring model weights or GPU runtime.
//
// Every method returns an explicit "not available" error so accidental calls
// surface immediately rather than silently returning garbage.

// EXPERIMENTAL -- not validated with real models

#include "conversation/qwen_omni_inferencer.hpp"

#include <memory>
#include <string>


struct QwenOmniInferencer::Impl {};

QwenOmniInferencer::QwenOmniInferencer(std::string variantName)
    : variantName_(std::move(variantName))
    , pimpl_(std::make_unique<Impl>())
{}

QwenOmniInferencer::~QwenOmniInferencer() = default;

tl::expected<void, std::string>
QwenOmniInferencer::loadModel(const std::string& /*modelDir*/)
{
    return tl::unexpected(
        std::string("Qwen-Omni not available: built with stub inferencer"));
}

tl::expected<void, std::string>
QwenOmniInferencer::generate(
    const std::string& /*prompt*/,
    const GenerationParams& /*params*/,
    const std::function<void(std::string_view, bool, bool)>& /*callback*/)
{
    return tl::unexpected(
        std::string("Qwen-Omni not available: built with stub inferencer"));
}

void QwenOmniInferencer::cancel() noexcept
{
    cancelRequested_.store(true, std::memory_order_release);
}

void QwenOmniInferencer::unloadModel() noexcept {}

size_t QwenOmniInferencer::currentVramUsageBytes() const noexcept
{
    return 0;
}

std::unique_ptr<ConversationInferencer> createQwenOmni7bInferencer()
{
    return std::make_unique<QwenOmniInferencer>("qwen-omni-7b");
}

std::unique_ptr<ConversationInferencer> createQwenOmni3bInferencer()
{
    return std::make_unique<QwenOmniInferencer>("qwen-omni-3b");
}

