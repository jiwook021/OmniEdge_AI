#pragma once

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "conversation_inferencer.hpp"


/// Deterministic mock conversation inferencer for node-level tests (no GPU required).
/// Configurable via public fields: tokensToEmit, simulateError, cancelCalled, nativeStt, nativeTts, nativeVision.
class MockConversationInferencer : public ConversationInferencer {
public:
    std::vector<std::string> tokensToEmit{"Hello", ",", " world", "!"};
    bool                     simulateError{false};
    std::atomic<bool>        cancelCalled{false};
    std::string              lastPrompt;       ///< Captured by generate() for inspection
    GenerationParams         lastParams;       ///< Captured by generate() for inspection
    bool                     nativeStt{true};  ///< Controls supportsNativeStt() return value (Gemma-4 default)
    bool                     nativeTts{false}; ///< Controls supportsNativeTts() return value (Gemma-4 default)
    bool                     nativeVision{true}; ///< Controls supportsNativeVision() return value (Gemma-4 default)
    std::string              mockTranscription{"Hello from mock STT"};

    // Video conversation tracking
    bool                     lastGenerateHadVideo{false}; ///< True if last generate used video path
    std::size_t              lastVideoFrameSize{0};       ///< Size of last video frame received
    uint32_t                 lastFrameWidth{0};
    uint32_t                 lastFrameHeight{0};

    [[nodiscard]] bool supportsNativeStt() const noexcept override
    {
        return nativeStt;
    }

    [[nodiscard]] bool supportsNativeTts() const noexcept override
    {
        return nativeTts;
    }

    [[nodiscard]] bool supportsNativeVision() const noexcept override
    {
        return nativeVision;
    }

    tl::expected<std::string, std::string> transcribe(
        std::span<const float> /*pcmAudio*/,
        uint32_t /*sampleRateHz*/) override
    {
        if (simulateError) {
            return tl::unexpected(std::string("mock transcription error"));
        }
        return mockTranscription;
    }

    tl::expected<void, std::string> loadModel(
        const std::string& /*modelDir*/) override
    {
        modelLoaded_ = true;
        return {};
    }

    tl::expected<void, std::string> generate(
        const std::string& prompt,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) override
    {
        lastPrompt = prompt;
        lastParams = params;
        lastGenerateHadVideo = false;
        lastVideoFrameSize = 0;
        lastFrameWidth = 0;
        lastFrameHeight = 0;
        if (simulateError) {
            return tl::unexpected(std::string("mock inferencer error"));
        }
        for (std::size_t i = 0; i < tokensToEmit.size(); ++i) {
            if (cancelCalled.load(std::memory_order_acquire)) {
                callback("", /*sentenceBoundary=*/false, /*done=*/true);
                return {};
            }
            const bool isDone = (i + 1 == tokensToEmit.size());
            callback(tokensToEmit[i], /*sentenceBoundary=*/false, isDone);
        }
        return {};
    }

    tl::expected<void, std::string> generateWithVideo(
        const std::string& prompt,
        std::span<const uint8_t> videoFrame,
        uint32_t frameWidth,
        uint32_t frameHeight,
        const GenerationParams& params,
        const std::function<void(std::string_view, bool, bool)>& callback) override
    {
        // Delegate to generate() first (which resets tracking fields),
        // then set video tracking fields so they survive the reset.
        auto result = generate(prompt, params, callback);
        lastGenerateHadVideo = true;
        lastVideoFrameSize = videoFrame.size();
        lastFrameWidth = frameWidth;
        lastFrameHeight = frameHeight;
        return result;
    }

    void cancel() noexcept override
    {
        cancelCalled.store(true, std::memory_order_release);
    }

    void unloadModel() noexcept override { modelLoaded_ = false; }
    [[nodiscard]] size_t currentVramUsageBytes() const noexcept override { return 0; }
    [[nodiscard]] std::string name() const override { return "mock-conversation"; }

private:
    bool modelLoaded_{false};
};

