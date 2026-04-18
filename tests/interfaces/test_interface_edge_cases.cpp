// ---------------------------------------------------------------------------
// test_interface_edge_cases.cpp -- Interface contract edge case tests
//
// Tests: STT empty/single-frame mel, noSpeechProb at threshold,
//        TTS empty text / long text / invalid voice,
//        Conversation empty prompt / cancel during callback / error.
//
// Uses mock inferencers from tests/mocks/ -- no GPU required.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "stt_inferencer.hpp"
#include "tts_inferencer.hpp"
#include "conversation_inferencer.hpp"
#include "tests/mocks/mock_stt_inferencer.hpp"
#include "tests/mocks/mock_tts_inferencer.hpp"
#include "tests/mocks/mock_conversation_inferencer.hpp"

namespace {

// =========================================================================
// STT Inferencer edge cases (MockSTTInferencer)
// =========================================================================

class STTEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mock_.loadModel("enc", "dec", "tok");
    }

    MockSTTInferencer mock_;
};

TEST_F(STTEdgeCaseTest, TranscribeEmptyMelSpectrogram_ReturnsError)
{
    // Empty mel spectrogram (zero frames) -- inferencer should return an error,
    // not crash or produce garbage.
    std::span<const float> emptyMel;
    auto result = mock_.transcribe(emptyMel, /*numFrames=*/0);

    ASSERT_FALSE(result.has_value());
    EXPECT_FALSE(result.error().empty())
        << "Error message should be non-empty for empty mel input";
}

TEST_F(STTEdgeCaseTest, TranscribeSingleFrameMel_ReturnsResult)
{
    // A single-frame mel spectrogram is an edge case: minimal valid input.
    // The mock should return a result since the span is non-empty.
    constexpr int kMelBins = 80;  // Whisper uses 80 mel bins
    std::vector<float> singleFrame(kMelBins, 0.0f);

    auto result = mock_.transcribe(
        std::span<const float>(singleFrame), /*numFrames=*/1);

    ASSERT_TRUE(result.has_value())
        << "Single-frame mel should produce a result (not an error): "
        << result.error();
    EXPECT_EQ(result->text, "Hello world");
}

TEST_F(STTEdgeCaseTest, NoSpeechProbExactlyAtThreshold)
{
    // When noSpeechProb is exactly 0.5, the result should still be returned
    // (the hallucination filter applies thresholding, not the inferencer itself).
    constexpr int kMelBins = 80;
    std::vector<float> mel(kMelBins * 10, 0.0f);

    mock_.cannedNoSpeechProb = 0.5f;

    auto result = mock_.transcribe(
        std::span<const float>(mel), /*numFrames=*/10);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->noSpeechProb, 0.5f);
}

TEST_F(STTEdgeCaseTest, TranscribeWithoutLoadModel_ReturnsError)
{
    // Calling transcribe before loadModel should return an error.
    MockSTTInferencer unloaded;
    constexpr int kMelBins = 80;
    std::vector<float> mel(kMelBins, 0.0f);

    auto result = unloaded.transcribe(
        std::span<const float>(mel), /*numFrames=*/1);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("not loaded"), std::string::npos);
}

TEST_F(STTEdgeCaseTest, UnloadThenTranscribe_ReturnsError)
{
    // Load, unload, then transcribe -- should fail.
    mock_.unloadModel();

    constexpr int kMelBins = 80;
    std::vector<float> mel(kMelBins, 0.0f);

    auto result = mock_.transcribe(
        std::span<const float>(mel), /*numFrames=*/1);

    ASSERT_FALSE(result.has_value());
}

// =========================================================================
// TTS Inferencer edge cases (MinimalMockInferencer)
// =========================================================================

class TTSEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        (void)mock_.loadModel("model.onnx", "voices/");
    }

    MinimalMockInferencer mock_;
};

TEST_F(TTSEdgeCaseTest, SynthesizeEmptyText_ReturnsError)
{
    // Synthesizing empty text is invalid -- should return an error.
    auto result = mock_.synthesize("", "af_heart", 1.0f);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("empty"), std::string::npos)
        << "Error should mention empty text: " << result.error();
}

TEST_F(TTSEdgeCaseTest, SynthesizeVeryLongText_Succeeds)
{
    // 10,000 characters is a realistic edge case for very long sentences.
    // The mock should handle it without crashing.
    const std::string longText(10'000, 'A');
    auto result = mock_.synthesize(longText, "af_heart", 1.0f);

    ASSERT_TRUE(result.has_value())
        << "Long text should succeed: " << result.error();
    // The mock returns 1 second of silence (24000 samples) regardless of text length.
    EXPECT_EQ(result->size(), 24'000u);
    EXPECT_EQ(mock_.synthesizeCallCount, 1);
    EXPECT_EQ(mock_.lastSynthesizedText, longText);
}

TEST_F(TTSEdgeCaseTest, SynthesizeWhitespaceOnlyText_Succeeds)
{
    // Whitespace-only text is a valid call (non-empty string).
    // The mock does not trim -- it just returns audio.
    auto result = mock_.synthesize("   ", "af_heart", 1.0f);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 24'000u);
}

TEST_F(TTSEdgeCaseTest, SynthesizeWithInvalidVoiceName_StillReturns)
{
    // The mock does not validate voice names. Production inferencers should
    // return an error for invalid voice IDs, but the mock is lenient.
    auto result = mock_.synthesize(
        "Hello.", "nonexistent_voice_xyz", 1.0f);

    // The mock succeeds regardless of voice name.
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(mock_.lastSynthesizedText, "Hello.");
}

TEST_F(TTSEdgeCaseTest, MultipleSynthesizeCalls_CountTracked)
{
    // Verify that call counting works across multiple invocations.
    for (int i = 0; i < 10; ++i) {
        auto result = mock_.synthesize(
            "Sentence " + std::to_string(i) + ".", "af_heart", 1.0f);
        ASSERT_TRUE(result.has_value());
    }
    EXPECT_EQ(mock_.synthesizeCallCount, 10);
}

// =========================================================================
// Conversation Inferencer edge cases (MockConversationInferencer)
// =========================================================================

class ConversationEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mockInferencer_.loadModel("model_dir");
    }

    MockConversationInferencer mockInferencer_;
};

TEST_F(ConversationEdgeCaseTest, GenerateWithEmptyPrompt_StillCallsCallback)
{
    // Empty prompt is a degenerate case but should not crash.
    std::vector<std::string> receivedTokens;
    bool done = false;

    GenerationParams params;
    auto result = mockInferencer_.generate(
        "",  // empty prompt
        params,
        [&](std::string_view token, bool /*boundary*/, bool isDone) {
            receivedTokens.emplace_back(token);
            done = isDone;
        });

    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_TRUE(done) << "Callback should have been called with done=true";
    EXPECT_EQ(mockInferencer_.lastPrompt, "");
    EXPECT_FALSE(receivedTokens.empty());
}

TEST_F(ConversationEdgeCaseTest, CancelDuringTokenCallback_NoCrash)
{
    // Cancel generation from within the token callback itself.
    mockInferencer_.tokensToEmit = {"A", "B", "C", "D", "E", "F", "G", "H"};
    std::vector<std::string> receivedTokens;

    GenerationParams params;
    auto result = mockInferencer_.generate(
        "prompt",
        params,
        [&](std::string_view token, bool /*boundary*/, bool /*done*/) {
            receivedTokens.emplace_back(token);
            if (receivedTokens.size() == 2) {
                mockInferencer_.cancel();
            }
        });

    ASSERT_TRUE(result.has_value()) << result.error();
    // Should have received at most 3 tokens: A, B, then cancel triggers.
    EXPECT_LE(receivedTokens.size(), 4u)
        << "Cancellation should stop token emission promptly";
}

TEST_F(ConversationEdgeCaseTest, GenerateWithErrorSimulation_ReturnsError)
{
    mockInferencer_.simulateError = true;

    GenerationParams params;
    auto result = mockInferencer_.generate(
        "Hello",
        params,
        [](std::string_view, bool, bool) {});

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("mock inferencer error"), std::string::npos);
}

TEST_F(ConversationEdgeCaseTest, SupportsNativeTtsReflectsSetting)
{
    mockInferencer_.nativeTts = true;
    EXPECT_TRUE(mockInferencer_.supportsNativeTts());

    mockInferencer_.nativeTts = false;
    EXPECT_FALSE(mockInferencer_.supportsNativeTts());
}

TEST_F(ConversationEdgeCaseTest, SupportsNativeSttReflectsSetting)
{
    mockInferencer_.nativeStt = true;
    EXPECT_TRUE(mockInferencer_.supportsNativeStt());

    mockInferencer_.nativeStt = false;
    EXPECT_FALSE(mockInferencer_.supportsNativeStt());
}

TEST_F(ConversationEdgeCaseTest, TranscribeReturnsConfiguredText)
{
    mockInferencer_.mockTranscription = "Test transcription";
    auto result = mockInferencer_.transcribe(std::span<const float>{}, 16000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "Test transcription");
}

TEST_F(ConversationEdgeCaseTest, TranscribeErrorWhenSimulateErrorSet)
{
    mockInferencer_.simulateError = true;
    auto result = mockInferencer_.transcribe(std::span<const float>{}, 16000);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("mock transcription error"), std::string::npos);
}

TEST_F(ConversationEdgeCaseTest, GenerationParamsPassedToInferencer)
{
    GenerationParams params;
    params.temperature = 1.5f;
    params.topP = 0.3f;
    params.maxTokens = 4096;

    auto result = mockInferencer_.generate(
        "test",
        params,
        [](std::string_view, bool, bool) {});

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(mockInferencer_.lastParams.temperature, 1.5f);
    EXPECT_FLOAT_EQ(mockInferencer_.lastParams.topP, 0.3f);
    EXPECT_EQ(mockInferencer_.lastParams.maxTokens, 4096);
}

} // namespace
