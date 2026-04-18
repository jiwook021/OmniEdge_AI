// test_trt_whisper_inferencer.cpp — Inferencer-level Google Tests for TrtWhisperInferencer
//
// CPU tests (error paths):
//   - transcribe() before loadModel() returns error
//   - unloadModel() is safe to call multiple times
//   - currentVramUsageBytes() returns 0 before loadModel()
//
// GPU-gated tests (actual inference):
//   - loadModel() succeeds with valid engine + tokenizer dirs
//   - currentVramUsageBytes() > 0 after load
//   - transcribe(silence) returns near-empty or short text
//   - transcribe(30s mel) returns TranscribeResult with valid fields
//
// GPU tests require three env vars:
//   OE_TEST_STT_ENCODER_DIR   — path to Whisper TRT encoder engine
//   OE_TEST_STT_DECODER_DIR   — path to Whisper TRT decoder engine
//   OE_TEST_STT_TOKENIZER_DIR — path to tokenizer vocab (whisper_vocab.json)

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "stt/trt_whisper_inferencer.hpp"
#include "stt_inferencer.hpp"
#include "stt/mel_spectrogram.hpp"
#include "common/constants/whisper_constants.hpp"


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

std::string encoderDir()
{
	const char* env = std::getenv("OE_TEST_STT_ENCODER_DIR");
	return env ? std::string(env) : std::string{};
}

std::string decoderDir()
{
	const char* env = std::getenv("OE_TEST_STT_DECODER_DIR");
	return env ? std::string(env) : std::string{};
}

std::string tokenizerDir()
{
	const char* env = std::getenv("OE_TEST_STT_TOKENIZER_DIR");
	return env ? std::string(env) : std::string{};
}

bool gpuTestPathsAvailable()
{
	return !encoderDir().empty() && !decoderDir().empty() &&
	       !tokenizerDir().empty() &&
	       std::filesystem::exists(encoderDir()) &&
	       std::filesystem::exists(decoderDir()) &&
	       std::filesystem::exists(tokenizerDir());
}

/// Generate a 30-second silence mel spectrogram (all zeros).
/// Shape: [128 × 3000] row-major, matching Whisper's expected input.
std::vector<float> makeSilenceMel()
{
	constexpr int kMels   = 128;
	constexpr int kFrames = 3000;
	return std::vector<float>(kMels * kFrames, 0.0f);
}

/// Generate a 30-second sine tone mel spectrogram via MelSpectrogram.
/// Uses 440 Hz @ 16 kHz sample rate.
std::vector<float> makeSineMel()
{
	constexpr int kSampleRate = 16000;
	constexpr int kDurationS  = 30;
	constexpr int kNumSamples = kSampleRate * kDurationS;
	constexpr float kFreq     = 440.0f;
	constexpr float kTwoPi    = 2.0f * 3.14159265358979f;

	std::vector<float> pcm(kNumSamples);
	for (int i = 0; i < kNumSamples; ++i) {
		pcm[i] = 0.5f * std::sin(kTwoPi * kFreq * static_cast<float>(i) / kSampleRate);
	}

	MelSpectrogram mel(128, 400, 160, kSampleRate);
	auto melData = mel.compute(pcm.data(), pcm.size());

	// Pad or truncate to exactly [128 × 3000]
	constexpr int kExpected = 128 * 3000;
	melData.resize(kExpected, 0.0f);
	return melData;
}

} // namespace

// ---------------------------------------------------------------------------
// CPU-only tests — no label, no GPU required
// ---------------------------------------------------------------------------

TEST(TrtWhisperInferencerCpuTest, VramUsageZeroBeforeLoad)
{
	TrtWhisperInferencer inferencer;
	EXPECT_EQ(inferencer.currentVramUsageBytes(), 0u);
}

TEST(TrtWhisperInferencerCpuTest, UnloadModelIdempotent)
{
	TrtWhisperInferencer inferencer;
	EXPECT_NO_THROW(inferencer.unloadModel());
	EXPECT_NO_THROW(inferencer.unloadModel());
}

TEST(TrtWhisperInferencerCpuTest, TranscribeBeforeLoadReturnsError)
{
	TrtWhisperInferencer inferencer;
	auto mel = makeSilenceMel();
	auto result = inferencer.transcribe(
	    std::span<const float>(mel), 3000);
	EXPECT_FALSE(result.has_value());
	EXPECT_FALSE(result.error().empty());
}

TEST(TrtWhisperInferencerCpuTest, LoadModelMissingDirsThrows)
{
	TrtWhisperInferencer inferencer;
	EXPECT_THROW(
	    inferencer.loadModel("/tmp/no_such_encoder", "/tmp/no_such_decoder",
	                      "/tmp/no_such_tokenizer"),
	    std::runtime_error);
}

// ---------------------------------------------------------------------------
// GPU-dependent tests
// Require:
//   OE_TEST_STT_ENCODER_DIR
//   OE_TEST_STT_DECODER_DIR
//   OE_TEST_STT_TOKENIZER_DIR
// ---------------------------------------------------------------------------

class TrtWhisperInferencerGpuTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		if (!gpuTestPathsAvailable()) {
			GTEST_SKIP() << "GPU test paths not available "
			                "(set OE_TEST_STT_ENCODER_DIR, OE_TEST_STT_DECODER_DIR, "
			                "OE_TEST_STT_TOKENIZER_DIR)";
		}
		inferencer_ = std::make_unique<TrtWhisperInferencer>();
		ASSERT_NO_THROW(
		    inferencer_->loadModel(encoderDir(), decoderDir(), tokenizerDir()));
	}

	void TearDown() override
	{
		if (inferencer_) {
			inferencer_->unloadModel();
		}
	}

	std::unique_ptr<TrtWhisperInferencer> inferencer_;
};

TEST_F(TrtWhisperInferencerGpuTest, LoadModelSucceeds)
{
	EXPECT_GT(inferencer_->currentVramUsageBytes(), 0u);
}

TEST_F(TrtWhisperInferencerGpuTest, VramUsageNonZeroAfterLoad)
{
	// Whisper models typically use > 100 MB
	constexpr std::size_t kMinExpected = 100ull * 1024ull * 1024ull;
	EXPECT_GE(inferencer_->currentVramUsageBytes(), kMinExpected);
}

TEST_F(TrtWhisperInferencerGpuTest, TranscribeSilenceReturnsResult)
{
	auto mel = makeSilenceMel();
	auto result = inferencer_->transcribe(
	    std::span<const float>(mel), 3000);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Silence should produce short text (possibly empty or hallucination).
	// The key thing: no crash, valid result.
	EXPECT_GE(result->noSpeechProb, 0.0f);
	EXPECT_LE(result->noSpeechProb, 1.0f);
	EXPECT_FALSE(result->language.empty());
}

TEST_F(TrtWhisperInferencerGpuTest, TranscribeSilenceHighNoSpeechProb)
{
	auto mel = makeSilenceMel();
	auto result = inferencer_->transcribe(
	    std::span<const float>(mel), 3000);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Silence should ideally have high no-speech probability.
	// Be lenient — hallucination on silence is a known Whisper behavior.
	EXPECT_GE(result->noSpeechProb, 0.1f)
	    << "noSpeechProb should be non-trivial for silence, text='" << result->text << "'";
}

TEST_F(TrtWhisperInferencerGpuTest, TranscribeSineToneProducesResult)
{
	auto mel = makeSineMel();
	auto result = inferencer_->transcribe(
	    std::span<const float>(mel), 3000);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Sine tone is not speech — no crash is the main requirement.
	EXPECT_GE(result->noSpeechProb, 0.0f);
	EXPECT_LE(result->noSpeechProb, 1.0f);
}

TEST_F(TrtWhisperInferencerGpuTest, TranscribeReturnedLanguageIsEnglish)
{
	auto mel = makeSilenceMel();
	auto result = inferencer_->transcribe(
	    std::span<const float>(mel), 3000);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Default language should be English.
	EXPECT_EQ(result->language, "en");
}

TEST_F(TrtWhisperInferencerGpuTest, UnloadAndReloadWorks)
{
	inferencer_->unloadModel();
	EXPECT_EQ(inferencer_->currentVramUsageBytes(), 0u);

	ASSERT_NO_THROW(
	    inferencer_->loadModel(encoderDir(), decoderDir(), tokenizerDir()));
	EXPECT_GT(inferencer_->currentVramUsageBytes(), 0u);

	// Should still transcribe after reload.
	auto mel = makeSilenceMel();
	auto result = inferencer_->transcribe(
	    std::span<const float>(mel), 3000);
	ASSERT_TRUE(result.has_value()) << result.error();
}

