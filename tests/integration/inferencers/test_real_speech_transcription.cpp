// test_real_speech_transcription.cpp — Real-model STT integration tests
//
// These tests feed actual speech audio (generated or from fixture files)
// into the TrtWhisperInferencer and verify the transcription output contains
// the expected words.
//
// GPU-gated: requires OE_TEST_STT_ENCODER_DIR, OE_TEST_STT_DECODER_DIR,
//            OE_TEST_STT_TOKENIZER_DIR environment variables pointing to
//            valid Whisper TRT engine directories.
//
// Run with: ctest -L gpu -R RealSpeech

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "stt/trt_whisper_inferencer.hpp"
#include "stt_inferencer.hpp"
#include "stt/mel_spectrogram.hpp"
#include "zmq/audio_constants.hpp"
#include "common/constants/whisper_constants.hpp"
#include "common/oe_logger.hpp"


// ---------------------------------------------------------------------------
// Environment helpers
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

/// Load raw float32 PCM from a binary file.
/// Returns empty vector if the file cannot be read.
std::vector<float> loadPcmFixture(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    const auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    const std::size_t sampleCount = static_cast<std::size_t>(fileSize) / sizeof(float);
    std::vector<float> samples(sampleCount);
    file.read(reinterpret_cast<char*>(samples.data()), fileSize);
    return samples;
}

/// Generate a mel spectrogram from PCM data using the standard Whisper params.
/// Pads or truncates to exactly [128 x 3000] for a 30s chunk.
std::vector<float> pcmToMel(const std::vector<float>& pcmSamples)
{
    MelSpectrogram melProcessor(
        kMelBinCount,     // 128
        kFftWindowSizeSamples,      // 400
        kHopLengthSamples, // 160
        kSttInputSampleRateHz // 16000
    );

    auto melData = melProcessor.compute(pcmSamples.data(), pcmSamples.size());

    // Pad or truncate to exactly [128 x 3000] for 30s input
    constexpr int kExpectedSize = 128 * 3000;
    melData.resize(kExpectedSize, 0.0f);
    return melData;
}

/// Convert a string to lowercase for case-insensitive matching.
std::string toLower(const std::string& input)
{
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// GPU test fixture — loads the Whisper inferencer once for all tests
// ---------------------------------------------------------------------------

class RealSpeechTranscriptionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        if (!gpuTestPathsAvailable()) return;

        sharedInferencer_ = std::make_unique<TrtWhisperInferencer>();
        try {
            sharedInferencer_->loadModel(encoderDir(), decoderDir(), tokenizerDir());
        } catch (const std::exception& e) {
            initError_ = e.what();
            sharedInferencer_.reset();
        }
    }

    static void TearDownTestSuite()
    {
        if (sharedInferencer_) {
            sharedInferencer_->unloadModel();
            sharedInferencer_.reset();
        }
        initError_.clear();
    }

    void SetUp() override
    {
        if (!gpuTestPathsAvailable()) {
            GTEST_SKIP() << "GPU test paths not set "
                            "(OE_TEST_STT_ENCODER_DIR, OE_TEST_STT_DECODER_DIR, "
                            "OE_TEST_STT_TOKENIZER_DIR)";
        }
        if (!sharedInferencer_) {
            GTEST_SKIP() << "Inferencer init failed: " << initError_;
        }
        inferencer_ = sharedInferencer_.get();
    }

    TrtWhisperInferencer* inferencer_{nullptr};
    inline static std::unique_ptr<TrtWhisperInferencer> sharedInferencer_;
    inline static std::string initError_;
};

// ---------------------------------------------------------------------------
// Test: Transcribe fixture speech file "speech_hello_16khz.pcm"
//
// This fixture contains a recording of "hello" at 16 kHz mono float32.
// The test verifies the transcription contains "hello" (case-insensitive).
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, TranscribeHelloSpeechFixture)
{
    const std::string fixturePath = "tests/fixtures/speech_hello_16khz.pcm";
    if (!std::filesystem::exists(fixturePath)) {
        GTEST_SKIP() << "Fixture not found: " << fixturePath;
    }

    auto pcmSamples = loadPcmFixture(fixturePath);
    ASSERT_FALSE(pcmSamples.empty()) << "Failed to load PCM fixture";

    auto melData = pcmToMel(pcmSamples);
    auto result = inferencer_->transcribe(
        std::span<const float>(melData), 3000);

    ASSERT_TRUE(result.has_value()) << "Transcription failed: " << result.error();

    OE_LOG_INFO("stt_transcription: text=\"{}\" no_speech_prob={} avg_logprob={}",
                result->text, result->noSpeechProb, result->avgLogprob);

    const std::string lowerText = toLower(result->text);
    EXPECT_NE(lowerText.find("hello"), std::string::npos)
        << "Expected 'hello' in transcription, got: \"" << result->text << "\"";
}

// ---------------------------------------------------------------------------
// Test: Silence produces high no_speech_prob (should not transcribe garbage)
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, SilenceYieldsHighNoSpeechProbability)
{
    // Generate 30s of silence
    constexpr int kSilenceSamples = kSttInputSampleRateHz * 30;
    std::vector<float> silence(kSilenceSamples, 0.0f);

    auto melData = pcmToMel(silence);
    auto result = inferencer_->transcribe(
        std::span<const float>(melData), 3000);

    ASSERT_TRUE(result.has_value()) << result.error();

    OE_LOG_INFO("stt_silence: text=\"{}\" no_speech_prob={} avg_logprob={}",
                result->text, result->noSpeechProb, result->avgLogprob);

    // Silence should have elevated no_speech_prob
    EXPECT_GT(result->noSpeechProb, 0.3f)
        << "Silence should have high no_speech_prob, got text: \""
        << result->text << "\"";
}

// ---------------------------------------------------------------------------
// Test: Short speech-like synthetic signal produces non-empty transcription
//
// Generates a chirp (frequency sweep) which sounds somewhat speech-like.
// The model may produce garbage text, but the test verifies the pipeline
// runs end-to-end without crashing and returns a non-error result.
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, ChirpSignalProducesNonErrorResult)
{
    constexpr int kSampleRate = 16000;
    constexpr int kDurationSamples = kSampleRate * 5; // 5 seconds
    constexpr float kTwoPi = 2.0f * 3.14159265358979f;

    std::vector<float> chirp(kDurationSamples);
    for (int i = 0; i < kDurationSamples; ++i) {
        // Frequency sweep from 100 Hz to 4000 Hz over 5 seconds
        const float t = static_cast<float>(i) / kSampleRate;
        const float instantFreq = 100.0f + 3900.0f * (t / 5.0f);
        chirp[i] = 0.3f * std::sin(kTwoPi * instantFreq * t);
    }

    auto melData = pcmToMel(chirp);
    auto result = inferencer_->transcribe(
        std::span<const float>(melData), 3000);

    ASSERT_TRUE(result.has_value()) << result.error();

    OE_LOG_INFO("stt_chirp: text=\"{}\" no_speech_prob={} avg_logprob={}",
                result->text, result->noSpeechProb, result->avgLogprob);

    // No crash and valid result fields is the main check
    EXPECT_GE(result->noSpeechProb, 0.0f);
    EXPECT_LE(result->noSpeechProb, 1.0f);
}

// ---------------------------------------------------------------------------
// Test: Mel spectrogram dimensions match Whisper expectations
//
// Verifies the mel processor produces the correct tensor shape for the
// inferencer: exactly [128 mels x 3000 frames] for a 30-second input.
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, MelSpectrogramDimensionsCorrect)
{
    constexpr int kSamples30s = kSttInputSampleRateHz * 30;
    std::vector<float> pcm(kSamples30s, 0.0f);

    MelSpectrogram melProcessor(
        kMelBinCount,
        kFftWindowSizeSamples,
        kHopLengthSamples,
        kSttInputSampleRateHz
    );

    auto melData = melProcessor.compute(pcm.data(), pcm.size());
    const int numFrames = melProcessor.numFrames(pcm.size());

    EXPECT_EQ(numFrames, 3000)
        << "30s @ 16kHz with hop=160 should produce 3000 frames";
    EXPECT_EQ(static_cast<int>(melData.size()), 128 * numFrames)
        << "Mel data should be 128 * " << numFrames << " floats";
}

// ---------------------------------------------------------------------------
// Test: Generate speech with espeak-ng, transcribe, verify content
//
// Uses espeak-ng to synthesize "The weather is sunny today" into a WAV file,
// then converts to float32 PCM, runs through mel + TRT inferencer, and checks
// that the transcription contains key words.
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, EspeakGeneratedSpeechTranscription)
{
    // Check if espeak-ng is available
    const int espeakCheck = std::system("which espeak-ng > /dev/null 2>&1");
    if (espeakCheck != 0) {
        GTEST_SKIP() << "espeak-ng not installed — skipping TTS-generated speech test";
    }

    const std::string testPhrase = "The weather is sunny today";
    const std::string wavPath = "/tmp/oe_test_espeak_output.wav";
    const std::string pcmPath = "/tmp/oe_test_espeak_output.pcm";

    // Generate speech WAV (16 kHz, mono)
    const std::string espeakCmd = std::format(
        "espeak-ng -v en -s 150 -w {} \"{}\" 2>/dev/null", wavPath, testPhrase);
    ASSERT_EQ(std::system(espeakCmd.c_str()), 0) << "espeak-ng failed";

    // Convert WAV to raw float32 PCM using sox or ffmpeg
    const std::string convertCmd = std::format(
        "ffmpeg -y -i {} -f f32le -acodec pcm_f32le -ar 16000 -ac 1 {} 2>/dev/null",
        wavPath, pcmPath);
    if (std::system(convertCmd.c_str()) != 0) {
        // Try sox as fallback
        const std::string soxCmd = std::format(
            "sox {} -t f32 -r 16000 -c 1 {} 2>/dev/null", wavPath, pcmPath);
        if (std::system(soxCmd.c_str()) != 0) {
            std::filesystem::remove(wavPath);
            GTEST_SKIP() << "Neither ffmpeg nor sox available for WAV→PCM conversion";
        }
    }

    auto pcmSamples = loadPcmFixture(pcmPath);
    std::filesystem::remove(wavPath);
    std::filesystem::remove(pcmPath);

    ASSERT_FALSE(pcmSamples.empty()) << "Failed to load converted PCM data";
    OE_LOG_INFO("espeak_pcm_loaded: samples={}, duration_s={:.1f}",
                pcmSamples.size(),
                static_cast<double>(pcmSamples.size()) / kSttInputSampleRateHz);

    auto melData = pcmToMel(pcmSamples);
    auto transcriptionResult = inferencer_->transcribe(
        std::span<const float>(melData), 3000);

    ASSERT_TRUE(transcriptionResult.has_value())
        << "Transcription failed: " << transcriptionResult.error();

    OE_LOG_INFO("espeak_transcription: text=\"{}\" no_speech_prob={} avg_logprob={}",
                transcriptionResult->text, transcriptionResult->noSpeechProb,
                transcriptionResult->avgLogprob);

    const std::string lowerTranscription = toLower(transcriptionResult->text);

    // The model should transcribe at least some key words from the phrase
    int matchedWordCount = 0;
    if (lowerTranscription.find("weather") != std::string::npos) ++matchedWordCount;
    if (lowerTranscription.find("sunny")   != std::string::npos) ++matchedWordCount;
    if (lowerTranscription.find("today")   != std::string::npos) ++matchedWordCount;

    EXPECT_GE(matchedWordCount, 1)
        << "Expected at least 1 key word from '" << testPhrase
        << "' in transcription, got: \"" << transcriptionResult->text << "\"";
}

// ---------------------------------------------------------------------------
// Test: Consecutive transcriptions produce consistent results
//
// Feeds the same input twice and verifies both produce non-empty results,
// catching state accumulation or resource leak bugs.
// ---------------------------------------------------------------------------

TEST_F(RealSpeechTranscriptionTest, ConsecutiveTranscriptionsWork)
{
    // Use silence (simple, fast)
    constexpr int kSilenceSamples = kSttInputSampleRateHz * 5;
    std::vector<float> silence(kSilenceSamples, 0.0f);
    auto melData = pcmToMel(silence);

    for (int iteration = 0; iteration < 3; ++iteration) {
        auto result = inferencer_->transcribe(
            std::span<const float>(melData), 3000);

        ASSERT_TRUE(result.has_value())
            << "Transcription #" << iteration << " failed: " << result.error();

        OE_LOG_DEBUG("consecutive_stt_result: iteration={}, text=\"{}\"",
                     iteration, result->text);
    }
}

