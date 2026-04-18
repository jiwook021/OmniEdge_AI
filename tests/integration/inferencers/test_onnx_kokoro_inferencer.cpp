#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tts_inferencer.hpp"
#include "tts/onnx_kokoro_inferencer.hpp"

// ---------------------------------------------------------------------------
// Tests for OnnxKokoroInferencer and the MockTTSInferencer factory.
//
// GPU tests (loadModel / synthesize with a real ONNX model) are tagged
// LABELS gpu and require the model file at the configured path.
//
// CPU-only tests (MockTTSInferencer, error paths) have no label and run in CI.
// ---------------------------------------------------------------------------


// ===========================================================================
// MockTTSInferencer — defined here, registered via createMockTTSInferencer()
// ===========================================================================

/**
 * @brief Returns silent PCM at a fixed 24 kHz / 0.5 s length.
 *
 * Used to test TTSNode without GPU or model files.
 */
class MockTTSInferencer : public TTSInferencer {
public:
	[[nodiscard]] tl::expected<void, std::string> loadModel(
		const std::string& /*onnxModelPath*/,
		const std::string& /*voiceDir*/) override
	{
		loaded_ = true;
		return {};
	}

	[[nodiscard]] tl::expected<std::vector<float>, std::string> synthesize(
		std::string_view text,
		std::string_view /*voiceName*/,
		float            /*speed*/) override
	{
		if (!loaded_) {
			return tl::unexpected(std::string("model not loaded"));
		}
		if (text.empty()) {
			return tl::unexpected(std::string("empty text"));
		}
		// Return 0.5 s of silence at 24 kHz.
		return std::vector<float>(12'000, 0.0f);
	}

	void unloadModel() noexcept override { loaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const override { return 0; }

	[[nodiscard]] std::string name() const override { return "mock-tts"; }

private:
	bool loaded_{false};
};

// Factory registered in the production tts_inferencer.hpp header.
std::unique_ptr<TTSInferencer> createMockTTSInferencer()
{
	return std::make_unique<MockTTSInferencer>();
}

// ===========================================================================
// MockTTSInferencer behaviour tests (CPU-only)
// ===========================================================================

TEST(MockTTSInferencerTest, SynthesizeBeforeLoadFails)
{
	auto inferencer = createMockTTSInferencer();
	auto result  = inferencer->synthesize("hello", "af_heart", 1.0f);
	EXPECT_FALSE(result.has_value());
}

TEST(MockTTSInferencerTest, SynthesizeAfterLoadSucceeds)
{
	auto inferencer = createMockTTSInferencer();
	ASSERT_TRUE(inferencer->loadModel("/fake", "/fake").has_value());

	auto result = inferencer->synthesize("Hello world.", "af_heart", 1.0f);
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(result->size(), 12'000u);
}

TEST(MockTTSInferencerTest, SynthesizeEmptyTextFails)
{
	auto inferencer = createMockTTSInferencer();
	inferencer->loadModel("/fake", "/fake");
	auto result = inferencer->synthesize("", "af_heart", 1.0f);
	EXPECT_FALSE(result.has_value());
}

TEST(MockTTSInferencerTest, UnloadThenSynthesizeFails)
{
	auto inferencer = createMockTTSInferencer();
	inferencer->loadModel("/fake", "/fake");
	inferencer->unloadModel();
	auto result = inferencer->synthesize("hello", "af_heart", 1.0f);
	EXPECT_FALSE(result.has_value());
}

TEST(MockTTSInferencerTest, CurrentVramZero)
{
	auto inferencer = createMockTTSInferencer();
	EXPECT_EQ(inferencer->currentVramUsageBytes(), 0u);
}

TEST(MockTTSInferencerTest, NameIsMock)
{
	auto inferencer = createMockTTSInferencer();
	EXPECT_EQ(inferencer->name(), "mock-tts");
}

// ===========================================================================
// OnnxKokoroInferencer — model-absent error path (CPU-only)
// ===========================================================================

TEST(OnnxKokoroInferencerTest, LoadModelMissingFileReturnsError)
{
	OnnxKokoroInferencer inferencer;
	auto result = inferencer.loadModel(
		"/nonexistent/path/kokoro.onnx",
		"/nonexistent/voices");
	EXPECT_FALSE(result.has_value());
	EXPECT_FALSE(result.error().empty());
}

TEST(OnnxKokoroInferencerTest, SynthesizeBeforeLoadReturnsError)
{
	OnnxKokoroInferencer inferencer;
	auto result = inferencer.synthesize("Hello.", "af_heart", 1.0f);
	EXPECT_FALSE(result.has_value());
}

TEST(OnnxKokoroInferencerTest, NameIsOnnxKokoro)
{
	OnnxKokoroInferencer inferencer;
	EXPECT_EQ(inferencer.name(), "onnx-kokoro");
}

TEST(OnnxKokoroInferencerTest, CurrentVramZeroBeforeLoad)
{
	OnnxKokoroInferencer inferencer;
	EXPECT_EQ(inferencer.currentVramUsageBytes(), 0u);
}

TEST(OnnxKokoroInferencerTest, UnloadBeforeLoadIsSafe)
{
	OnnxKokoroInferencer inferencer;
	// Must not crash or throw.
	EXPECT_NO_THROW(inferencer.unloadModel());
}

// ===========================================================================
// OnnxKokoroInferencer — GPU-dependent tests
// Skipped automatically in CPU-only CI (ctest -LE gpu).
// ===========================================================================

// These tests require:
//   - A real kokoro-v1_0-int8.onnx at OMNIEDGE_ONNX_MODEL env var
//   - A voice directory with af_heart.npy at OMNIEDGE_VOICE_DIR env var
//   - A CUDA device
//
// Run manually: OMNIEDGE_ONNX_MODEL=./models/onnx/kokoro-v1_0-int8.onnx \
//               OMNIEDGE_VOICE_DIR=./models/kokoro/voices \
//               ctest -L gpu -R test_onnx_kokoro_inferencer

static const char* gModelPath = std::getenv("OMNIEDGE_ONNX_MODEL");
static const char* gVoiceDir  = std::getenv("OMNIEDGE_VOICE_DIR");

class OnnxKokoroGpuTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		if (!gModelPath || !gVoiceDir) {
			GTEST_SKIP() <<
				"Set OMNIEDGE_ONNX_MODEL and OMNIEDGE_VOICE_DIR to run GPU tests.";
		}
	}
	OnnxKokoroInferencer inferencer_;
};

TEST_F(OnnxKokoroGpuTest, LoadModelSucceeds)
{
	auto result = inferencer_.loadModel(gModelPath, gVoiceDir);
	EXPECT_TRUE(result.has_value()) << result.error();
}

TEST_F(OnnxKokoroGpuTest, SynthesizeShortSentenceReturnsAudio)
{
	ASSERT_TRUE(inferencer_.loadModel(gModelPath, gVoiceDir).has_value());

	auto result = inferencer_.synthesize("Hello.", "af_heart", 1.0f);
	ASSERT_TRUE(result.has_value()) << result.error();
	// At 24 kHz, "Hello." should be at least 0.2 s = 4800 samples.
	EXPECT_GT(result->size(), 4800u);
}

TEST_F(OnnxKokoroGpuTest, VramUsageNonZeroAfterLoad)
{
	ASSERT_TRUE(inferencer_.loadModel(gModelPath, gVoiceDir).has_value());
	EXPECT_GT(inferencer_.currentVramUsageBytes(), 0u);
}

TEST_F(OnnxKokoroGpuTest, SynthesizeSpeedHalfProducesLongerAudio)
{
	ASSERT_TRUE(inferencer_.loadModel(gModelPath, gVoiceDir).has_value());

	auto normal = inferencer_.synthesize("Hello world.", "af_heart", 1.0f);
	auto slow   = inferencer_.synthesize("Hello world.", "af_heart", 0.5f);
	ASSERT_TRUE(normal.has_value());
	ASSERT_TRUE(slow.has_value());
	// Slower speed → more samples.
	EXPECT_GT(slow->size(), normal->size());
}

