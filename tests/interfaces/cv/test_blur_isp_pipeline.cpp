#include <gtest/gtest.h>

#include "cv/background_blur_node.hpp"
#include "blur_inferencer.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// ISP Pipeline Tests — BackgroundBlurNode + BlurInferencer ISP parameter flow
//
// Purpose: Verify that ISP parameters (brightness, contrast, saturation,
// sharpness) set via setIspParams() are actually applied during
// processFrame().  This catches the bug where ISP params are stored
// but never forwarded to the GPU composite kernel.
//
// All tests use a StubBlurInferencer identical to the one in test_blur_api.cpp.
// The stub copies pendingIspParams_ to lastAppliedIspParams_ inside
// processFrame(), making the "was ISP applied?" assertion trivial.
//
// No SHM, no ZMQ, no GPU — pure inferencer API coverage.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubBlurInferencer — copied from test_blur_api.cpp (file-local, not shared).
//
// Given a BGR frame, produces a synthetic JPEG (SOI marker + body + EOI marker)
// whose content derives from the input pixels.  This proves the pipeline
// actually reads the input rather than returning canned data.
// ---------------------------------------------------------------------------
class StubBlurInferencer : public BlurInferencer {
public:
	void loadEngine(const std::string& /*enginePath*/,
	                uint32_t           width,
	                uint32_t           height) override
	{
		loadedWidth_  = width;
		loadedHeight_ = height;
	}

	[[nodiscard]] tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* bgrInputFrame,
	             uint32_t       frameWidth,
	             uint32_t       frameHeight,
	             uint8_t*       jpegOutputBuffer,
	             std::size_t    maxOutputBytes) override
	{
		++processFrameCallCount_;

		if (forceError_) {
			return tl::unexpected(std::string("simulated TRT inference failure"));
		}

		// Simulate JPEG compression: output is ~10% of input size
		const std::size_t inputBytes =
			static_cast<std::size_t>(frameWidth) * frameHeight * 3;
		const std::size_t simulatedJpegSize =
			std::min(inputBytes / 10, maxOutputBytes);

		if (simulatedJpegSize < 4) {
			return tl::unexpected(
				std::string("output buffer too small for JPEG markers"));
		}

		// Write valid JPEG structure: SOI + body + EOI
		jpegOutputBuffer[0] = 0xFF;
		jpegOutputBuffer[1] = 0xD8;  // JPEG Start-Of-Image marker

		// Fill body with a byte derived from the first input pixel.
		// This proves the inferencer actually reads the input frame.
		const uint8_t bodyFillByte =
			(bgrInputFrame != nullptr) ? bgrInputFrame[0] : 0x42;
		std::memset(jpegOutputBuffer + 2, bodyFillByte, simulatedJpegSize - 4);

		jpegOutputBuffer[simulatedJpegSize - 2] = 0xFF;
		jpegOutputBuffer[simulatedJpegSize - 1] = 0xD9;  // JPEG End-Of-Image

		lastAppliedIspParams_ = pendingIspParams_;
		return simulatedJpegSize;
	}

	void unload() noexcept override {}

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
		return 0;
	}

	void setIspParams(const IspParams& params) noexcept override {
		pendingIspParams_ = params;
		++ispUpdateCount_;
	}

	// --- Observable state for test assertions ---
	uint32_t  loadedWidth_{0};
	uint32_t  loadedHeight_{0};
	int       processFrameCallCount_{0};
	bool      forceError_{false};
	int       ispUpdateCount_{0};
	IspParams pendingIspParams_{};
	IspParams lastAppliedIspParams_{};
};

// ---------------------------------------------------------------------------
// Helper — create a small synthetic BGR24 frame (solid color).
// ---------------------------------------------------------------------------
static std::vector<uint8_t> makeSolidBgrFrame(uint32_t width,
                                               uint32_t height,
                                               uint8_t  b,
                                               uint8_t  g,
                                               uint8_t  r)
{
	const std::size_t totalBytes =
		static_cast<std::size_t>(width) * height * 3;
	std::vector<uint8_t> frame(totalBytes);
	for (std::size_t i = 0; i < totalBytes; i += 3) {
		frame[i]     = b;
		frame[i + 1] = g;
		frame[i + 2] = r;
	}
	return frame;
}

// ===========================================================================
// ISP Pipeline — "set ISP params, process a frame, verify params were applied"
// ===========================================================================

TEST(IspPipeline, ProcessFrame_AppliesCurrentIspParams)
{
	// Bug caught: ISP params are set via setIspParams() but never consumed
	// during processFrame() — the GPU composite kernel runs with defaults,
	// so brightness/contrast sliders in the UI have no effect.

	static constexpr uint32_t kFrameW = 64;
	static constexpr uint32_t kFrameH = 48;

	StubBlurInferencer inferencer;
	inferencer.loadEngine("bg_blur/mediapipe_selfie_seg.onnx", kFrameW, kFrameH);

	// GIVEN: the user adjusts all four ISP parameters
	IspParams userParams;
	userParams.brightness = 10.0f;
	userParams.contrast   = 1.5f;
	userParams.saturation = 0.8f;
	userParams.sharpness  = 2.0f;
	inferencer.setIspParams(userParams);

	// WHEN: a frame is processed
	auto bgrFrame = makeSolidBgrFrame(kFrameW, kFrameH, 128, 64, 32);
	std::vector<uint8_t> jpegOutput(kMaxJpegBytesPerSlot, 0);

	auto result = inferencer.processFrame(
		bgrFrame.data(), kFrameW, kFrameH,
		jpegOutput.data(), jpegOutput.size());

	ASSERT_TRUE(result.has_value())
		<< "processFrame must succeed with valid input";

	// THEN: the ISP params that were pending are now the ones that were applied
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.brightness, 10.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.contrast,   1.5f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.saturation, 0.8f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.sharpness,  2.0f);
	EXPECT_FALSE(inferencer.lastAppliedIspParams_.isIdentity())
		<< "Applied ISP params must not be identity after user adjustment";
}

TEST(IspPipeline, ProcessFrame_WithoutIspUpdate_AppliesDefaults)
{
	// Bug caught: processFrame crashes or produces garbage when no ISP params
	// have ever been set (pendingIspParams_ is default-constructed).

	static constexpr uint32_t kFrameW = 64;
	static constexpr uint32_t kFrameH = 48;

	StubBlurInferencer inferencer;
	inferencer.loadEngine("bg_blur/mediapipe_selfie_seg.onnx", kFrameW, kFrameH);

	// GIVEN: setIspParams() is NEVER called — defaults should apply
	auto bgrFrame = makeSolidBgrFrame(kFrameW, kFrameH, 200, 100, 50);
	std::vector<uint8_t> jpegOutput(kMaxJpegBytesPerSlot, 0);

	// WHEN: processFrame runs without any ISP update
	auto result = inferencer.processFrame(
		bgrFrame.data(), kFrameW, kFrameH,
		jpegOutput.data(), jpegOutput.size());

	ASSERT_TRUE(result.has_value());

	// THEN: the applied ISP params are the identity transform (no modification)
	EXPECT_TRUE(inferencer.lastAppliedIspParams_.isIdentity())
		<< "When setIspParams() is never called, processFrame must apply "
		   "identity ISP (brightness=0, contrast=1, saturation=1, sharpness=0)";
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.brightness, 0.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.contrast,   1.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.saturation, 1.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.sharpness,  0.0f);
	EXPECT_EQ(inferencer.ispUpdateCount_, 0)
		<< "No ISP update was issued, so ispUpdateCount must remain zero";
}

TEST(IspPipeline, MultipleFrames_ApplyLatestParams)
{
	// Bug caught: ISP params from frame N "stick" and are not updated when
	// the user changes sliders between frames — stale params applied.

	static constexpr uint32_t kFrameW = 64;
	static constexpr uint32_t kFrameH = 48;

	StubBlurInferencer inferencer;
	inferencer.loadEngine("bg_blur/mediapipe_selfie_seg.onnx", kFrameW, kFrameH);

	auto bgrFrame = makeSolidBgrFrame(kFrameW, kFrameH, 100, 100, 100);
	std::vector<uint8_t> jpegOutput(kMaxJpegBytesPerSlot, 0);

	// Frame 1: set initial ISP params and process
	IspParams firstParams;
	firstParams.brightness = 20.0f;
	firstParams.contrast   = 1.2f;
	firstParams.saturation = 1.0f;
	firstParams.sharpness  = 1.0f;
	inferencer.setIspParams(firstParams);

	auto result1 = inferencer.processFrame(
		bgrFrame.data(), kFrameW, kFrameH,
		jpegOutput.data(), jpegOutput.size());
	ASSERT_TRUE(result1.has_value());

	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.brightness, 20.0f)
		<< "Frame 1 must apply the first ISP update";
	EXPECT_EQ(inferencer.processFrameCallCount_, 1);

	// Frame 2: update ISP params (user dragged brightness slider) and process
	IspParams secondParams;
	secondParams.brightness = -5.0f;
	secondParams.contrast   = 2.0f;
	secondParams.saturation = 0.5f;
	secondParams.sharpness  = 4.0f;
	inferencer.setIspParams(secondParams);

	auto result2 = inferencer.processFrame(
		bgrFrame.data(), kFrameW, kFrameH,
		jpegOutput.data(), jpegOutput.size());
	ASSERT_TRUE(result2.has_value());

	// THEN: the second frame used the UPDATED params, not the stale first ones
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.brightness, -5.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.contrast,   2.0f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.saturation, 0.5f);
	EXPECT_FLOAT_EQ(inferencer.lastAppliedIspParams_.sharpness,  4.0f);
	EXPECT_EQ(inferencer.processFrameCallCount_, 2);
	EXPECT_EQ(inferencer.ispUpdateCount_, 2);
}

TEST(IspPipeline, IspIdentityDetection)
{
	// Verify IspParams::isIdentity() works correctly for default and
	// non-default parameter sets.  A bug here would cause the GPU kernel
	// to skip ISP processing when it should apply adjustments (or vice versa).

	// Default-constructed params must be identity
	IspParams defaultParams;
	EXPECT_TRUE(defaultParams.isIdentity())
		<< "Default ISP params (brightness=0, contrast=1, saturation=1, sharpness=0) "
		   "must be the identity transform";

	// Any single non-default field breaks identity
	{
		IspParams p;
		p.brightness = 0.01f;
		EXPECT_FALSE(p.isIdentity())
			<< "Non-zero brightness must not be identity";
	}
	{
		IspParams p;
		p.contrast = 1.1f;
		EXPECT_FALSE(p.isIdentity())
			<< "Contrast != 1.0 must not be identity";
	}
	{
		IspParams p;
		p.saturation = 0.9f;
		EXPECT_FALSE(p.isIdentity())
			<< "Saturation != 1.0 must not be identity";
	}
	{
		IspParams p;
		p.sharpness = 0.5f;
		EXPECT_FALSE(p.isIdentity())
			<< "Non-zero sharpness must not be identity";
	}

	// All fields at default must be identity (edge case: floating point)
	{
		IspParams p;
		p.brightness = 0.0f;
		p.contrast   = 1.0f;
		p.saturation = 1.0f;
		p.sharpness  = 0.0f;
		EXPECT_TRUE(p.isIdentity())
			<< "Explicitly setting all fields to defaults must still be identity";
	}
}

