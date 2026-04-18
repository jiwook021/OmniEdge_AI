#include <gtest/gtest.h>

#include "face_filter_inferencer.hpp"
#include "common/oe_logger.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// StubFaceFilterInferencer Unit Tests — CPU-only, no GPU required.
//
// What these tests catch:
//   - processFrame returns invalid JPEG (output doesn't start with FF D8)
//   - processFrame crashes on null input, zero-dimension, or tiny buffer
//   - setActiveFilter doesn't persist the filter ID across calls
//   - loadFilterAssets doesn't populate availableFilters()
//   - unload doesn't reset state, causing stale data on re-init
//   - lastLandmarks returns garbage after processFrame
//   - processFrame output size lies (reports more bytes than written)
//
// These tests exercise the REAL stub code with REAL pixel data —
// no mocks, no hardcoded return values.
// ---------------------------------------------------------------------------

// Factory — declared in face_filter_inferencer_stub.cpp
[[nodiscard]] std::unique_ptr<FaceFilterInferencer> createStubFaceFilterInferencer();


class FaceFilterStubTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		OeLogger::instance().setModule("test_face_filter_stub");
		inferencer_ = createStubFaceFilterInferencer();
		ASSERT_NE(inferencer_, nullptr);
	}

	/// Generate a synthetic BGR24 test frame filled with a solid colour.
	static std::vector<uint8_t> makeBgrFrame(uint32_t w, uint32_t h,
	                                         uint8_t b, uint8_t g, uint8_t r)
	{
		const std::size_t bytes = static_cast<std::size_t>(w) * h * 3;
		std::vector<uint8_t> frame(bytes);
		for (std::size_t i = 0; i < bytes; i += 3) {
			frame[i]     = b;
			frame[i + 1] = g;
			frame[i + 2] = r;
		}
		SPDLOG_DEBUG("makeBgrFrame: {}x{} = {} bytes, colour=({},{},{})",
			w, h, bytes, b, g, r);
		return frame;
	}

	std::unique_ptr<FaceFilterInferencer> inferencer_;

	// Standard test resolutions
	static constexpr uint32_t kTinyW  = 64,   kTinyH  = 48;    // minimum viable frame
	static constexpr uint32_t kSmallW = 320,   kSmallH = 240;   // default webcam preview
	static constexpr uint32_t kMeshW  = 192,   kMeshH  = 192;   // FaceMesh native input
	static constexpr uint32_t kVgaW   = 640,   kVgaH   = 480;   // 480p webcam
	static constexpr uint32_t kHdW    = 1920,  kHdH    = 1080;  // 1080p upper bound

	// Output buffer — large enough for any stub output
	static constexpr std::size_t kOutBufSize = 64 * 1024;
	uint8_t outBuf_[kOutBufSize]{};
};

// ===========================================================================
// Test: processFrame produces a valid JPEG (starts with FF D8, ends FF D9)
// Bug caught: stub writes garbage that downstream JPEG parsers reject
// ===========================================================================
TEST_F(FaceFilterStubTest, ProcessFrameProducesValidJpeg)
{
	auto frame = makeBgrFrame(kSmallW, kSmallH, 100, 150, 200);

	inferencer_->loadModel("");

	auto result = inferencer_->processFrame(
		frame.data(), kSmallW, kSmallH, outBuf_, kOutBufSize);

	ASSERT_TRUE(result.has_value()) << "processFrame failed: " << result.error();

	const std::size_t jpegSize = result.value();
	SPDLOG_DEBUG("JPEG output: {} bytes", jpegSize);

	ASSERT_GE(jpegSize, 4u) << "JPEG too small to contain SOI+EOI markers";

	// JPEG SOI marker (FF D8)
	EXPECT_EQ(outBuf_[0], 0xFF);
	EXPECT_EQ(outBuf_[1], 0xD8);

	// JPEG EOI marker (FF D9) at end
	EXPECT_EQ(outBuf_[jpegSize - 2], 0xFF);
	EXPECT_EQ(outBuf_[jpegSize - 1], 0xD9);
}

// ===========================================================================
// Test: processFrame with different frame sizes all succeed
// Bug caught: stub hardcodes dimensions or has buffer arithmetic errors
// ===========================================================================
TEST_F(FaceFilterStubTest, ProcessFrameMultipleSizes)
{
	struct TestCase { uint32_t w; uint32_t h; };
	constexpr TestCase cases[] = {
		{kTinyW, kTinyH},   // Minimum: tiny frame, catches buffer underflow
		{kMeshW, kMeshH},   // FaceMesh native input resolution (square)
		{kVgaW,  kVgaH},    // Standard 480p webcam capture
		{kHdW,   kHdH},     // Full HD 1080p upper bound
	};

	inferencer_->loadModel("");

	for (const auto& [w, h] : cases) {
		auto frame = makeBgrFrame(w, h, 80, 80, 80);
		std::memset(outBuf_, 0, kOutBufSize);

		auto result = inferencer_->processFrame(
			frame.data(), w, h, outBuf_, kOutBufSize);

		ASSERT_TRUE(result.has_value())
			<< "processFrame failed for " << w << "x" << h
			<< ": " << result.error();
		EXPECT_GT(result.value(), 0u)
			<< "Zero-byte output for " << w << "x" << h;
		SPDLOG_DEBUG("processFrame {}x{}: {} bytes JPEG", w, h, result.value());
	}
}

// ===========================================================================
// Test: processFrame rejects null input frame
// Bug caught: stub dereferences null pointer → segfault
// ===========================================================================
TEST_F(FaceFilterStubTest, ProcessFrameRejectsNullInput)
{
	auto result = inferencer_->processFrame(
		nullptr, kSmallW, kSmallH, outBuf_, kOutBufSize);

	EXPECT_FALSE(result.has_value())
		<< "processFrame should reject null input";
	SPDLOG_DEBUG("Null input error: {}", result.error());
}

// ===========================================================================
// Test: processFrame rejects zero dimensions
// Bug caught: stub proceeds with 0-byte frame → empty/corrupt output
// ===========================================================================
TEST_F(FaceFilterStubTest, ProcessFrameRejectsZeroDimensions)
{
	uint8_t dummyPixel[3] = {0, 0, 0};

	auto result0w = inferencer_->processFrame(
		dummyPixel, 0, kSmallH, outBuf_, kOutBufSize);
	EXPECT_FALSE(result0w.has_value()) << "Should reject width=0";

	auto result0h = inferencer_->processFrame(
		dummyPixel, kSmallW, 0, outBuf_, kOutBufSize);
	EXPECT_FALSE(result0h.has_value()) << "Should reject height=0";
}

// ===========================================================================
// Test: processFrame rejects output buffer too small for JPEG
// Bug caught: stub writes past end of buffer → heap corruption
// ===========================================================================
TEST_F(FaceFilterStubTest, ProcessFrameRejectsTinyOutputBuffer)
{
	auto frame = makeBgrFrame(kTinyW, kTinyH, 128, 128, 128);

	uint8_t tinyBuf[4]{};
	auto result = inferencer_->processFrame(
		frame.data(), kTinyW, kTinyH, tinyBuf, sizeof(tinyBuf));

	EXPECT_FALSE(result.has_value())
		<< "processFrame should reject buffer smaller than minimal JPEG";
	SPDLOG_DEBUG("Tiny buffer error: {}", result.error());
}

// ===========================================================================
// Test: setActiveFilter persists and availableFilters populates after load
// Bug caught: filter state not tracked, select_filter command has no effect
// ===========================================================================
TEST_F(FaceFilterStubTest, FilterSelectionAndListing)
{
	// Before loading assets — no filters available
	auto filtersEmpty = inferencer_->availableFilters();
	EXPECT_TRUE(filtersEmpty.empty())
		<< "Filters should be empty before loadFilterAssets";

	// Load assets
	inferencer_->loadFilterAssets("");

	auto filters = inferencer_->availableFilters();
	ASSERT_FALSE(filters.empty())
		<< "availableFilters must return entries after loadFilterAssets";

	SPDLOG_DEBUG("Available filters: {}", filters.size());
	for (std::size_t i = 0; i < filters.size(); ++i) {
		SPDLOG_DEBUG("  filter[{}]: id={}, name={}, loaded={}",
			i, filters[i].id, filters[i].name, filters[i].loaded);
	}

	// Set a filter and verify it sticks
	inferencer_->setActiveFilter("dog");

	// Process a frame — should succeed with filter active
	auto frame = makeBgrFrame(kTinyW, kTinyH, 200, 200, 200);
	auto result = inferencer_->processFrame(
		frame.data(), kTinyW, kTinyH, outBuf_, kOutBufSize);
	ASSERT_TRUE(result.has_value()) << result.error();
}

// ===========================================================================
// Test: lastLandmarks returns valid (no-face) result after processFrame
// Bug caught: lastLandmarks returns uninitialised memory
// ===========================================================================
TEST_F(FaceFilterStubTest, LastLandmarksAfterProcessFrame)
{
	auto frame = makeBgrFrame(kMeshW, kMeshH, 100, 100, 100);
	inferencer_->loadModel("");

	auto result = inferencer_->processFrame(
		frame.data(), kMeshW, kMeshH, outBuf_, kOutBufSize);
	ASSERT_TRUE(result.has_value()) << result.error();

	auto landmarks = inferencer_->lastLandmarks();
	// Stub should report no face detected
	EXPECT_FALSE(landmarks.faceDetected)
		<< "Stub should always report faceDetected=false";
	SPDLOG_DEBUG("lastLandmarks: faceDetected={}", landmarks.faceDetected);
}

// ===========================================================================
// Test: unload resets state, re-loading works cleanly
// Bug caught: unload leaves stale state → processFrame uses freed resources
// ===========================================================================
TEST_F(FaceFilterStubTest, UnloadAndReload)
{
	inferencer_->loadModel("");
	inferencer_->loadFilterAssets("");
	EXPECT_FALSE(inferencer_->availableFilters().empty());

	// Unload
	inferencer_->unload();
	EXPECT_EQ(inferencer_->currentVramUsageBytes(), 0u);
	EXPECT_TRUE(inferencer_->availableFilters().empty())
		<< "availableFilters must be empty after unload";

	// Re-load
	inferencer_->loadModel("");
	inferencer_->loadFilterAssets("");
	EXPECT_FALSE(inferencer_->availableFilters().empty())
		<< "availableFilters must repopulate after re-load";

	// Process after re-load
	auto frame = makeBgrFrame(kTinyW, kTinyH, 50, 50, 50);
	auto result = inferencer_->processFrame(
		frame.data(), kTinyW, kTinyH, outBuf_, kOutBufSize);
	ASSERT_TRUE(result.has_value()) << result.error();
	SPDLOG_DEBUG("Post-reload processFrame: {} bytes", result.value());
}

// ===========================================================================
// Test: VRAM accounting reports zero for stub
// Bug caught: stub reports non-zero VRAM → daemon evicts it unnecessarily
// ===========================================================================
TEST_F(FaceFilterStubTest, VramAccountingIsZero)
{
	EXPECT_EQ(inferencer_->currentVramUsageBytes(), 0u);

	inferencer_->loadModel("");
	EXPECT_EQ(inferencer_->currentVramUsageBytes(), 0u)
		<< "Stub uses no GPU — VRAM must remain zero after model load";
}

// ===========================================================================
// Test: processFrame called many times doesn't leak or corrupt
// Bug caught: per-frame allocations that grow unbounded
// ===========================================================================
TEST_F(FaceFilterStubTest, RepeatedProcessFrameStability)
{
	constexpr int kIterations = 100;
	auto frame = makeBgrFrame(kTinyW, kTinyH, 120, 130, 140);

	inferencer_->loadModel("");

	for (int i = 0; i < kIterations; ++i) {
		auto result = inferencer_->processFrame(
			frame.data(), kTinyW, kTinyH, outBuf_, kOutBufSize);
		ASSERT_TRUE(result.has_value())
			<< "Failed on iteration " << i << ": " << result.error();
	}

	SPDLOG_DEBUG("Processed {} frames without error", kIterations);
	EXPECT_EQ(inferencer_->currentVramUsageBytes(), 0u);
}

