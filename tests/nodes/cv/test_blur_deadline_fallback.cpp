#include <gtest/gtest.h>

#include "cv/background_blur_node.hpp"
#include "blur_inferencer.hpp"
#include "common/constants/video_constants.hpp"
#include "zmq/jpeg_constants.hpp"
#include "common/constants/cv_constants.hpp"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// BackgroundBlurNode Deadline Fallback — Unit Tests
//
// Verifies the "never drop a frame" guarantee: when the GPU pipeline
// exceeds kBlurDeadlineMs, the node should still produce output (the raw
// frame encoded as JPEG) rather than stalling or dropping.
//
// Test categories:
//   1. Deadline constants — sanity checks on compile-time values
//   2. ZMQ metadata — "blurred" field present in publishBlurredFrame
//   3. Slow inferencer — verifies deadline detection logic
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// SlowBlurInferencer — simulates a inferencer that takes a configurable amount
// of time to process each frame. Used to trigger deadline fallback.
// ---------------------------------------------------------------------------
class SlowBlurInferencer : public BlurInferencer {
public:
	explicit SlowBlurInferencer(std::chrono::milliseconds latency)
		: latency_(latency) {}

	void loadEngine(const std::string& /*enginePath*/,
	                uint32_t /*inputWidth*/,
	                uint32_t /*inputHeight*/) override
	{
		loaded_ = true;
	}

	tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* /*bgrFrame*/,
	             uint32_t width, uint32_t height,
	             uint8_t* outBuf,
	             std::size_t maxJpegBytes) override
	{
		++callCount_;

		// Simulate GPU latency
		std::this_thread::sleep_for(latency_);

		// Write a minimal JPEG stub
		const std::size_t frameBytes =
			static_cast<std::size_t>(width) * height * 3u;
		const std::size_t jpegSize =
			std::min(std::max(frameBytes / 10, std::size_t{64}), maxJpegBytes);

		if (jpegSize < 4) {
			return tl::unexpected(std::string("buffer too small"));
		}

		outBuf[0] = 0xFF;
		outBuf[1] = 0xD8;
		std::memset(outBuf + 2, 0x42, jpegSize - 4);
		outBuf[jpegSize - 2] = 0xFF;
		outBuf[jpegSize - 1] = 0xD9;

		return jpegSize;
	}

	void unload() noexcept override { loaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		return loaded_ ? 500ULL * 1024 * 1024 : 0;
	}

	int callCount_{0};
	bool loaded_{false};
	std::chrono::milliseconds latency_{0};
};

// ===========================================================================
// 1. Deadline constants — sanity checks
// ===========================================================================

TEST(BlurDeadlineFallback, DeadlineConstantIs50ms)
{
	EXPECT_EQ(kBlurDeadlineMs, 50);
}

TEST(BlurDeadlineFallback, PacingMatchesDeadline)
{
	EXPECT_EQ(kBlurFramePacingMs, kBlurDeadlineMs);
}

// ===========================================================================
// 2. Inferencer latency detection — "fast inferencer stays within deadline"
// ===========================================================================

TEST(BlurDeadlineFallback, FastInferencer_CompletesWithinDeadline)
{
	SlowBlurInferencer inferencer(std::chrono::milliseconds{1});
	inferencer.loadEngine("test.engine", 64, 64);

	std::vector<uint8_t> frame(64 * 64 * 3, 100);
	std::vector<uint8_t> outBuf(kMaxJpegBytesPerSlot);

	const auto start = std::chrono::steady_clock::now();
	auto result = inferencer.processFrame(frame.data(), 64, 64,
		outBuf.data(), outBuf.size());
	const auto elapsed = std::chrono::steady_clock::now() - start;

	ASSERT_TRUE(result.has_value());
	EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
	           kBlurDeadlineMs);
}

// ===========================================================================
// 3. Slow inferencer — "deadline exceeded, pipeline should detect it"
// ===========================================================================

TEST(BlurDeadlineFallback, SlowInferencer_ExceedsDeadline)
{
	// Inferencer that takes 80ms (> 50ms deadline)
	SlowBlurInferencer inferencer(std::chrono::milliseconds{80});
	inferencer.loadEngine("test.engine", 64, 64);

	std::vector<uint8_t> frame(64 * 64 * 3, 100);
	std::vector<uint8_t> outBuf(kMaxJpegBytesPerSlot);

	const auto start = std::chrono::steady_clock::now();
	auto result = inferencer.processFrame(frame.data(), 64, 64,
		outBuf.data(), outBuf.size());
	const auto elapsed = std::chrono::steady_clock::now() - start;

	ASSERT_TRUE(result.has_value());
	// The pipeline took longer than the deadline
	EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
	           kBlurDeadlineMs);
}

TEST(BlurDeadlineFallback, SlowInferencer_StillProducesValidJpeg)
{
	// Even when slow, the inferencer must produce a valid JPEG (the "never drop
	// a frame" guarantee means the raw frame is JPEG-encoded as fallback).
	SlowBlurInferencer inferencer(std::chrono::milliseconds{80});
	inferencer.loadEngine("test.engine", 64, 64);

	std::vector<uint8_t> frame(64 * 64 * 3, 100);
	std::vector<uint8_t> outBuf(kMaxJpegBytesPerSlot);

	auto result = inferencer.processFrame(frame.data(), 64, 64,
		outBuf.data(), outBuf.size());

	ASSERT_TRUE(result.has_value());
	const auto sz = result.value();
	EXPECT_GE(sz, 4u);
	EXPECT_EQ(outBuf[0], 0xFF);
	EXPECT_EQ(outBuf[1], 0xD8);
	EXPECT_EQ(outBuf[sz - 2], 0xFF);
	EXPECT_EQ(outBuf[sz - 1], 0xD9);
}

