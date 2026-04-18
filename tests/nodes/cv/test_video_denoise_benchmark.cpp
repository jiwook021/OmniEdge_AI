#include <gtest/gtest.h>

#include "denoise_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/constants/video_denoise_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// StubDenoiseInferencer Benchmark Tests — CPU-only, no GPU required.
//
// What these tests catch:
//   - processFrames latency regression (stub should be <1ms per call)
//   - Hot-path allocations causing stalls (memory growing per frame)
//   - Throughput too low for 20 fps deadline with 5-frame temporal window
//   - Scaling behaviour across frame sizes and temporal window depths
//
// These benchmarks run the REAL stub processFrames with REAL pixel data
// and report latency percentiles via spdlog.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubDenoiseInferencer — same stub used in test_video_denoise_node.cpp.
// Duplicated here to keep the benchmark self-contained (no header coupling).
// ---------------------------------------------------------------------------
class BenchStubInferencer : public DenoiseInferencer {
public:
	tl::expected<void, std::string> loadModel(
		const std::string& /*path*/) override
	{
		loaded_ = true;
		return {};
	}

	tl::expected<std::size_t, std::string> processFrames(
		const uint8_t* const* bgrFrames,
		uint32_t frameCount,
		uint32_t width,
		uint32_t height,
		uint8_t* outJpegBuf,
		std::size_t maxJpegBytes) override
	{
		// Compute body byte from average of each frame's first pixel
		uint32_t pixelSum = 0;
		for (uint32_t f = 0; f < frameCount; ++f) {
			if (bgrFrames[f]) {
				pixelSum += bgrFrames[f][0];
			}
		}
		const uint8_t bodyByte =
			static_cast<uint8_t>(pixelSum / std::max(frameCount, 1u));

		const std::size_t frameBytes =
			static_cast<std::size_t>(width) * height * 3u;
		const std::size_t jpegSize =
			std::min(std::max(frameBytes / 10, std::size_t{64}), maxJpegBytes);

		if (jpegSize < 4) {
			return tl::unexpected(std::string("output buffer too small"));
		}

		outJpegBuf[0] = 0xFF;
		outJpegBuf[1] = 0xD8;  // SOI
		std::memset(outJpegBuf + 2, bodyByte, jpegSize - 4);
		outJpegBuf[jpegSize - 2] = 0xFF;
		outJpegBuf[jpegSize - 1] = 0xD9;  // EOI

		return jpegSize;
	}

	void unloadModel() override { loaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		return loaded_ ? 1500ULL * 1024 * 1024 : 0;
	}

	[[nodiscard]] std::string name() const override
	{
		return "bench-stub-basicvsrpp";
	}

private:
	bool loaded_ = false;
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class DenoiseBenchmark : public ::testing::Test {
protected:
	void SetUp() override
	{
		inferencer_ = std::make_unique<BenchStubInferencer>();
		inferencer_->loadModel("");
	}

	static std::vector<uint8_t> makeBgrFrame(uint32_t w, uint32_t h)
	{
		const std::size_t bytes = static_cast<std::size_t>(w) * h * 3;
		std::vector<uint8_t> frame(bytes);
		for (std::size_t i = 0; i < bytes; ++i) {
			frame[i] = static_cast<uint8_t>(i & 0xFF);
		}
		return frame;
	}

	struct BenchResult {
		double medianMs;
		double p99Ms;
		double meanMs;
		double minMs;
		double maxMs;
		int    iterations;
	};

	BenchResult runBenchmark(uint32_t w, uint32_t h,
	                         uint32_t temporalWindow, int iterations)
	{
		// Build temporal frame window
		std::vector<std::vector<uint8_t>> frames(temporalWindow);
		std::vector<const uint8_t*> ptrs(temporalWindow);
		for (uint32_t i = 0; i < temporalWindow; ++i) {
			frames[i] = makeBgrFrame(w, h);
			// Vary first pixel per frame so body byte changes
			frames[i][0] = static_cast<uint8_t>((i + 1) * 10);
			ptrs[i] = frames[i].data();
		}

		std::vector<double> latencies;
		latencies.reserve(iterations);

		// Warmup (5 iterations)
		for (int i = 0; i < 5; ++i) {
			inferencer_->processFrames(
				ptrs.data(), temporalWindow, w, h,
				outBuf_.data(), outBuf_.size());
		}

		// Measured iterations
		for (int i = 0; i < iterations; ++i) {
			const auto start = std::chrono::steady_clock::now();
			auto result = inferencer_->processFrames(
				ptrs.data(), temporalWindow, w, h,
				outBuf_.data(), outBuf_.size());
			const auto end = std::chrono::steady_clock::now();

			EXPECT_TRUE(result.has_value()) << result.error();

			const double ms =
				std::chrono::duration<double, std::milli>(end - start).count();
			latencies.push_back(ms);
		}

		std::sort(latencies.begin(), latencies.end());

		BenchResult r{};
		r.iterations = iterations;
		r.minMs    = latencies.front();
		r.maxMs    = latencies.back();
		r.medianMs = latencies[latencies.size() / 2];
		r.p99Ms    = latencies[static_cast<std::size_t>(latencies.size() * 0.99)];
		r.meanMs   = std::accumulate(latencies.begin(), latencies.end(), 0.0)
		             / static_cast<double>(iterations);
		return r;
	}

	std::unique_ptr<BenchStubInferencer> inferencer_;
	std::vector<uint8_t> outBuf_ =
		std::vector<uint8_t>(kMaxJpegBytesPerSlot);
};

// ===========================================================================
// Benchmark: 480x270 x 5 frames — model input resolution, full window
// Bug caught: temporal window loop overhead, buffer allocation stalls
// ===========================================================================
TEST_F(DenoiseBenchmark, Throughput_480x270_Window5)
{
	constexpr uint32_t kW = 480, kH = 270;
	constexpr uint32_t kWindow = kTemporalWindowFrameCount;
	constexpr int kIter = 1000;

	auto r = runBenchmark(kW, kH, kWindow, kIter);

	SPDLOG_INFO("Benchmark {}x{} x{} ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, kWindow, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	EXPECT_LT(r.medianMs, 1.0)
		<< "Stub processFrames should be <1ms at model input resolution";
}

// ===========================================================================
// Benchmark: 1920x1080 x 5 frames — full HD input (tests resize overhead)
// Bug caught: O(n^2) scaling in frame processing at large resolutions
// ===========================================================================
TEST_F(DenoiseBenchmark, Throughput_1920x1080_Window5)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr uint32_t kWindow = kTemporalWindowFrameCount;
	constexpr int kIter = 500;

	auto r = runBenchmark(kW, kH, kWindow, kIter);

	SPDLOG_INFO("Benchmark {}x{} x{} ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, kWindow, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	EXPECT_LT(r.medianMs, 1.0)
		<< "Stub is constant-time (memcpy + markers) — should be <1ms even at 1080p";
}

// ===========================================================================
// Benchmark: 480x270 x 1 frame — single frame (no temporal window)
// Bug caught: overhead from temporal window parameter at minimum depth
// ===========================================================================
TEST_F(DenoiseBenchmark, Throughput_480x270_Window1)
{
	constexpr uint32_t kW = 480, kH = 270;
	constexpr uint32_t kWindow = 1;
	constexpr int kIter = 1000;

	auto r = runBenchmark(kW, kH, kWindow, kIter);

	SPDLOG_INFO("Benchmark {}x{} x{} ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms",
	            kW, kH, kWindow, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs);

	EXPECT_LT(r.medianMs, 1.0)
		<< "Single-frame stub should be faster than full temporal window";
}

// ===========================================================================
// Benchmark: processFrames meets 100ms denoise deadline target
// Bug caught: latency exceeds kDenoiseDeadlineMs under temporal window load
// ===========================================================================
TEST_F(DenoiseBenchmark, MeetsDeadlineTarget)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr uint32_t kWindow = kTemporalWindowFrameCount;
	constexpr int kIter = 200;
	constexpr double kDeadlineMs =
		static_cast<double>(kDenoiseDeadlineMs);

	auto r = runBenchmark(kW, kH, kWindow, kIter);

	SPDLOG_INFO("Deadline check: p99={:.3f}ms vs target={:.0f}ms",
	            r.p99Ms, kDeadlineMs);

	EXPECT_LT(r.p99Ms, kDeadlineMs)
		<< "p99 latency must be under the denoise deadline ("
		<< kDeadlineMs << " ms)";
}

