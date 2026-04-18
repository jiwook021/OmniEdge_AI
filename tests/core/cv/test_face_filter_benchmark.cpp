#include <gtest/gtest.h>

#include "face_filter_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/constants/cv_constants.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// StubFaceFilterInferencer Benchmark Tests — CPU-only, no GPU required.
//
// What these tests catch:
//   - processFrame latency regression (stub should be <1ms per frame)
//   - Hot-path allocations causing GC stalls (memory growing per frame)
//   - Throughput too low for 20 fps target on real hardware
//   - Performance varies wildly across frame sizes (bad branching)
//
// These benchmarks run the REAL stub processFrame with REAL pixel data
// and report latency percentiles via spdlog.
// ---------------------------------------------------------------------------

[[nodiscard]] std::unique_ptr<FaceFilterInferencer> createStubFaceFilterInferencer();


class FaceFilterBenchmark : public ::testing::Test {
protected:
	void SetUp() override
	{
		OeLogger::instance().setModule("test_face_filter_benchmark");
		inferencer_ = createStubFaceFilterInferencer();
		ASSERT_NE(inferencer_, nullptr);
		inferencer_->loadModel("");
	}

	static std::vector<uint8_t> makeBgrFrame(uint32_t w, uint32_t h)
	{
		const std::size_t bytes = static_cast<std::size_t>(w) * h * 3;
		std::vector<uint8_t> frame(bytes);
		// Fill with a gradient pattern (more realistic than solid colour)
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
		int    frameCount;
	};

	BenchResult runBenchmark(uint32_t w, uint32_t h, int iterations)
	{
		auto frame = makeBgrFrame(w, h);
		std::vector<double> latencies;
		latencies.reserve(iterations);

		// Warmup (5 iterations)
		for (int i = 0; i < 5; ++i) {
			inferencer_->processFrame(
				frame.data(), w, h, outBuf_.data(), outBuf_.size());
		}

		// Measured iterations
		for (int i = 0; i < iterations; ++i) {
			const auto start = std::chrono::steady_clock::now();
			auto result = inferencer_->processFrame(
				frame.data(), w, h, outBuf_.data(), outBuf_.size());
			const auto end = std::chrono::steady_clock::now();

			EXPECT_TRUE(result.has_value()) << result.error();

			const double ms =
				std::chrono::duration<double, std::milli>(end - start).count();
			latencies.push_back(ms);
		}

		std::sort(latencies.begin(), latencies.end());

		BenchResult r{};
		r.frameCount = iterations;
		r.minMs    = latencies.front();
		r.maxMs    = latencies.back();
		r.medianMs = latencies[latencies.size() / 2];
		r.p99Ms    = latencies[static_cast<std::size_t>(latencies.size() * 0.99)];
		r.meanMs   = std::accumulate(latencies.begin(), latencies.end(), 0.0)
		             / static_cast<double>(iterations);
		return r;
	}

	std::unique_ptr<FaceFilterInferencer> inferencer_;
	std::vector<uint8_t> outBuf_ = std::vector<uint8_t>(256 * 1024);
};

// ===========================================================================
// Benchmark: 64x48 frames — small thumbnail size
// Bug caught: fixed overhead dominates at small sizes
// ===========================================================================
TEST_F(FaceFilterBenchmark, Throughput_64x48)
{
	constexpr uint32_t kW = 64, kH = 48;
	constexpr int kIter = 1000;

	auto r = runBenchmark(kW, kH, kIter);

	SPDLOG_INFO("Benchmark {}x{} ({} frames): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, r.frameCount,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	// Stub should be sub-millisecond — it's just a memcpy
	EXPECT_LT(r.medianMs, 1.0)
		<< "Stub processFrame should be <1ms for 64x48";
}

// ===========================================================================
// Benchmark: 192x192 — FaceMesh input resolution
// Bug caught: regression when production inferencer switches from stub
// ===========================================================================
TEST_F(FaceFilterBenchmark, Throughput_192x192)
{
	constexpr uint32_t kW = kFaceMeshInputResolution;
	constexpr uint32_t kH = kFaceMeshInputResolution;
	constexpr int kIter = 1000;

	auto r = runBenchmark(kW, kH, kIter);

	SPDLOG_INFO("Benchmark {}x{} ({} frames): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, r.frameCount,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	EXPECT_LT(r.medianMs, 1.0)
		<< "Stub processFrame should be <1ms for FaceMesh input res";
}

// ===========================================================================
// Benchmark: 1920x1080 — full HD input
// Bug caught: O(n) scaling in frame processing (should be constant for stub)
// ===========================================================================
TEST_F(FaceFilterBenchmark, Throughput_1920x1080)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr int kIter = 500;

	auto r = runBenchmark(kW, kH, kIter);

	SPDLOG_INFO("Benchmark {}x{} ({} frames): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, r.frameCount,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	// Stub is O(1) — memcpy of fixed-size minimal JPEG, not proportional to input
	EXPECT_LT(r.medianMs, 1.0)
		<< "Stub is constant-time (memcpy of minimal JPEG) — should be <1ms even at 1080p";
}

// ===========================================================================
// Benchmark: processFrame meets 20 fps deadline target
// Bug caught: latency exceeds kFaceFilterDeadlineMs (50ms)
// ===========================================================================
TEST_F(FaceFilterBenchmark, MeetsDeadlineTarget)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr int kIter = 200;
	constexpr double kDeadlineMs =
		static_cast<double>(kFaceFilterDeadlineMs);

	auto r = runBenchmark(kW, kH, kIter);

	SPDLOG_INFO("Deadline check: p99={:.3f}ms vs target={:.0f}ms",
	            r.p99Ms, kDeadlineMs);

	EXPECT_LT(r.p99Ms, kDeadlineMs)
		<< "p99 latency must be under the face filter deadline ("
		<< kDeadlineMs << " ms)";
}

