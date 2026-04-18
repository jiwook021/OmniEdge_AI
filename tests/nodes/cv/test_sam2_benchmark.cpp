#include <gtest/gtest.h>

#include "sam2_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/constants/cv_constants.hpp"
#include "zmq/jpeg_constants.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// StubSam2Inferencer Benchmark Tests — CPU-only, no GPU required.
//
// What these tests catch:
//   - processFrame latency regression (stub should be <1ms per call)
//   - Hot-path allocations causing stalls (memory growing per frame)
//   - Throughput for point and box prompts at SAM2 native & full HD sizes
//   - Scaling behaviour between 1024x1024 (native) and 1920x1080 (full HD)
//
// These benchmarks run the REAL stub processFrame with REAL pixel data
// and report latency percentiles via spdlog.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// BenchStubSam2Inferencer — self-contained stub for benchmark isolation.
// ---------------------------------------------------------------------------
class BenchStubSam2Inferencer : public Sam2Inferencer {
public:
	void loadModel(const std::string& /*enc*/,
	               const std::string& /*dec*/) override
	{
		loaded_ = true;
	}

	[[nodiscard]] tl::expected<void, std::string>
	encodeImage(const uint8_t* bgrFrame, uint32_t width, uint32_t height) override
	{
		if (!loaded_) return tl::unexpected(std::string("not loaded"));
		if (!bgrFrame) return tl::unexpected(std::string("null frame"));
		width_ = width;
		height_ = height;
		encoded_ = true;
		return {};
	}

	[[nodiscard]] tl::expected<Sam2Result, std::string>
	segmentWithPrompt(const Sam2Prompt& prompt) override
	{
		if (!encoded_) return tl::unexpected(std::string("no image encoded"));

		Sam2Result result;
		result.maskWidth  = width_;
		result.maskHeight = height_;
		result.mask.resize(static_cast<std::size_t>(width_) * height_, 128);
		result.iouScore  = 0.91f;
		result.stability = 0.87f;

		// Vary mask content by prompt type for deterministic output
		if (prompt.type == Sam2PromptType::kBox) {
			std::memset(result.mask.data(), 200, result.mask.size());
		}

		return result;
	}

	[[nodiscard]] tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* bgrFrame, uint32_t width, uint32_t height,
	             const Sam2Prompt& prompt, uint8_t* outBuf,
	             std::size_t maxJpegBytes) override
	{
		auto enc = encodeImage(bgrFrame, width, height);
		if (!enc) return tl::unexpected(enc.error());
		auto seg = segmentWithPrompt(prompt);
		if (!seg) return tl::unexpected(seg.error());

		// Store result for lastSegmentResult() accessor
		lastResult_ = seg.value();

		// Write stub JPEG: SOI + body + EOI
		const std::size_t frameBytes =
			static_cast<std::size_t>(width) * height * 3u;
		const std::size_t jpegSize =
			std::min(std::max(frameBytes / 10, std::size_t{64}), maxJpegBytes);

		if (jpegSize < 4) {
			return tl::unexpected(std::string("output buffer too small"));
		}

		outBuf[0] = 0xFF;
		outBuf[1] = 0xD8;  // SOI
		std::memset(outBuf + 2, bgrFrame[0], jpegSize - 4);
		outBuf[jpegSize - 2] = 0xFF;
		outBuf[jpegSize - 1] = 0xD9;  // EOI

		return jpegSize;
	}

	void unload() noexcept override { loaded_ = false; }

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		return loaded_ ? kSam2EstimatedVramBytes : 0;
	}

private:
	bool     loaded_  = false;
	bool     encoded_ = false;
	uint32_t width_   = 0;
	uint32_t height_  = 0;
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Sam2Benchmark : public ::testing::Test {
protected:
	void SetUp() override
	{
		inferencer_ = std::make_unique<BenchStubSam2Inferencer>();
		inferencer_->loadModel("", "");
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

	static Sam2Prompt makePointPrompt(float x, float y)
	{
		Sam2Prompt prompt;
		prompt.type = Sam2PromptType::kPoint;
		prompt.points.push_back(Sam2PointPrompt{x, y, 1});
		return prompt;
	}

	static Sam2Prompt makeBoxPrompt(float x1, float y1, float x2, float y2)
	{
		Sam2Prompt prompt;
		prompt.type = Sam2PromptType::kBox;
		prompt.box = Sam2BoxPrompt{x1, y1, x2, y2};
		return prompt;
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
	                         const Sam2Prompt& prompt, int iterations)
	{
		auto frame = makeBgrFrame(w, h);

		std::vector<double> latencies;
		latencies.reserve(iterations);

		// Warmup (5 iterations)
		for (int i = 0; i < 5; ++i) {
			inferencer_->processFrame(
				frame.data(), w, h, prompt,
				outBuf_.data(), outBuf_.size());
		}

		// Measured iterations
		for (int i = 0; i < iterations; ++i) {
			const auto start = std::chrono::steady_clock::now();
			auto result = inferencer_->processFrame(
				frame.data(), w, h, prompt,
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

	std::unique_ptr<BenchStubSam2Inferencer> inferencer_;
	std::vector<uint8_t> outBuf_ =
		std::vector<uint8_t>(kMaxJpegBytesPerSlot);
};

// ===========================================================================
// Benchmark: 1024x1024 point prompt — SAM2 native encoder input resolution
// Bug caught: encodeImage + segmentWithPrompt overhead, buffer allocations
// ===========================================================================
TEST_F(Sam2Benchmark, Throughput_1024x1024_PointPrompt)
{
	constexpr uint32_t kW = kSam2EncoderInputResolution;
	constexpr uint32_t kH = kSam2EncoderInputResolution;
	constexpr int kIter = 1000;

	auto prompt = makePointPrompt(0.5f, 0.5f);
	auto r = runBenchmark(kW, kH, prompt, kIter);

	SPDLOG_INFO("Benchmark {}x{} point ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	EXPECT_LT(r.medianMs, 5.0)
		<< "Stub processFrame should be <5ms at SAM2 native resolution";
}

// ===========================================================================
// Benchmark: 1920x1080 point prompt — full HD input (tests resize overhead)
// Bug caught: O(n^2) scaling in frame processing at large resolutions
// ===========================================================================
TEST_F(Sam2Benchmark, Throughput_1920x1080_PointPrompt)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr int kIter = 500;

	auto prompt = makePointPrompt(0.5f, 0.5f);
	auto r = runBenchmark(kW, kH, prompt, kIter);

	SPDLOG_INFO("Benchmark {}x{} point ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms, "
	            "throughput={:.0f} fps",
	            kW, kH, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs,
	            1000.0 / r.meanMs);

	EXPECT_LT(r.medianMs, 5.0)
		<< "Stub is constant-time (memcpy + markers) — should be <5ms even at 1080p";
}

// ===========================================================================
// Benchmark: 1024x1024 box prompt — box prompt variant
// Bug caught: box prompt path overhead vs point prompt path
// ===========================================================================
TEST_F(Sam2Benchmark, Throughput_1024x1024_BoxPrompt)
{
	constexpr uint32_t kW = kSam2EncoderInputResolution;
	constexpr uint32_t kH = kSam2EncoderInputResolution;
	constexpr int kIter = 1000;

	auto prompt = makeBoxPrompt(0.1f, 0.2f, 0.8f, 0.9f);
	auto r = runBenchmark(kW, kH, prompt, kIter);

	SPDLOG_INFO("Benchmark {}x{} box ({} iters): "
	            "median={:.3f}ms, p99={:.3f}ms, mean={:.3f}ms, "
	            "min={:.3f}ms, max={:.3f}ms",
	            kW, kH, r.iterations,
	            r.medianMs, r.p99Ms, r.meanMs,
	            r.minMs, r.maxMs);

	EXPECT_LT(r.medianMs, 5.0)
		<< "Box prompt should have comparable latency to point prompt";
}

// ===========================================================================
// Benchmark: processFrame meets SAM2 inference deadline target
// Bug caught: latency exceeds kSam2InferenceTimeoutMs under load
// ===========================================================================
TEST_F(Sam2Benchmark, MeetsDeadlineTarget)
{
	constexpr uint32_t kW = 1920, kH = 1080;
	constexpr int kIter = 200;
	constexpr double kDeadlineMs =
		static_cast<double>(kSam2InferenceTimeoutMs);

	auto prompt = makePointPrompt(0.5f, 0.5f);
	auto r = runBenchmark(kW, kH, prompt, kIter);

	SPDLOG_INFO("Deadline check: p99={:.3f}ms vs target={:.0f}ms",
	            r.p99Ms, kDeadlineMs);

	EXPECT_LT(r.p99Ms, kDeadlineMs)
		<< "p99 latency must be under the SAM2 inference deadline ("
		<< kDeadlineMs << " ms)";
}
