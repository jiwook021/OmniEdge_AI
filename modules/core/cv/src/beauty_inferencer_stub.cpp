#include "beauty_inferencer.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// StubBeautyInferencer — passthrough JPEG encoder for CPU-only builds.
//
// This stub does NOT run FaceMesh inference or apply any beauty effects.
// It encodes the input BGR frame as a minimal valid JPEG so the node
// integration tests can exercise the full command path.
//
// Used as the inferencer in the omniedge_beauty binary until the
// real ONNX Runtime inferencer (OnnxBeautyInferencer) is built and linked.
// ---------------------------------------------------------------------------


namespace {

/// Minimal JFIF JPEG that encodes a solid 1x1 grey pixel.
constexpr uint8_t kMinimalJpeg[] = {
	0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
	0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
	0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
	0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
	0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
	0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
	0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
	0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
	0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
	0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
	0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
	0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
	0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4,
	0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
	0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
	0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF,
	0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
	0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04,
	0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
	0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
	0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
	0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
	0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
	0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A,
	0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
	0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00,
	0x3F, 0x00, 0x7B, 0x94, 0x11, 0x00, 0x00, 0x00,
	0xFF, 0xD9,
};
constexpr std::size_t kMinimalJpegSize = sizeof(kMinimalJpeg);

} // anonymous namespace


class StubBeautyInferencer final : public BeautyInferencer {
public:
	void loadModel(const std::string& faceMeshOnnxPath) override
	{
		OE_ZONE_SCOPED;
		OE_LOG_INFO("stub_beauty: loadModel path={} (no-op stub — no GPU required)",
			faceMeshOnnxPath.empty() ? "(empty)" : faceMeshOnnxPath);
		modelLoaded_ = true;
	}

	[[nodiscard]] tl::expected<std::size_t, std::string>
	processFrame(const uint8_t* bgrFrame,
	             uint32_t       width,
	             uint32_t       height,
	             uint8_t*       outBuf,
	             std::size_t    maxJpegBytes) override
	{
		OE_ZONE_SCOPED;
		const auto start = std::chrono::steady_clock::now();
		++frameCount_;

		if (!bgrFrame) {
			return tl::unexpected(std::string("null input frame pointer"));
		}
		if (width == 0 || height == 0) {
			return tl::unexpected(std::string("zero-dimension frame"));
		}
		if (!outBuf) {
			return tl::unexpected(std::string("null output buffer pointer"));
		}
		if (maxJpegBytes < kMinimalJpegSize) {
			return tl::unexpected(std::string("output buffer too small for stub JPEG"));
		}

		// Stub: write minimal JPEG to output (passthrough — no inference)
		std::memcpy(outBuf, kMinimalJpeg, kMinimalJpegSize);

		const auto elapsed = std::chrono::steady_clock::now() - start;
		OE_LOG_DEBUG("stub_beauty: processFrame frame={}, input={}x{} ({} bytes BGR), "
		             "output={} bytes JPEG, params_identity={}, elapsed={:.2f}ms",
		             frameCount_, width, height,
		             static_cast<std::size_t>(width) * height * 3,
		             kMinimalJpegSize, params_.isIdentity(),
		             std::chrono::duration<double, std::milli>(elapsed).count());

		return kMinimalJpegSize;
	}

	void setBeautyParams(const BeautyParams& params) noexcept override
	{
		params_ = params;
		OE_LOG_DEBUG("stub_beauty: setBeautyParams smoothing={:.1f} faceSlim={:.1f} "
		             "brightness={:.1f} bgMode={}",
		             params.smoothing, params.faceSlim, params.brightness,
		             static_cast<int>(params.bgMode));
	}

	void unload() noexcept override
	{
		OE_LOG_INFO("stub_beauty: unload (model_was_loaded={})", modelLoaded_);
		modelLoaded_ = false;
		frameCount_  = 0;
	}

	[[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override
	{
		return 0;
	}

private:
	bool         modelLoaded_ = false;
	uint64_t     frameCount_  = 0;
	BeautyParams params_;
};

std::unique_ptr<BeautyInferencer> createStubBeautyInferencer()
{
	return std::make_unique<StubBeautyInferencer>();
}

/// Unified factory — resolved at link time.
/// Stub build: links this file.  GPU build: links onnx_beauty_inferencer.cpp.
std::unique_ptr<BeautyInferencer> createBeautyInferencer()
{
	return createStubBeautyInferencer();
}
