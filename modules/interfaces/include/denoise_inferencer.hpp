#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — DenoiseInferencer: pure virtual video denoising interface
//
// All video denoising inferencers implement this interface, enabling mock
// injection for tests and inferencer swapping with a single YAML field change.
//
// Implementations:
//   OnnxBasicVsrppInferencer — ONNX Runtime + CUDA EP (production)
//   MockDenoiseInferencer    — returns pass-through JPEG (unit tests)
// ---------------------------------------------------------------------------


/**
 * @brief Pure interface for temporal video denoising inferencers.
 *
 * The inferencer receives a sliding window of N consecutive BGR24 frames
 * and returns the denoised center frame encoded as JPEG.
 *
 * Thread safety: implementations are NOT thread-safe.
 *  Callers must serialize access.
 */
class DenoiseInferencer {
public:
	virtual ~DenoiseInferencer() = default;

	/**
	 * @brief Load ONNX model and initialise inference session.
	 *
	 * Must be called once before processFrames().
	 *
	 * @param onnxModelPath  Path to the BasicVSR++ ONNX model file.
	 * @return               Void on success, error string on failure.
	 */
	[[nodiscard]] virtual tl::expected<void, std::string> loadModel(
		const std::string& onnxModelPath) = 0;

	/**
	 * @brief Denoise a temporal window of BGR24 frames.
	 *
	 * Performs temporal denoising across the provided frame sequence.
	 * The denoised center frame is encoded as JPEG and written to outJpegBuf.
	 *
	 * @param bgrFrames      Ordered list of BGR24 frame pointers (temporal window).
	 * @param frameCount     Number of frames in the window.
	 * @param width          Frame width in pixels.
	 * @param height         Frame height in pixels.
	 * @param outJpegBuf     Output buffer for JPEG-encoded denoised frame.
	 * @param maxJpegBytes   Size of outJpegBuf in bytes.
	 * @return               JPEG byte count on success, or error string.
	 */
	[[nodiscard]] virtual tl::expected<std::size_t, std::string> processFrames(
		const uint8_t* const* bgrFrames,
		uint32_t frameCount,
		uint32_t width,
		uint32_t height,
		uint8_t* outJpegBuf,
		std::size_t maxJpegBytes) = 0;

	/**
	 * @brief Unload model weights and free GPU memory.
	 *
	 * Safe to call even if loadModel() was never called.
	 */
	virtual void unloadModel() = 0;

	/**
	 * @brief Current VRAM usage in bytes.
	 *
	 * Used by OmniEdgeDaemon for eviction accounting.
	 * Returns 0 if the model is not loaded.
	 */
	[[nodiscard]] virtual std::size_t currentVramUsageBytes() const noexcept = 0;

	/**
	 * @brief Inferencer identifier string matching the YAML `inferencer:` field.
	 *
	 * e.g. "onnx-basicvsrpp"
	 */
	[[nodiscard]] virtual std::string name() const = 0;

	/**
	 * @brief Return the CUDA stream used by this inferencer for GPU work.
	 *
	 * Used by CudaFence to record completion on the correct stream.
	 * Default returns nullptr (default stream) for CPU-only stubs.
	 */
	[[nodiscard]] virtual cudaStream_t cudaStream() const noexcept { return nullptr; }
};

// ---------------------------------------------------------------------------
// Factory functions — one per concrete inferencer
// ---------------------------------------------------------------------------

/** @brief Production inferencer: ONNX Runtime with CUDA Execution Provider. */
[[nodiscard]] std::unique_ptr<DenoiseInferencer> createOnnxBasicVsrppInferencer();

