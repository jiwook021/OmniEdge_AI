#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — ONNX Runtime IO Binding Utilities
//
// Provides GPU-resident memory info and pre-allocated GPU buffer management
// for zero-copy ONNX inference via Ort::IoBinding.
//
// Usage:
//   auto gpuMem = oe::onnx::gpuMemoryInfo();
//   auto binding = oe::onnx::GpuIoBinding(session, gpuMem);
//   binding.allocateInput(0, inputShape, inputBytes);
//   binding.allocateOutput(0, outputShape, outputBytes);
//   binding.run();
//   // Read back small outputs via cudaMemcpyAsync from binding.outputPtr(0)
// ---------------------------------------------------------------------------

#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include "common/oe_logger.hpp"
#include "gpu/oe_cuda_check.hpp"


namespace oe::onnx {

// ---------------------------------------------------------------------------
// gpuMemoryInfo — cached MemoryInfo for CUDA device 0
//
// Thread-safe: function-local static, initialized once.
// ---------------------------------------------------------------------------
[[nodiscard]] inline const Ort::MemoryInfo& gpuMemoryInfo()
{
	static const Ort::MemoryInfo info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
	return info;
}

// ---------------------------------------------------------------------------
// GpuBuffer — RAII wrapper for a single device allocation used by IO Binding.
// ---------------------------------------------------------------------------
struct GpuBuffer {
	void*       ptr   = nullptr;
	std::size_t bytes = 0;

	GpuBuffer() = default;

	explicit GpuBuffer(std::size_t numBytes) : bytes(numBytes)
	{
		if (numBytes > 0) {
			OE_CUDA_CHECK(cudaMalloc(&ptr, numBytes));
		}
	}

	~GpuBuffer() { free(); }

	GpuBuffer(const GpuBuffer&) = delete;
	GpuBuffer& operator=(const GpuBuffer&) = delete;

	GpuBuffer(GpuBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes)
	{
		o.ptr = nullptr;
		o.bytes = 0;
	}

	GpuBuffer& operator=(GpuBuffer&& o) noexcept
	{
		if (this != &o) {
			free();
			ptr = o.ptr;
			bytes = o.bytes;
			o.ptr = nullptr;
			o.bytes = 0;
		}
		return *this;
	}

	void free() noexcept
	{
		if (ptr) { cudaFree(ptr); ptr = nullptr; bytes = 0; }
	}
};

// ---------------------------------------------------------------------------
// GpuIoBinding — manages pre-allocated GPU buffers bound to an ONNX session.
//
// Allocates device memory for each input/output tensor at init time, binds
// them to an Ort::IoBinding, and provides typed pointers for CUDA kernel
// interop.  All inference runs via session.Run(runOpts, binding) with zero
// implicit H2D/D2H copies.
//
// Thread safety: none.  One instance per session per thread.
// ---------------------------------------------------------------------------
class GpuIoBinding {
public:
	GpuIoBinding() = default;

	/// Initialize binding for a session.  Call allocateInput/Output before run().
	explicit GpuIoBinding(Ort::Session& session)
		: session_(&session), binding_(session)
	{}

	GpuIoBinding(const GpuIoBinding&) = delete;
	GpuIoBinding& operator=(const GpuIoBinding&) = delete;
	GpuIoBinding(GpuIoBinding&&) = default;
	GpuIoBinding& operator=(GpuIoBinding&&) = default;

	/// Allocate a GPU input buffer and bind it.
	/// @param name    ONNX input tensor name (e.g. "input" or "images")
	/// @param shape   Tensor shape (e.g. {1, 3, 640, 640})
	/// @param elemSize  sizeof element (sizeof(float)=4, sizeof(__half)=2)
	void allocateInput(const char* name,
	                   const std::vector<int64_t>& shape,
	                   std::size_t elemSize)
	{
		std::size_t numElements = 1;
		for (auto d : shape) numElements *= static_cast<std::size_t>(d);
		const std::size_t bytes = numElements * elemSize;

		inputBuffers_.emplace_back(bytes);

		auto tensor = Ort::Value::CreateTensor(
			gpuMemoryInfo(),
			inputBuffers_.back().ptr, bytes,
			shape.data(), shape.size(),
			elemSize == sizeof(float) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
			                          : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

		binding_.BindInput(name, tensor);
		inputTensors_.push_back(std::move(tensor));
	}

	/// Allocate a GPU output buffer and bind it.
	void allocateOutput(const char* name,
	                    const std::vector<int64_t>& shape,
	                    std::size_t elemSize)
	{
		std::size_t numElements = 1;
		for (auto d : shape) numElements *= static_cast<std::size_t>(d);
		const std::size_t bytes = numElements * elemSize;

		outputBuffers_.emplace_back(bytes);

		auto tensor = Ort::Value::CreateTensor(
			gpuMemoryInfo(),
			outputBuffers_.back().ptr, bytes,
			shape.data(), shape.size(),
			elemSize == sizeof(float) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
			                          : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

		binding_.BindOutput(name, tensor);
		outputTensors_.push_back(std::move(tensor));
	}

	/// Bind an output to let ONNX Runtime allocate it (for dynamic shapes).
	void bindOutputToDevice(const char* name)
	{
		binding_.BindOutput(name, gpuMemoryInfo());
	}

	/// Run inference.  All data stays on GPU.
	void run()
	{
		session_->Run(Ort::RunOptions{nullptr}, binding_);
	}

	/// Get raw device pointer for input buffer at index.
	template <typename T = float>
	[[nodiscard]] T* inputPtr(std::size_t idx)
	{
		return static_cast<T*>(inputBuffers_.at(idx).ptr);
	}

	/// Get raw device pointer for output buffer at index.
	template <typename T = float>
	[[nodiscard]] T* outputPtr(std::size_t idx)
	{
		return static_cast<T*>(outputBuffers_.at(idx).ptr);
	}

	/// Get output bytes for buffer at index.
	[[nodiscard]] std::size_t outputBytes(std::size_t idx) const
	{
		return outputBuffers_.at(idx).bytes;
	}

	/// Get input bytes for buffer at index.
	[[nodiscard]] std::size_t inputBytes(std::size_t idx) const
	{
		return inputBuffers_.at(idx).bytes;
	}

	/// Check if binding is initialized.
	[[nodiscard]] bool valid() const noexcept { return session_ != nullptr; }

private:
	Ort::Session*            session_ = nullptr;
	Ort::IoBinding           binding_{nullptr};
	std::vector<GpuBuffer>   inputBuffers_;
	std::vector<GpuBuffer>   outputBuffers_;
	std::vector<Ort::Value>  inputTensors_;
	std::vector<Ort::Value>  outputTensors_;
};

} // namespace oe::onnx
