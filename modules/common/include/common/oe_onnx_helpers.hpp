#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Shared ONNX Runtime Utilities
//
// Eliminates duplicated session configuration, I/O name caching, and
// MemoryInfo creation across all ONNX Runtime inferencer classes.
// ---------------------------------------------------------------------------

#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "common/oe_logger.hpp"


namespace oe::onnx {

// ---------------------------------------------------------------------------
// configureSession — set up TRT EP (optional) + CUDA EP (required)
//
// Identical boilerplate previously duplicated in:
//   onnx_face_recog_inferencer.cpp, onnx_basicvsrpp_inferencer.cpp,
//   onnx_dtln_inferencer.cpp
//
// @param opts       Session options to configure (must be freshly constructed)
// @param label      Module name for log messages (e.g. "scrfd", "basicvsrpp")
// @param useTRT     If true, appends TensorRT EP before CUDA EP
// ---------------------------------------------------------------------------
inline void configureSession(Ort::SessionOptions& opts,
                             std::string_view label,
                             bool useTRT = true)
{
	opts.SetIntraOpNumThreads(1);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (useTRT) {
		OrtTensorRTProviderOptions trtOpts{};
		trtOpts.device_id = 0;
		trtOpts.trt_max_workspace_size = 1ULL << 30;  // 1 GB workspace
		trtOpts.trt_fp16_enable = 1;
		trtOpts.trt_engine_cache_enable = 1;
		trtOpts.trt_engine_cache_path = "/tmp/oe_trt_cache";

		try {
			opts.AppendExecutionProvider_TensorRT(trtOpts);
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("{}_trt_ep_failed: {}", label, ex.what());
		}
	}

	OrtCUDAProviderOptions cudaOpts{};
	cudaOpts.device_id = 0;
	cudaOpts.arena_extend_strategy = 0;
	cudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
	cudaOpts.do_copy_in_default_stream = 0;

	try {
		opts.AppendExecutionProvider_CUDA(cudaOpts);
	} catch (const Ort::Exception& ex) {
		throw std::runtime_error(std::format(
			"{} CUDA EP registration failed: {}", label, ex.what()));
	}
}

// ---------------------------------------------------------------------------
// configureCudaOnlySession — CUDA EP only (no TRT), for lightweight models
// ---------------------------------------------------------------------------
inline void configureCudaOnlySession(Ort::SessionOptions& opts,
                                     std::string_view label)
{
	configureSession(opts, label, /*useTRT=*/false);
}

// ---------------------------------------------------------------------------
// cacheIONames — extract and cache input/output tensor names from a session
//
// Previously duplicated with identical logic in all 3 inferencer files.
// ---------------------------------------------------------------------------
inline void cacheIONames(Ort::Session& session,
                         std::vector<std::string>& inNames,
                         std::vector<std::string>& outNames,
                         std::vector<const char*>& inPtrs,
                         std::vector<const char*>& outPtrs)
{
	Ort::AllocatorWithDefaultOptions allocator;

	inNames.clear();
	outNames.clear();

	for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
		auto name = session.GetInputNameAllocated(i, allocator);
		inNames.emplace_back(name.get());
	}
	for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
		auto name = session.GetOutputNameAllocated(i, allocator);
		outNames.emplace_back(name.get());
	}

	inPtrs.clear();
	outPtrs.clear();
	for (const auto& n : inNames)  inPtrs.push_back(n.c_str());
	for (const auto& n : outNames) outPtrs.push_back(n.c_str());
}

// ---------------------------------------------------------------------------
// cacheNames — variant for caching only inputs OR only outputs
//
// Replaces the lambda in onnx_dtln_inferencer.cpp that handles two sessions
// with separate input/output name vectors.
// ---------------------------------------------------------------------------
inline void cacheNames(Ort::Session& session,
                       std::vector<std::string>& names,
                       std::vector<const char*>& ptrs,
                       bool isInput)
{
	Ort::AllocatorWithDefaultOptions allocator;
	const std::size_t count = isInput
		? session.GetInputCount()
		: session.GetOutputCount();

	names.clear();
	ptrs.clear();
	for (std::size_t i = 0; i < count; ++i) {
		auto namePtr = isInput
			? session.GetInputNameAllocated(i, allocator)
			: session.GetOutputNameAllocated(i, allocator);
		names.emplace_back(namePtr.get());
	}
	for (const auto& n : names) {
		ptrs.push_back(n.c_str());
	}
}

// ---------------------------------------------------------------------------
// cpuMemoryInfo — cached MemoryInfo instance (avoid per-frame recreation)
//
// Previously recreated on every inference call in face_recog and basicvsrpp.
// Thread-safe: function-local static, initialized once.
// ---------------------------------------------------------------------------
[[nodiscard]] inline const Ort::MemoryInfo& cpuMemoryInfo()
{
	static const Ort::MemoryInfo info =
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	return info;
}

} // namespace oe::onnx
