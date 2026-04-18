#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Shared ONNX Runtime Utilities
//
// Eliminates duplicated session configuration, I/O name caching, and
// MemoryInfo creation across all ONNX Runtime inferencer classes.
//
// Uses ONNX Runtime V2 provider APIs for CUDA EP and TensorRT EP,
// enabling settings unavailable in the legacy struct APIs:
//   - enable_cuda_graph (fixed-shape model optimization)
//   - cudnn_conv_use_max_workspace (better algorithm selection)
//   - trt_timing_cache_enable (faster engine rebuilds)
// ---------------------------------------------------------------------------

#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "common/oe_logger.hpp"


namespace oe::onnx {

// ---------------------------------------------------------------------------
// SessionConfig — per-model EP configuration
//
// Controls which ONNX Runtime execution provider features are enabled.
// Pass to configureSession() or SessionHandle::load().
// ---------------------------------------------------------------------------
struct SessionConfig {
	bool        useTRT            = true;   ///< Append TensorRT EP before CUDA EP
	std::size_t gpuMemLimitMiB    = 0;      ///< Cap CUDA arena (0 = no explicit cap)
	bool        enableCudaGraph   = false;  ///< Cache CUDA graph on first run (fixed-shape models only)
	bool        exhaustiveCudnn   = false;  ///< EXHAUSTIVE cuDNN algo search (slower first inference)
	bool        maxCudnnWorkspace = false;  ///< Allow max cuDNN workspace for best algo selection
};

// ---------------------------------------------------------------------------
// trtCachePath — project-relative TensorRT engine cache directory
//
// Resolves via OE_PROJECT_ROOT (exported by run_all.sh) so that `git clone` +
// run works on any machine.  Engines are GPU-architecture-specific and must be
// gitignored, but the directory structure is created automatically.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string trtCachePath()
{
	const char* root = std::getenv("OE_PROJECT_ROOT");
	std::string base = root ? std::string(root) + "/model.cache/trt_engines"
	                        : "./model.cache/trt_engines";
	std::filesystem::create_directories(base);
	return base;
}

// ---------------------------------------------------------------------------
// trtTimingCachePath — project-relative TensorRT timing cache directory
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string trtTimingCachePath()
{
	const char* root = std::getenv("OE_PROJECT_ROOT");
	std::string base = root ? std::string(root) + "/model.cache/trt_timing"
	                        : "./model.cache/trt_timing";
	std::filesystem::create_directories(base);
	return base;
}

// ---------------------------------------------------------------------------
// configureSession — set up TRT EP (optional) + CUDA EP (required)
//
// Uses V2 provider APIs for both TensorRT and CUDA execution providers,
// enabling modern features: CUDA graphs, exhaustive cuDNN search,
// TRT timing cache, and max cuDNN workspace.
//
// @param opts    Session options to configure (must be freshly constructed)
// @param label   Module name for log messages (e.g. "scrfd", "basicvsrpp")
// @param config  EP configuration (see SessionConfig)
// ---------------------------------------------------------------------------
inline void configureSession(Ort::SessionOptions& opts,
                             std::string_view label,
                             const SessionConfig& config)
{
	opts.SetIntraOpNumThreads(1);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	// --- TensorRT EP (V2 API) ---
	if (config.useTRT) {
		OrtTensorRTProviderOptionsV2* trtOpts = nullptr;
		try {
			Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trtOpts));
		} catch (const Ort::Exception& ex) {
			OE_LOG_WARN("{}_trt_ep_create_failed [OE-DAEMON-5001]: {}", label, ex.what());
			trtOpts = nullptr;
		}

		if (trtOpts != nullptr) {
			static const std::string engineCache = trtCachePath();
			static const std::string timingCache  = trtTimingCachePath();

			const std::vector<const char*> trtKeys = {
				"device_id",
				"trt_max_workspace_size",
				"trt_fp16_enable",
				"trt_engine_cache_enable",
				"trt_engine_cache_path",
				"trt_timing_cache_enable",
				"trt_timing_cache_path",
				"trt_builder_optimization_level",
			};
			const std::vector<const char*> trtValues = {
				"0",
				"1073741824",          // 1 GB workspace
				"1",                   // FP16 inference
				"1",                   // persistent engine cache
				engineCache.c_str(),
				"1",                   // timing cache across engine builds
				timingCache.c_str(),
				"3",                   // default optimization level
			};

			try {
				Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
					trtOpts, trtKeys.data(), trtValues.data(), trtKeys.size()));
				opts.AppendExecutionProvider_TensorRT_V2(*trtOpts);
			} catch (const Ort::Exception& ex) {
				OE_LOG_WARN("{}_trt_ep_failed: {}", label, ex.what());
			}

			Ort::GetApi().ReleaseTensorRTProviderOptions(trtOpts);
		}
	}

	// --- CUDA EP (V2 API) ---
	OrtCUDAProviderOptionsV2* cudaOpts = nullptr;
	try {
		Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOpts));
	} catch (const Ort::Exception& ex) {
		OE_LOG_WARN("{}_cuda_ep_create_failed [OE-DAEMON-5001]: {} (falling back to CPU EP)",
			label, ex.what());
		return;
	}

	const std::string memLimitStr = (config.gpuMemLimitMiB > 0)
		? std::to_string(config.gpuMemLimitMiB * 1024ULL * 1024ULL)
		: "0";   // 0 = no explicit cap

	const char* algoSearch = config.exhaustiveCudnn ? "EXHAUSTIVE" : "HEURISTIC";
	const char* cudaGraph  = config.enableCudaGraph ? "1" : "0";
	const char* maxWs      = config.maxCudnnWorkspace ? "1" : "0";

	std::vector<const char*> cudaKeys = {
		"device_id",
		"arena_extend_strategy",
		"cudnn_conv_algo_search",
		"do_copy_in_default_stream",
		"cudnn_conv_use_max_workspace",
		"enable_cuda_graph",
	};
	std::vector<const char*> cudaValues = {
		"0",
		"kSameAsRequested",    // grow on demand, avoid over-allocation
		algoSearch,
		"0",                   // separate stream for copies (enables overlap)
		maxWs,
		cudaGraph,
	};

	// Only set gpu_mem_limit if explicitly requested
	if (config.gpuMemLimitMiB > 0) {
		cudaKeys.push_back("gpu_mem_limit");
		cudaValues.push_back(memLimitStr.c_str());
	}

	try {
		Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
			cudaOpts, cudaKeys.data(), cudaValues.data(), cudaKeys.size()));
		opts.AppendExecutionProvider_CUDA_V2(*cudaOpts);
	} catch (const Ort::Exception& ex) {
		Ort::GetApi().ReleaseCUDAProviderOptions(cudaOpts);
		throw std::runtime_error(std::format(
			"{} CUDA EP registration failed: {}", label, ex.what()));
	}

	Ort::GetApi().ReleaseCUDAProviderOptions(cudaOpts);
}

// ---------------------------------------------------------------------------
// cacheIONames — extract and cache input/output tensor names from a session
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
// Thread-safe: function-local static, initialized once.
// ---------------------------------------------------------------------------
[[nodiscard]] inline const Ort::MemoryInfo& cpuMemoryInfo()
{
	static const Ort::MemoryInfo info =
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	return info;
}

} // namespace oe::onnx
