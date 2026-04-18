#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — OnnxSessionHandle: composable ONNX Runtime session wrapper
//
// Encapsulates the repeated pattern of (Env, SessionOptions, Session,
// cached I/O name vectors) that appears in every ONNX inferencer's PIMPL.
//
// Inferencers compose 1 or more handles:
//   - DTLN:       2 handles (stft + time)
//   - Face Recog: 2 handles (detect + recog)
//   - BasicVSR++: 1 handle
//
// Usage:
//   oe::onnx::SessionHandle encoder{"Encoder"};
//   auto result = encoder.load(modelPath, SessionConfig{.useTRT = true});
//   if (!result) return tl::unexpected(result.error());
//   auto outputs = encoder.session->Run(..., encoder.inputPtrs(), ...);
// ---------------------------------------------------------------------------

#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <tl/expected.hpp>

#include "common/oe_onnx_helpers.hpp"


namespace oe::onnx {

struct SessionHandle {
	Ort::Env                      env;
	Ort::SessionOptions           opts;
	std::unique_ptr<Ort::Session> session;

	std::vector<std::string>      inputNames;
	std::vector<std::string>      outputNames;
	std::vector<const char*>      inputNamePtrs;
	std::vector<const char*>      outputNamePtrs;

	bool loaded = false;

	/// Construct with a label for ONNX Runtime logging.
	explicit SessionHandle(const char* label)
		: env(ORT_LOGGING_LEVEL_WARNING, label)
	{}

	SessionHandle(const SessionHandle&)            = delete;
	SessionHandle& operator=(const SessionHandle&) = delete;
	SessionHandle(SessionHandle&&)                 = default;
	SessionHandle& operator=(SessionHandle&&)      = default;

	// -----------------------------------------------------------------
	// load — configure EP, create session, cache I/O names
	//
	// @param modelPath  Path to .onnx model file
	// @param config     EP configuration (CUDA graph, cuDNN, TRT, VRAM cap)
	// @return           void on success, error string on failure
	// -----------------------------------------------------------------
	[[nodiscard]] tl::expected<void, std::string>
	load(const std::string& modelPath, const SessionConfig& config)
	{
		// Reset options to avoid accumulating EPs across reloads
		opts = Ort::SessionOptions{};

		try {
			configureSession(opts, "session", config);
		} catch (const std::runtime_error& ex) {
			return tl::unexpected(std::format("EP registration failed: {}", ex.what()));
		}

		try {
			session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
		} catch (const Ort::Exception& ex) {
			return tl::unexpected(std::format(
				"Session creation failed for '{}': {}", modelPath, ex.what()));
		}

		cacheIONames(*session, inputNames, outputNames, inputNamePtrs, outputNamePtrs);
		loaded = true;
		return {};
	}

	// -----------------------------------------------------------------
	// loadCudaOnly — CUDA EP only (no TRT), for lightweight models
	// -----------------------------------------------------------------
	[[nodiscard]] tl::expected<void, std::string>
	loadCudaOnly(const std::string& modelPath, std::size_t gpuMemLimitMiB = 0)
	{
		return load(modelPath, SessionConfig{
			.useTRT         = false,
			.gpuMemLimitMiB = gpuMemLimitMiB,
		});
	}

	// -----------------------------------------------------------------
	// loadCpuOnly — CPU-only execution (VAD, small utility models)
	// -----------------------------------------------------------------
	[[nodiscard]] tl::expected<void, std::string>
	loadCpuOnly(const std::string& modelPath, int intraThreads = 1)
	{
		opts = Ort::SessionOptions{};
		opts.SetIntraOpNumThreads(intraThreads);
		opts.SetInterOpNumThreads(1);
		opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		try {
			session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
		} catch (const Ort::Exception& ex) {
			return tl::unexpected(std::format(
				"CPU session creation failed for '{}': {}", modelPath, ex.what()));
		}

		cacheIONames(*session, inputNames, outputNames, inputNamePtrs, outputNamePtrs);
		loaded = true;
		return {};
	}

	// -----------------------------------------------------------------
	// unload — destroy session, clear cached names
	// -----------------------------------------------------------------
	void unload()
	{
		session.reset();
		inputNames.clear();
		outputNames.clear();
		inputNamePtrs.clear();
		outputNamePtrs.clear();
		loaded = false;
	}

	// -----------------------------------------------------------------
	// Accessors for Run() arguments
	// -----------------------------------------------------------------
	[[nodiscard]] const char* const* inputPtrs()  const { return inputNamePtrs.data(); }
	[[nodiscard]] const char* const* outputPtrs() const { return outputNamePtrs.data(); }
	[[nodiscard]] std::size_t        inputCount()  const { return inputNamePtrs.size(); }
	[[nodiscard]] std::size_t        outputCount() const { return outputNamePtrs.size(); }
};

} // namespace oe::onnx
