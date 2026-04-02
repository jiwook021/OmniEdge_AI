#include "audio_denoise/onnx_dtln_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/constants/memory_constants.hpp"
#include "vram/vram_thresholds.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxDtlnInferencer::Impl {
	Ort::Env                env{ORT_LOGGING_LEVEL_WARNING, "DTLN"};
	Ort::SessionOptions     stftSessionOpts;
	Ort::SessionOptions     timeSessionOpts;
	std::unique_ptr<Ort::Session> stftSession;  // Stage 1: STFT domain
	std::unique_ptr<Ort::Session> timeSession;  // Stage 2: time domain

	// Cached I/O names for both stages
	std::vector<std::string>       stftInputNames;
	std::vector<std::string>       stftOutputNames;
	std::vector<const char*>       stftInputNamePtrs;
	std::vector<const char*>       stftOutputNamePtrs;

	std::vector<std::string>       timeInputNames;
	std::vector<std::string>       timeOutputNames;
	std::vector<const char*>       timeInputNamePtrs;
	std::vector<const char*>       timeOutputNamePtrs;

	// LSTM hidden states (carried between frames)
	std::vector<float>             stftLstmState;  // Stage 1 LSTM state
	std::vector<float>             timeLstmState;  // Stage 2 LSTM state

	// Cached MemoryInfo (avoid recreating per frame)
	Ort::MemoryInfo memInfo{Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault)};

	// Reusable input buffer (avoid per-frame allocation)
	std::vector<float> inputBuffer;

	bool modelLoaded = false;
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

OnnxDtlnInferencer::OnnxDtlnInferencer()
	: impl_(std::make_unique<Impl>())
{
}

OnnxDtlnInferencer::~OnnxDtlnInferencer()
{
	unloadModel();
}

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------

tl::expected<void, std::string> OnnxDtlnInferencer::loadModel(
	const std::string& model1Path,
	const std::string& model2Path)
{
	try {
		auto configureSession = [](Ort::SessionOptions& opts) {
			opts.SetIntraOpNumThreads(1);
			opts.SetGraphOptimizationLevel(
				GraphOptimizationLevel::ORT_ENABLE_ALL);

			// CUDA EP required — no CPU fallback
			OrtCUDAProviderOptions cudaOptions{};
			cudaOptions.device_id = 0;
			cudaOptions.arena_extend_strategy = 0;
			cudaOptions.do_copy_in_default_stream = 1;
			opts.AppendExecutionProvider_CUDA(cudaOptions);
		};

		try {
			configureSession(impl_->stftSessionOpts);
			configureSession(impl_->timeSessionOpts);
		} catch (const Ort::Exception& ex) {
			return tl::unexpected(std::format(
				"DTLN CUDA EP registration failed (is CUDA available?): {}", ex.what()));
		}

		// Load both stages on CUDA
		impl_->stftSession = std::make_unique<Ort::Session>(
			impl_->env, model1Path.c_str(), impl_->stftSessionOpts);
		impl_->timeSession = std::make_unique<Ort::Session>(
			impl_->env, model2Path.c_str(), impl_->timeSessionOpts);

		// Cache I/O names for both stages
		Ort::AllocatorWithDefaultOptions allocator;

		auto cacheNames = [&allocator](
			Ort::Session& session,
			std::vector<std::string>& names,
			std::vector<const char*>& ptrs,
			bool isInput)
		{
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
		};

		cacheNames(*impl_->stftSession, impl_->stftInputNames,  impl_->stftInputNamePtrs,  true);
		cacheNames(*impl_->stftSession, impl_->stftOutputNames, impl_->stftOutputNamePtrs, false);
		cacheNames(*impl_->timeSession, impl_->timeInputNames,  impl_->timeInputNamePtrs,  true);
		cacheNames(*impl_->timeSession, impl_->timeOutputNames, impl_->timeOutputNamePtrs, false);

		// Initialize LSTM hidden states to zeros
		// DTLN uses 2 LSTM layers with 128 units each per stage
		constexpr std::size_t kLstmUnits = 128;
		constexpr std::size_t kLstmLayers = 2;
		constexpr std::size_t kStateSize = kLstmLayers * kLstmUnits;
		impl_->stftLstmState.assign(kStateSize, 0.0f);
		impl_->timeLstmState.assign(kStateSize, 0.0f);

		impl_->modelLoaded = true;
		OE_LOG_INFO("dtln_models_loaded: stage1={}, stage2={}", model1Path, model2Path);
		return {};
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"ONNX Runtime error loading DTLN: {}", ex.what()));
	}
}

// ---------------------------------------------------------------------------
// processFrame
// ---------------------------------------------------------------------------

tl::expected<std::vector<float>, std::string> OnnxDtlnInferencer::processFrame(
	std::span<const float> pcmInput)
{
	if (!impl_->modelLoaded) {
		return tl::unexpected(std::string("Model not loaded"));
	}

	const std::size_t frameSize = pcmInput.size();

	try {
		const auto& memoryInfo = impl_->memInfo;

		// --- Stage 1: STFT domain processing ---
		// Input: [1, frame_size] audio frame + [2, 128] LSTM state
		const std::array<int64_t, 2> inputShape = {
			1, static_cast<int64_t>(frameSize)};
		const std::array<int64_t, 2> stateShape = {
			2, 128};

		// Reuse input buffer to avoid per-frame heap allocation
		impl_->inputBuffer.assign(pcmInput.begin(), pcmInput.end());
		auto& inputCopy = impl_->inputBuffer;
		auto inputTensor = Ort::Value::CreateTensor<float>(
			memoryInfo,
			inputCopy.data(),
			inputCopy.size(),
			inputShape.data(),
			inputShape.size());

		auto stateTensor1 = Ort::Value::CreateTensor<float>(
			memoryInfo,
			impl_->stftLstmState.data(),
			impl_->stftLstmState.size(),
			stateShape.data(),
			stateShape.size());

		std::array<Ort::Value, 2> stage1Inputs;
		stage1Inputs[0] = std::move(inputTensor);
		stage1Inputs[1] = std::move(stateTensor1);

		auto stage1Outputs = impl_->stftSession->Run(
			Ort::RunOptions{nullptr},
			impl_->stftInputNamePtrs.data(),
			stage1Inputs.data(),
			stage1Inputs.size(),
			impl_->stftOutputNamePtrs.data(),
			impl_->stftOutputNamePtrs.size());

		// Update LSTM state from stage 1 output
		if (stage1Outputs.size() > 1) {
			const float* newState = stage1Outputs[1].GetTensorData<float>();
			const auto stateInfo = stage1Outputs[1].GetTensorTypeAndShapeInfo();
			const std::size_t stateElements =
				static_cast<std::size_t>(stateInfo.GetElementCount());
			impl_->stftLstmState.assign(newState, newState + stateElements);
		}

		// --- Stage 2: time domain refinement ---
		const float* stage1Out = stage1Outputs[0].GetTensorData<float>();
		const auto stage1Info = stage1Outputs[0].GetTensorTypeAndShapeInfo();
		const auto stage1Shape = stage1Info.GetShape();

		auto stage2Input = Ort::Value::CreateTensor<float>(
			memoryInfo,
			const_cast<float*>(stage1Out),
			static_cast<std::size_t>(stage1Info.GetElementCount()),
			stage1Shape.data(),
			stage1Shape.size());

		auto stateTensor2 = Ort::Value::CreateTensor<float>(
			memoryInfo,
			impl_->timeLstmState.data(),
			impl_->timeLstmState.size(),
			stateShape.data(),
			stateShape.size());

		std::array<Ort::Value, 2> stage2Inputs;
		stage2Inputs[0] = std::move(stage2Input);
		stage2Inputs[1] = std::move(stateTensor2);

		auto stage2Outputs = impl_->timeSession->Run(
			Ort::RunOptions{nullptr},
			impl_->timeInputNamePtrs.data(),
			stage2Inputs.data(),
			stage2Inputs.size(),
			impl_->timeOutputNamePtrs.data(),
			impl_->timeOutputNamePtrs.size());

		// Update LSTM state from stage 2 output
		if (stage2Outputs.size() > 1) {
			const float* newState = stage2Outputs[1].GetTensorData<float>();
			const auto stateInfo = stage2Outputs[1].GetTensorTypeAndShapeInfo();
			const std::size_t stateElements =
				static_cast<std::size_t>(stateInfo.GetElementCount());
			impl_->timeLstmState.assign(newState, newState + stateElements);
		}

		// Extract denoised audio from stage 2 output
		const float* denoisedData = stage2Outputs[0].GetTensorData<float>();
		const auto outInfo = stage2Outputs[0].GetTensorTypeAndShapeInfo();
		const std::size_t outCount =
			static_cast<std::size_t>(outInfo.GetElementCount());

		return std::vector<float>(denoisedData, denoisedData + outCount);

	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"DTLN inference error: {}", ex.what()));
	}
}

// ---------------------------------------------------------------------------
// resetState
// ---------------------------------------------------------------------------

void OnnxDtlnInferencer::resetState()
{
	std::fill(impl_->stftLstmState.begin(), impl_->stftLstmState.end(), 0.0f);
	std::fill(impl_->timeLstmState.begin(), impl_->timeLstmState.end(), 0.0f);
	OE_LOG_INFO("dtln_state_reset");
}

// ---------------------------------------------------------------------------
// unloadModel
// ---------------------------------------------------------------------------

void OnnxDtlnInferencer::unloadModel()
{
	impl_->stftSession.reset();
	impl_->timeSession.reset();
	impl_->stftLstmState.clear();
	impl_->timeLstmState.clear();
	impl_->modelLoaded = false;
	OE_LOG_INFO("dtln_models_unloaded");
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxDtlnInferencer::currentVramUsageBytes() const noexcept
{
	if (!impl_->modelLoaded) return 0;
	return kDtlnMiB * kBytesPerMebibyte;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<AudioDenoiseInferencer> createOnnxDtlnInferencer()
{
	return std::make_unique<OnnxDtlnInferencer>();
}

