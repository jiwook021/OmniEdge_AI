#include "audio_denoise/onnx_dtln_inferencer.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_onnx_helpers.hpp"
#include "common/onnx_session_handle.hpp"
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
// DTLN LSTM constants (shared by loadModel and processFrame)
// ---------------------------------------------------------------------------
constexpr std::size_t kLstmUnits  = 128;
constexpr std::size_t kLstmLayers = 2;
constexpr std::size_t kStateSize  = kLstmLayers * kLstmUnits;  // 256

// ---------------------------------------------------------------------------
// PIMPL — hides ONNX Runtime types from the header
// ---------------------------------------------------------------------------

struct OnnxDtlnInferencer::Impl {
	oe::onnx::SessionHandle stft{"DTLN_STFT"};   // Stage 1: STFT domain
	oe::onnx::SessionHandle time{"DTLN_Time"};    // Stage 2: time domain

	// LSTM hidden states (carried between frames)
	std::vector<float>             stftLstmState;  // Stage 1 LSTM state
	std::vector<float>             timeLstmState;  // Stage 2 LSTM state

	// Reusable input buffer (avoid per-frame allocation)
	std::vector<float> inputBuffer;

	// Owned copy of stage1 output for stage2 input — avoids aliasing
	// ONNX Runtime-owned memory via const_cast.  Uses assign() to
	// reuse capacity after the first frame (zero per-frame allocation).
	std::vector<float> stage1OutputCopy;
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
	if (impl_->stft.loaded) {
		unloadModel();
	}

	// Load both stages via SessionHandle (CUDA EP only)
	if (auto r = impl_->stft.loadCudaOnly(model1Path); !r) {
		return tl::unexpected(std::format("DTLN stage1: {}", r.error()));
	}
	if (auto r = impl_->time.loadCudaOnly(model2Path); !r) {
		return tl::unexpected(std::format("DTLN stage2: {}", r.error()));
	}

	// Initialize LSTM hidden states to zeros
	impl_->stftLstmState.assign(kStateSize, 0.0f);
	impl_->timeLstmState.assign(kStateSize, 0.0f);

	OE_LOG_INFO("dtln_models_loaded: stage1={}, stage2={}", model1Path, model2Path);
	return {};
}

// ---------------------------------------------------------------------------
// processFrame
// ---------------------------------------------------------------------------

tl::expected<std::vector<float>, std::string> OnnxDtlnInferencer::processFrame(
	std::span<const float> pcmInput)
{
	if (!impl_->stft.loaded) {
		return tl::unexpected(std::string("Model not loaded"));
	}

	const std::size_t frameSize = pcmInput.size();

	try {
		const auto& memoryInfo = oe::onnx::cpuMemoryInfo();

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

		auto stage1Outputs = impl_->stft.session->Run(
			Ort::RunOptions{nullptr},
			impl_->stft.inputPtrs(),
			stage1Inputs.data(),
			stage1Inputs.size(),
			impl_->stft.outputPtrs(),
			impl_->stft.outputCount());

		// Validate stage 1 produced at least the main output
		if (stage1Outputs.empty()) {
			return tl::unexpected(std::string("DTLN stage 1 produced no outputs"));
		}

		// Update LSTM state from stage 1 output
		if (stage1Outputs.size() > 1) {
			const float* newState = stage1Outputs[1].GetTensorData<float>();
			if (!newState) {
				return tl::unexpected(std::string("DTLN stage 1 LSTM state tensor data is null"));
			}
			const auto stateInfo = stage1Outputs[1].GetTensorTypeAndShapeInfo();
			const std::size_t stateElements =
				static_cast<std::size_t>(stateInfo.GetElementCount());
			if (stateElements != kStateSize) {
				return tl::unexpected(std::format(
					"DTLN stage 1 LSTM state size mismatch: expected {}, got {}",
					kStateSize, stateElements));
			}
			impl_->stftLstmState.assign(newState, newState + stateElements);
		}

		// --- Stage 2: time domain refinement ---
		const float* stage1Out = stage1Outputs[0].GetTensorData<float>();
		if (!stage1Out) {
			return tl::unexpected(std::string("DTLN stage 1 output tensor data is null"));
		}
		const auto stage1Info = stage1Outputs[0].GetTensorTypeAndShapeInfo();
		const auto stage1Shape = stage1Info.GetShape();
		const std::size_t stage1Elements =
			static_cast<std::size_t>(stage1Info.GetElementCount());

		// Copy into owned buffer — ONNX Runtime owns the original and
		// may reuse or free it during the stage2 Run() call.
		impl_->stage1OutputCopy.assign(stage1Out, stage1Out + stage1Elements);

		auto stage2Input = Ort::Value::CreateTensor<float>(
			memoryInfo,
			impl_->stage1OutputCopy.data(),
			impl_->stage1OutputCopy.size(),
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

		auto stage2Outputs = impl_->time.session->Run(
			Ort::RunOptions{nullptr},
			impl_->time.inputPtrs(),
			stage2Inputs.data(),
			stage2Inputs.size(),
			impl_->time.outputPtrs(),
			impl_->time.outputCount());

		// Validate stage 2 produced at least the main output
		if (stage2Outputs.empty()) {
			return tl::unexpected(std::string("DTLN stage 2 produced no outputs"));
		}

		// Update LSTM state from stage 2 output
		if (stage2Outputs.size() > 1) {
			const float* newState = stage2Outputs[1].GetTensorData<float>();
			if (!newState) {
				return tl::unexpected(std::string("DTLN stage 2 LSTM state tensor data is null"));
			}
			const auto stateInfo = stage2Outputs[1].GetTensorTypeAndShapeInfo();
			const std::size_t stateElements =
				static_cast<std::size_t>(stateInfo.GetElementCount());
			if (stateElements != kStateSize) {
				return tl::unexpected(std::format(
					"DTLN stage 2 LSTM state size mismatch: expected {}, got {}",
					kStateSize, stateElements));
			}
			impl_->timeLstmState.assign(newState, newState + stateElements);
		}

		// Extract denoised audio from stage 2 output
		const float* denoisedData = stage2Outputs[0].GetTensorData<float>();
		if (!denoisedData) {
			return tl::unexpected(std::string("DTLN stage 2 output tensor data is null"));
		}
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
	impl_->stft.unload();
	impl_->time.unload();
	impl_->stftLstmState.clear();
	impl_->timeLstmState.clear();
	impl_->stage1OutputCopy.clear();
	OE_LOG_INFO("dtln_models_unloaded");
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxDtlnInferencer::currentVramUsageBytes() const noexcept
{
	if (!impl_->stft.loaded) return 0;
	return kDtlnMiB * kBytesPerMebibyte;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<AudioDenoiseInferencer> createOnnxDtlnInferencer()
{
	return std::make_unique<OnnxDtlnInferencer>();
}

