#include "ingest/silero_vad.hpp"

#include "common/constants/ingest_constants.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// OrtState — PImpl holding all ORT objects
// ---------------------------------------------------------------------------

struct SileroVad::OrtState {
	Ort::Env            env;
	Ort::SessionOptions options;
	Ort::Session        session;

	// Options must be fully configured before passing to the Session constructor.
	// Doing it in the ctor body is too late — Session is already built by then.
	static Ort::SessionOptions makeOptions()
	{
		Ort::SessionOptions opts;
		// CPU only — no CUDA provider for VAD
		opts.SetIntraOpNumThreads(1);
		opts.SetInterOpNumThreads(1);
		opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		return opts;
	}

	Ort::MemoryInfo    memInfo;

	static Ort::MemoryInfo makeMemInfo()
	{
		return Ort::MemoryInfo::CreateCpu(
			OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
	}

	explicit OrtState(const std::string& modelPath)
		: env(ORT_LOGGING_LEVEL_WARNING, "SileroVAD"),
		  options(makeOptions()),
		  session(env, modelPath.c_str(), options),
		  memInfo(makeMemInfo())
	{}
};

// ---------------------------------------------------------------------------
// GRU state dimensions
// ---------------------------------------------------------------------------

namespace {
// GRU state constants now live in ingest_constants.hpp
} // anonymous namespace

// ---------------------------------------------------------------------------
// SileroVad — constructor / destructor
// ---------------------------------------------------------------------------

SileroVad::SileroVad(const Config& config)
	: config_(config),
	  h_(kSileroVadHiddenStateSize, 0.0f),
	  c_(kSileroVadCellStateSize, 0.0f)
{
	try {
		ort_ = std::make_unique<OrtState>(config_.modelPath);
	} catch (const Ort::Exception& e) {
		throw std::runtime_error(
			std::format("[SileroVad] Failed to load model '{}': {}",
						config_.modelPath, e.what()));
	}
	OE_LOG_INFO("silero_vad_loaded: model={}, threshold={}",
		config_.modelPath, config_.speechThreshold);
}

SileroVad::~SileroVad() = default;

// ---------------------------------------------------------------------------
// resetState()
// ---------------------------------------------------------------------------

void SileroVad::resetState()
{
	std::fill(h_.begin(), h_.end(), 0.0f);
	std::fill(c_.begin(), c_.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// classify()
// ---------------------------------------------------------------------------

tl::expected<float, std::string> SileroVad::classify(std::span<const float> pcm)
{
	if (pcm.size() < config_.chunkSamples) {
		return tl::unexpected(
			std::format("[SileroVad] chunk too small: got {} need {}",
						pcm.size(), config_.chunkSamples));
	}

	const auto& memInfo = ort_->memInfo;

	// ---- Input tensors ----
	// 1. audio input [1, kSileroVadModelInputSamples] — copy PCM, zero-pad if shorter
	std::array<float, kSileroVadModelInputSamples> inputBuf{};
	const std::size_t copyN = std::min<std::size_t>(pcm.size(), kSileroVadModelInputSamples);
	std::copy_n(pcm.begin(), copyN, inputBuf.begin());

	std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(kSileroVadModelInputSamples)};
	auto inputTensor = Ort::Value::CreateTensor<float>(
		memInfo, inputBuf.data(), inputBuf.size(),
		inputShape.data(), inputShape.size());

	// 2. sample_rate [] int64 (ONNX scalar = empty shape)
	int64_t sampleRate = static_cast<int64_t>(config_.sampleRateHz);
	std::array<int64_t, 0> scalarShape{};
	auto sampleRateTensor = Ort::Value::CreateTensor<int64_t>(
		memInfo, &sampleRate, 1, scalarShape.data(), 0);

	// 3. h [2, 1, 64] float32
	std::array<int64_t, 3> stateShape{2, 1, 64};
	auto hTensor = Ort::Value::CreateTensor<float>(
		memInfo, h_.data(), h_.size(),
		stateShape.data(), stateShape.size());

	// 4. c [2, 1, 64] float32
	auto cTensor = Ort::Value::CreateTensor<float>(
		memInfo, c_.data(), c_.size(),
		stateShape.data(), stateShape.size());

	// ---- Run inference ----
	const std::array<const char*, 4> inputNames  = {"input", "sr", "h", "c"};
	const std::array<const char*, 3> outputNames = {"output", "hn", "cn"};

	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(inputTensor));
	inputs.push_back(std::move(sampleRateTensor));
	inputs.push_back(std::move(hTensor));
	inputs.push_back(std::move(cTensor));

	std::vector<Ort::Value> outputs;
	try {
		outputs = ort_->session.Run(
			Ort::RunOptions{nullptr},
			inputNames.data(),  inputs.data(),  inputs.size(),
			outputNames.data(), outputNames.size());
	} catch (const Ort::Exception& e) {
		return tl::unexpected(
			std::format("[SileroVad] inference failed: {}", e.what()));
	}

	// ---- Extract outputs ----
	const float speechProb = *outputs[0].GetTensorData<float>();

	OE_LOG_DEBUG("silero_vad_classify: prob={:.4f}, threshold={}",
	           speechProb, config_.speechThreshold);

	// Update GRU state for next chunk
	const float* hnData = outputs[1].GetTensorData<float>();
	const float* cnData = outputs[2].GetTensorData<float>();
	std::copy_n(hnData, kSileroVadHiddenStateSize, h_.begin());
	std::copy_n(cnData, kSileroVadCellStateSize, c_.begin());

	return speechProb;
}

