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
	  state_(kSileroVadStateSize, 0.0f),
	  context_(kSileroVadContextSize16k, 0.0f)
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
	std::fill(state_.begin(), state_.end(), 0.0f);
	std::fill(context_.begin(), context_.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// classify()
// ---------------------------------------------------------------------------

tl::expected<float, std::string> SileroVad::classify(std::span<const float> pcm)
{
	OE_ZONE_SCOPED;
	if (pcm.size() < config_.chunkSamples) {
		return tl::unexpected(
			std::format("[SileroVad] chunk too small: got {} need {}",
						pcm.size(), config_.chunkSamples));
	}

	const auto& memInfo = ort_->memInfo;

	// ---- Input tensors ----
	// 1. audio input [1, kSileroVadTotalInputSamples] — context + PCM chunk
	//    Silero VAD v5 requires a context window (last 64 samples from the
	//    previous chunk) prepended to the current 512-sample chunk.
	//    inputBuf_ is a class member to stay warm in L1/L2 cache at 33 Hz.

	// Copy context (64 samples) then PCM chunk (up to 512 samples)
	std::copy_n(context_.data(), kSileroVadContextSize16k, inputBuf_.begin());
	const std::size_t copyN = std::min<std::size_t>(pcm.size(), kSileroVadModelInputSamples);
	std::copy_n(pcm.begin(), copyN, inputBuf_.begin() + kSileroVadContextSize16k);

	std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(kSileroVadTotalInputSamples)};
	auto inputTensor = Ort::Value::CreateTensor<float>(
		memInfo, inputBuf_.data(), inputBuf_.size(),
		inputShape.data(), inputShape.size());

	// 2. sample_rate [] int64 (ONNX scalar = empty shape)
	int64_t sampleRate = static_cast<int64_t>(config_.sampleRateHz);
	std::array<int64_t, 0> scalarShape{};
	auto sampleRateTensor = Ort::Value::CreateTensor<int64_t>(
		memInfo, &sampleRate, 1, scalarShape.data(), 0);

	// 3. state [2, 1, 128] float32 (unified GRU state)
	std::array<int64_t, 3> stateShape{
		kSileroVadStateDim0, kSileroVadStateDim1, kSileroVadStateDim2};
	auto stateTensor = Ort::Value::CreateTensor<float>(
		memInfo, state_.data(), state_.size(),
		stateShape.data(), stateShape.size());

	// ---- Run inference ----
	const std::array<const char*, 3> inputNames  = {"input", "state", "sr"};
	const std::array<const char*, 2> outputNames = {"output", "stateN"};

	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(inputTensor));
	inputs.push_back(std::move(stateTensor));
	inputs.push_back(std::move(sampleRateTensor));

	std::vector<Ort::Value> outputs;
	try {
		outputs = ort_->session.Run(
			Ort::RunOptions{nullptr},
			inputNames.data(),  inputs.data(),  inputs.size(),
			outputNames.data(), outputNames.size());
	} catch (const Ort::Exception& e) {
		return tl::unexpected(
			std::format("OE-VAD-4001: SileroVad inference failed: {}", e.what()));
	}

	// ---- Extract outputs ----
	const float speechProb = *outputs[0].GetTensorData<float>();

	OE_LOG_DEBUG("silero_vad_classify: prob={:.4f}, threshold={}",
	           speechProb, config_.speechThreshold);

	// Update GRU state for next chunk
	const float* stateNData = outputs[1].GetTensorData<float>();
	std::copy_n(stateNData, kSileroVadStateSize, state_.begin());

	// Update context: last kSileroVadContextSize16k samples from the input
	std::copy_n(inputBuf_.data() + kSileroVadTotalInputSamples - kSileroVadContextSize16k,
	            kSileroVadContextSize16k, context_.begin());

	return speechProb;
}

