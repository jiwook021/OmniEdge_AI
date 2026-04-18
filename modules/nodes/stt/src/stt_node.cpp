#include "stt/stt_node.hpp"

#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "common/oe_shm_helpers.hpp"
#include "common/oe_tracy.hpp"
#include "common/time_utils.hpp"
#include "gpu/cuda_guard.hpp"
#include "gpu/cuda_stream.hpp"
#include "gpu/oe_cuda_check.hpp"


namespace {

// ---------------------------------------------------------------------------
} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

STTNode::STTNode(const Config& config)
	: config_(config)
{
}

STTNode::~STTNode()
{
	stop();
	if (cudaStream_) {
		cudaStreamDestroy(cudaStream_);
		cudaStream_ = nullptr;
	}
}

// ---------------------------------------------------------------------------
// setInferencer()
// ---------------------------------------------------------------------------

void STTNode::setInferencer(std::unique_ptr<STTInferencer> inferencer) noexcept
{
	inferencer_ = std::move(inferencer);
}

// ---------------------------------------------------------------------------
// configureTransport()
// ---------------------------------------------------------------------------

tl::expected<void, std::string> STTNode::configureTransport()
{
	OE_ZONE_SCOPED;

	// 1 — Logger
	OeLogger::instance().setModule(config_.moduleName);

	// 2 — MessageRouter
	{
		MessageRouter::Config routerCfg;
		routerCfg.moduleName  = config_.moduleName;
		routerCfg.pubPort     = config_.pubPort;
		routerCfg.pubHwm      = config_.zmqSendHighWaterMark;
		routerCfg.pollTimeout = config_.pollTimeout;
		messageRouter_ = std::make_unique<MessageRouter>(routerCfg);
	}

	// 3 — High-priority CUDA stream (priority clamped to device range)
	{
		auto clamped = clampCudaStreamPriority(config_.cudaStreamPriority);
		if (!clamped) {
			OE_LOG_WARN("stt_cuda_stream_priority_unavailable: {}; continuing without dedicated stream",
			            clamped.error());
		} else {
			const cudaError_t err = cudaStreamCreateWithPriority(&cudaStream_,
			                                                    cudaStreamNonBlocking,
			                                                    *clamped);
			if (err != cudaSuccess) {
				OE_LOG_WARN("stt_cuda_stream_create_failed: {}; continuing without dedicated stream",
				            cudaGetErrorString(err));
				cudaStream_ = nullptr;
			}
		}
	}

	// 4 — SHM consumer (/oe.aud.ingest — retry up to 10 times waiting for producer)
	auto shmResult = oe::shm::openConsumerWithRetry(
		config_.inputShmName, kAudioShmSegmentByteSize, "stt");
	if (!shmResult) return tl::unexpected(shmResult.error());
	audioInputShm_ = std::move(*shmResult);

	// 5 — Subscribe to upstream topics via MessageRouter
	messageRouter_->subscribe(config_.audioSubPort, "audio_chunk", /*conflate=*/true,
		[this](const nlohmann::json& msg) { processAudioChunk(msg); });
	messageRouter_->subscribe(config_.audioSubPort, "vad_status", /*conflate=*/false,
		[this](const nlohmann::json& msg) { handleVadStatus(msg); });

	// 6 — CPU mel spectrogram processor (precomputes Hann window + filterbank)
	melProcessor_ = std::make_unique<MelSpectrogram>(
		static_cast<int>(config_.numMelBins),
		static_cast<int>(config_.fftWindowSize),
		static_cast<int>(config_.hopLength),
		static_cast<int>(config_.sampleRateHz));

	// 7 — Hallucination filter
	{
		HallucinationFilter::Config filterCfg;
		filterCfg.noSpeechProbThreshold = config_.noSpeechProbThreshold;
		filterCfg.minAvgLogprob         = config_.minAvgLogprob;
		filterCfg.maxConsecutiveRepeats = config_.maxConsecutiveRepeats;
		hallucinationFilter_ = std::make_unique<HallucinationFilter>(filterCfg);
	}

	// 8 — Reserve audio accumulator
	audioBuffer_.reserve(
		static_cast<std::size_t>(config_.chunkSamples + config_.overlapSamples));

	OE_LOG_INFO("stt_configured: pub={}, audio_sub={}, shm={}",
	          config_.pubPort, config_.audioSubPort, config_.inputShmName);

	return {};
}

// ---------------------------------------------------------------------------
// loadInferencer()
// ---------------------------------------------------------------------------

tl::expected<void, std::string> STTNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer set — call setInferencer() before initialize()"));
	}

	try {
		inferencer_->loadModel(config_.encoderEngineDir,
		                    config_.decoderEngineDir,
		                    config_.tokenizerDir);
	} catch (const std::exception& e) {
		return tl::unexpected(std::string("loadModel failed: ") + e.what());
	}

	OE_LOG_INFO("stt_inferencer_loaded: encoder={}, decoder={}",
	          config_.encoderEngineDir, config_.decoderEngineDir);

	return {};
}

// ---------------------------------------------------------------------------
// onAfterStop() — optional CRTP hook: flush remaining audio after poll loop
// ---------------------------------------------------------------------------

void STTNode::onAfterStop()
{
	if (!audioBuffer_.empty()) {
		runInference();
	}
	OE_LOG_INFO("stt_run_exited");
}

// ---------------------------------------------------------------------------
// Config::validate()
// ---------------------------------------------------------------------------

tl::expected<STTNode::Config, std::string> STTNode::Config::validate(const Config& raw)
{
	ConfigValidator v;

	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("audioSubPort", raw.audioSubPort);
	v.requireNonEmpty("encoderEngineDir", raw.encoderEngineDir);
	v.requireNonEmpty("decoderEngineDir", raw.decoderEngineDir);
	v.requirePositive("sampleRateHz", static_cast<int>(raw.sampleRateHz));
	v.requirePositive("numMelBins", static_cast<int>(raw.numMelBins));
	v.requirePositive("chunkSamples", static_cast<int>(raw.chunkSamples));
	v.requireRangeF("noSpeechProbThreshold", raw.noSpeechProbThreshold, 0.0f, 1.0f);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}

	return raw;
}

// ---------------------------------------------------------------------------
// handleVadStatus()
// ---------------------------------------------------------------------------

void STTNode::handleVadStatus(const nlohmann::json& msg)
{
	const bool speaking = msg.value("speaking", true);
	if (!speaking && !audioBuffer_.empty()) {
		runInference();
		hallucinationFilter_->reset();
	}
}

// ---------------------------------------------------------------------------
// processAudioChunk()
// ---------------------------------------------------------------------------

void STTNode::processAudioChunk(const nlohmann::json& chunkMetadata)
{
	OE_ZONE_SCOPED;
	const int shmSlotIndex = chunkMetadata.value("slot",    -1);
	const int sampleCount  = chunkMetadata.value("samples",  0);

	if (shmSlotIndex < 0 || sampleCount <= 0) {
		OE_LOG_WARN("stt_invalid_chunk: slot={}, samples={}", shmSlotIndex, sampleCount);
		return;
	}
	if (static_cast<std::size_t>(shmSlotIndex) >= kRingBufferSlotCount) {
		OE_LOG_WARN("stt_slot_oob: slot={}", shmSlotIndex);
		return;
	}
	// Reject oversized inputs (input validation at system boundary)
	if (static_cast<uint32_t>(sampleCount) > kVadChunkSampleCount) {
		OE_LOG_WARN("stt_chunk_too_large: samples={}, max={}", sampleCount, kVadChunkSampleCount);
		return;
	}

	// Locate the slot in the SHM ring buffer
	const uint8_t* slotBase = audioInputShm_->bytes()
	                           + kAudioShmDataOffset
	                           + static_cast<std::size_t>(shmSlotIndex) * kAudioShmSlotByteSize;

	// Stale-read guard: compare sequence number before and after copy
	const auto* audioSlotHeader = reinterpret_cast<const ShmAudioHeader*>(slotBase);
	const uint64_t sequenceBeforeRead = audioSlotHeader->seqNumber;

	const float* sourcePcmData = reinterpret_cast<const float*>(
		slotBase + sizeof(ShmAudioHeader));

	OE_LOG_DEBUG("stt_chunk: slot={}, samples={}, buffer_before={}, seq={}",
	           shmSlotIndex, sampleCount, audioBuffer_.size(), sequenceBeforeRead);

	const std::size_t prevSize = audioBuffer_.size();
	audioBuffer_.resize(prevSize + static_cast<std::size_t>(sampleCount));
	std::memcpy(audioBuffer_.data() + prevSize,
	            sourcePcmData,
	            static_cast<std::size_t>(sampleCount) * sizeof(float));

	const uint64_t sequenceAfterRead = audioSlotHeader->seqNumber;
	if (sequenceAfterRead != sequenceBeforeRead) {
		// Producer overwrote this slot while we were reading — discard the tail
		audioBuffer_.resize(prevSize);
		OE_LOG_WARN("stt_stale_shm: slot={}, seq_before={}, seq_after={}", shmSlotIndex, sequenceBeforeRead, sequenceAfterRead);
		return;
	}

	// Trigger inference when we have a full 30 s window
	if (audioBuffer_.size() >= config_.chunkSamples) {
		runInference();
	}
}

// ---------------------------------------------------------------------------
// runInference()
// ---------------------------------------------------------------------------

void STTNode::runInference()
{
	OE_ZONE_SCOPED;
	if (audioBuffer_.empty()) {
		return;
	}

	const double inferenceStartMs = steadyClockMilliseconds();

	// ── VRAM pre-flight before inference ──────────────────────────────────
	if (auto check = ensureVramAvailableMiB(kSttInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("stt_vram_preflight_failed: {}", check.error());
		audioBuffer_.clear();
		return;
	}

	// Compute 128-bin log-mel spectrogram on CPU via MelSpectrogram class
	const std::vector<float> melFrames =
		melProcessor_->compute(audioBuffer_.data(), audioBuffer_.size());
	const int melFrameCount = melProcessor_->numFrames(audioBuffer_.size());

	OE_LOG_DEBUG("mel_spectrogram_computed: audio_samples={}, frames={}, n_mels={}",
	             audioBuffer_.size(), melFrameCount, config_.numMelBins);

	if (melFrames.empty() || melFrameCount == 0) {
		OE_LOG_WARN("stt_mel_empty: audio_samples={}, mel_frames={}", audioBuffer_.size(), melFrameCount);
		audioBuffer_.clear();
		return;
	}

	// Run TRT encoder + greedy decoder
	auto result = inferencer_->transcribe(
		std::span<const float>(melFrames.data(), melFrames.size()),
		static_cast<uint32_t>(melFrameCount));

	if (!result.has_value()) {
		OE_LOG_WARN("stt_infer_error: error={}", result.error());
		audioBuffer_.clear();
		return;
	}

	const double latencyMs = steadyClockMilliseconds() - inferenceStartMs;
	OE_LOG_DEBUG("stt_inference: latency_ms={:.1f}, vram_used={}MiB, host_rss={}KiB",
	             latencyMs, queryVramMiB().usedMiB, hostRssKiB());

	// Suppress empty/whitespace-only transcriptions — sending them to the LLM
	// results in a prompt with no user content.
	const auto& text = result.value().text;
	if (text.empty() || text.find_first_not_of(" \t\n\r") == std::string::npos) {
		OE_LOG_DEBUG("stt_empty_text_suppressed: text_len={}", text.size());
		audioBuffer_.clear();
		return;
	}

	// Apply hallucination filter (stateful — updates repeat counter internally)
	if (!hallucinationFilter_->isHallucination(result.value())) {
		publishTranscription(result.value(), latencyMs);
	} else {
		OE_LOG_INFO("stt_hallucination_discarded: text={}, no_speech_prob={}, avg_logprob={}",
		          result.value().text, result.value().noSpeechProb, result.value().avgLogprob);
	}

	// Keep last overlapSamples (1 s) for boundary continuity on the next chunk
	if (audioBuffer_.size() > config_.overlapSamples) {
		const std::size_t keepStart =
			audioBuffer_.size() - config_.overlapSamples;
		std::memmove(audioBuffer_.data(),
		             audioBuffer_.data() + keepStart,
		             config_.overlapSamples * sizeof(float));
		audioBuffer_.resize(config_.overlapSamples);
	} else {
		audioBuffer_.clear();
	}
}

// ---------------------------------------------------------------------------
// publishTranscription()
// ---------------------------------------------------------------------------

void STTNode::publishTranscription(const TranscribeResult& result,
                                           double                   latencyMs)
{
	nlohmann::json payload = {
		{"v",          kSchemaVersion},
		{"type",       "transcription"},
		{"text",       result.text},
		{"lang",       result.language.empty() ? "en" : result.language},
		{"source",     "stt"},
		{"confidence", result.avgLogprob},
	};
	messageRouter_->publish("transcription", payload);

	OE_LOG_INFO("stt_transcription_published: text={}, lang={}, latency_ms={:.1f}, confidence={}",
	          result.text, result.language, latencyMs, result.avgLogprob);
}
