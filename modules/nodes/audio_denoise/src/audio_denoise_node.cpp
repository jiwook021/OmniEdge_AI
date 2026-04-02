#include "audio_denoise/audio_denoise_node.hpp"

#include "common/constants/audio_denoise_constants.hpp"
#include "common/oe_tracy.hpp"
#include "gpu/cuda_guard.hpp"
#include "vram/vram_thresholds.hpp"

#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>

#include <nlohmann/json.hpp>


namespace {

// SHM ring-buffer layout constants moved to audio_denoise/audio_denoise_constants.hpp.

} // anonymous namespace

// --- Construction / Destruction ---

AudioDenoiseNode::AudioDenoiseNode(const Config& config,
                                   std::unique_ptr<AudioDenoiseInferencer> inferencer)
	: config_(config),
	  inferencer_(std::move(inferencer)),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  kPublisherDataHighWaterMark,
		  config.pollTimeout})
{
}

AudioDenoiseNode::~AudioDenoiseNode()
{
	stop();
}

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<AudioDenoiseNode::Config, std::string>
AudioDenoiseNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requireNonEmpty("model1Path", raw.model1Path);
	v.requireNonEmpty("model2Path", raw.model2Path);
	v.requirePort("subAudioPort", raw.subAudioPort);
	v.requirePort("subWsBridgePort", raw.subWsBridgePort);
	v.requirePort("pubPort", raw.pubPort);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// ---------------------------------------------------------------------------
// onConfigure
// ---------------------------------------------------------------------------

tl::expected<void, std::string> AudioDenoiseNode::onConfigure()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	if (!inferencer_) {
		return tl::unexpected(std::string("[AudioDenoiseNode] No inferencer — pass via constructor"));
	}

	// Input SHM: /oe.aud.ingest (consumer — does NOT unlink on close)
	shmInput_ = std::make_unique<ShmMapping>(
		config_.shmInput, kInputAudioShmSegmentByteSize, /*create=*/false);

	// Output SHM: /oe.aud.denoise (producer — will unlink on close)
	shmOutput_ = std::make_unique<ShmMapping>(
		config_.shmOutput, kOutputAudioShmSegmentByteSize, /*create=*/true);
	std::memset(shmOutput_->data(), 0, kOutputAudioShmSegmentByteSize);

	// Pre-allocate PCM scratch buffer to the max chunk size so handleAudioChunk()
	// never triggers a heap allocation on the hot path.
	inputPcmBuffer_.resize(kVadChunkSampleCount);

	// ZMQ subscriptions via MessageRouter
	messageRouter_.subscribe(config_.subAudioPort, "audio_chunk", /*conflate=*/true,
		[this](const nlohmann::json& msg) { handleAudioChunk(msg); });

	OE_LOG_INFO("audio_denoise_configured: pub={}, model1={}, model2={}",
		config_.pubPort, config_.model1Path, config_.model2Path);

	return {};
}

// ---------------------------------------------------------------------------
// onLoadInferencer
// ---------------------------------------------------------------------------

tl::expected<void, std::string> AudioDenoiseNode::onLoadInferencer()
{
	OE_ZONE_SCOPED;

	auto loadResult = inferencer_->loadModel(config_.model1Path, config_.model2Path);
	if (!loadResult) {
		return tl::unexpected(
			std::format("Inferencer loadModel failed: {}", loadResult.error()));
	}

	// Ensure LSTM hidden states start from zero for a fresh audio stream
	inferencer_->resetState();

	OE_LOG_INFO("audio_denoise_inferencer_loaded: model1={}, model2={}",
		config_.model1Path, config_.model2Path);

	return {};
}

// --- handleAudioChunk() ---

void AudioDenoiseNode::handleAudioChunk(const nlohmann::json& chunkMetadata)
{
	OE_ZONE_SCOPED;
	const int shmSlotIndex = chunkMetadata.value("slot",    -1);
	const int sampleCount  = chunkMetadata.value("samples",  0);

	if (shmSlotIndex < 0 || sampleCount <= 0) {
		OE_LOG_WARN("audio_denoise_invalid_chunk: slot={}, samples={}",
			shmSlotIndex, sampleCount);
		return;
	}
	if (static_cast<std::size_t>(shmSlotIndex) >= kRingBufferSlotCount) {
		OE_LOG_WARN("audio_denoise_slot_oob: slot={}", shmSlotIndex);
		return;
	}
	if (static_cast<uint32_t>(sampleCount) > kVadChunkSampleCount) {
		OE_LOG_WARN("audio_denoise_chunk_too_large: samples={}, max={}",
			sampleCount, kVadChunkSampleCount);
		return;
	}

	// Locate the slot in the SHM ring buffer
	const uint8_t* slotBase = shmInput_->bytes()
	                           + kAudioShmDataOffset
	                           + static_cast<std::size_t>(shmSlotIndex) * kAudioShmSlotByteSize;

	// Stale-read guard: compare sequence number before and after copy
	const auto* audioSlotHeader = reinterpret_cast<const ShmAudioHeader*>(slotBase);
	const uint64_t seqBefore = audioSlotHeader->seqNumber;

	const float* sourcePcm = reinterpret_cast<const float*>(
		slotBase + sizeof(ShmAudioHeader));

	// Copy input PCM locally (reuse pre-allocated buffer to avoid hot-path heap alloc)
	inputPcmBuffer_.resize(static_cast<std::size_t>(sampleCount));
	std::memcpy(inputPcmBuffer_.data(), sourcePcm,
		static_cast<std::size_t>(sampleCount) * sizeof(float));

	const uint64_t seqAfter = audioSlotHeader->seqNumber;
	if (seqAfter != seqBefore) {
		OE_LOG_WARN("audio_denoise_stale_shm: slot={}, seq_before={}, seq_after={}",
			shmSlotIndex, seqBefore, seqAfter);
		return;
	}

	// VRAM pre-flight: reject frame if GPU memory is critically low
	if (auto check = ensureVramAvailableMiB(kDtlnInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("audio_denoise_preflight_failed: {}", check.error());
		return;
	}

	// Process through DTLN inferencer
	auto result = inferencer_->processFrame(
		std::span<const float>(inputPcmBuffer_.data(), inputPcmBuffer_.size()));

	if (!result) {
		OE_LOG_WARN("audio_denoise_process_error: error={}", result.error());
		return;
	}
	const auto& denoisedPcm = result.value();
	OE_LOG_DEBUG("audio_denoise_done: samples={}, vram_used={}MiB, host_rss={}KiB",
	             denoisedPcm.size(), queryVramMiB().usedMiB, hostRssKiB());

	// Bounds check: denoised output must fit in the output SHM segment
	if (denoisedPcm.size() > kOutputShmMaxSampleCount) {
		OE_LOG_WARN("audio_denoise_output_overflow: samples={}, max={}",
			denoisedPcm.size(), kOutputShmMaxSampleCount);
		return;
	}

	// Write denoised PCM to output SHM
	auto* outHeader = reinterpret_cast<ShmAudioHeader*>(shmOutput_->bytes());
	outHeader->sampleRateHz = kSttInputSampleRateHz;
	outHeader->numSamples   = static_cast<uint32_t>(denoisedPcm.size());
	outHeader->seqNumber    = ++seqNumber_;
	outHeader->timestampNs  = static_cast<uint64_t>(
		std::chrono::steady_clock::now().time_since_epoch().count());

	float* outPcm = reinterpret_cast<float*>(
		shmOutput_->bytes() + sizeof(ShmAudioHeader));
	std::memcpy(outPcm, denoisedPcm.data(),
		denoisedPcm.size() * sizeof(float));

	// Publish ZMQ notification
	static thread_local nlohmann::json payload = {
		{"v",       kSchemaVersion},
		{"type",    "denoised_audio"},
		{"shm",     ""},
		{"samples", int64_t{0}},
		{"seq",     uint64_t{0}},
	};
	payload["shm"]     = config_.shmOutput;
	payload["samples"] = static_cast<int64_t>(denoisedPcm.size());
	payload["seq"]     = seqNumber_;
	messageRouter_.publish("denoised_audio", payload);
}

