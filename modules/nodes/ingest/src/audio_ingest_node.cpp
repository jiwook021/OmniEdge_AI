#include "ingest/audio_ingest_node.hpp"
#include "ingest/gstreamer_pipeline.hpp"
#include "ingest/silero_vad.hpp"

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/time_utils.hpp"

#include <chrono>
#include <cstring>
#include <format>
#include <thread>
#include <stdexcept>

#include <nlohmann/json.hpp>


namespace {

/// Slot size for the audio circular buffer: per-slot ShmAudioHeader + PCM data.
std::size_t audioSlotSize(uint32_t maxSamples) noexcept
{
	return sizeof(ShmAudioHeader) + static_cast<std::size_t>(maxSamples) * sizeof(float);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

AudioIngestNode::AudioIngestNode(const Config& config)
	: config_(config)
	, messageRouter_(MessageRouter::Config{
		config.moduleName,
		config.pubPort,
		config.zmqSendHighWaterMark,
		config.pollTimeout})
{
}

AudioIngestNode::~AudioIngestNode()
{
	stop();
}

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<AudioIngestNode::Config, std::string>
AudioIngestNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePositive("sampleRateHz", static_cast<int>(raw.sampleRateHz));
	v.requirePositive("chunkSamples", static_cast<int>(raw.chunkSamples));
	v.requireRangeF("vadSpeechThreshold", raw.vadSpeechThreshold, 0.0f, 1.0f);
	v.requireNonEmpty("vadModelPath", raw.vadModelPath);
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// ---------------------------------------------------------------------------
// onConfigure
// ---------------------------------------------------------------------------

tl::expected<void, std::string> AudioIngestNode::onConfigure()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// 1 — GStreamer init
	gst_init(nullptr, nullptr);

	// 2 — POSIX SHM circular buffer (producer)
	const std::size_t slotSz = audioSlotSize(config_.chunkSamples);
	shm_ = std::make_unique<ShmCircularBuffer<ShmAudioHeader>>(
		"/oe.aud.ingest", kRingBufferSlotCount, slotSz, /*create=*/true);

	// 3 — Silero VAD
	SileroVad::Config vadCfg;
	vadCfg.modelPath        = config_.vadModelPath;
	vadCfg.speechThreshold  = config_.vadSpeechThreshold;
	vadCfg.sampleRateHz     = config_.sampleRateHz;
	vadCfg.chunkSamples     = config_.chunkSamples;
	vad_ = std::make_unique<SileroVad>(vadCfg);

	// 4 — Subscribe to ui_command via MessageRouter (no conflation — control
	//     messages are sequential).  The handler runs on the poll thread.
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) {
			const auto action = parseUiAction(
				msg.value("action", std::string{}));
			handleUiCommand(action, msg);
		});

	// Register per-iteration callback for cross-thread drain.
	// Runs on the poll thread (main thread), where ZMQ sends are safe.
	messageRouter_.setOnPollCallback([this]() -> bool {
		drainPendingPublishes();
		return true;  // always continue polling
	});

	// Pre-allocate pending-publish queues to avoid hot-path heap allocation.
	// 64 entries covers ~1.9 s of 30 ms chunks — well beyond a single drain cycle.
	pendingChunks_.reserve(64);
	pendingVadMsgs_.reserve(64);

	OE_LOG_INFO("audio_ingest_configured: shm=/oe.aud.ingest, pub={}, vad_thr={}",
		config_.pubPort, config_.vadSpeechThreshold);

	return {};
}

// ---------------------------------------------------------------------------
// onLoadInferencer — build GStreamer pipeline object (do NOT start it)
// ---------------------------------------------------------------------------

tl::expected<void, std::string> AudioIngestNode::onLoadInferencer()
{
	OE_ZONE_SCOPED;

	// GStreamer pipeline
	const std::string pipelineStr = std::format(
		"tcpclientsrc host={} port={} "
		"! audio/x-raw,format=F32LE,rate={},channels=1 "
		"! audioconvert ! audioresample "
		"! audio/x-raw,format=F32LE,rate={},channels=1 "
		"! appsink name=audio_sink drop=true max-buffers=2",
		config_.windowsHostIp, config_.audioTcpPort,
		config_.sampleRateHz,  config_.sampleRateHz);

	pipeline_ = std::make_unique<GStreamerPipeline>(
		pipelineStr, "audio_sink",
		[this](const uint8_t* data, std::size_t size, uint64_t pts) {
			onAudioBuffer(data, size, pts);
		});

	return {};
}

// ---------------------------------------------------------------------------
// onBeforeRun — starts GStreamer pipeline (degraded start on failure)
// ---------------------------------------------------------------------------

void AudioIngestNode::onBeforeRun()
{
	// Start GStreamer pipeline in a background thread.  tcpclientsrc blocks for
	// 30-60 s waiting for the Windows-side TCP audio server.  By running this
	// in a background thread, the poll loop starts immediately and heartbeats
	// flow to the daemon — preventing the watchdog from SIGKILL-ing the module.
	// The thread is stored as a member and joined in onAfterStop() to prevent
	// use-after-free if the module is destroyed while start() is still blocking.
	pipelineStartThread_ = std::thread([this]() {
		try {
			pipeline_->start();
			pipelineActive_.store(true, std::memory_order_release);
			OE_LOG_INFO("audio_pipeline_started_async");
		} catch (const std::exception& e) {
			OE_LOG_WARN("audio_pipeline_degraded: error={} — "
			          "module will run without audio until source is available",
			          e.what());
		}
	});
}

// ---------------------------------------------------------------------------
// onAfterStop — stops GStreamer pipeline
// ---------------------------------------------------------------------------

void AudioIngestNode::onAfterStop()
{
	// Stop the pipeline first — this unblocks pipeline_->start() if it is
	// still waiting for the TCP connection in the background thread.
	if (pipeline_) {
		pipeline_->stop();
	}

	// Join the startup thread. Must happen after pipeline_->stop() so the
	// thread is unblocked and can exit cleanly.
	if (pipelineStartThread_.joinable()) {
		pipelineStartThread_.join();
	}

	OE_LOG_INFO("audio_ingest_run_exited");
}

// ---------------------------------------------------------------------------
// onAudioBuffer() — GStreamer streaming thread
// ---------------------------------------------------------------------------

void AudioIngestNode::onAudioBuffer(const uint8_t* data,
									 std::size_t    size,
									 uint64_t       /*pts*/)
{
	OE_ZONE_SCOPED;
	// Interpret raw bytes as float32 PCM
	const auto* samples    = reinterpret_cast<const float*>(data);
	const std::size_t nSamples = size / sizeof(float);

	// Process in config_.chunkSamples strides (30 ms each).
	// Lock stateMtx_ to protect writeSlot_, chunkSeq_, vadSpeaking_,
	// silenceSamples_, and the pending publish queues.
	std::lock_guard<std::mutex> lk(stateMtx_);

	std::size_t offset = 0;
	while (offset + config_.chunkSamples <= nSamples) {
		const std::span<const float> chunk(samples + offset, config_.chunkSamples);

		auto result = vad_->classify(chunk);
		if (!result) {
			OE_LOG_WARN("vad_classify_error: error={}", result.error());
			offset += config_.chunkSamples;
			continue;
		}

		const float prob    = result.value();
		const bool  speech  = vad_->isSpeech(prob);

		OE_LOG_DEBUG("vad_chunk: prob={:.3f}, speech={}, speaking={}, silence_samples={}",
		           prob, speech, vadSpeaking_, silenceSamples_);

		if (speech) {
			// Write speech chunk to SHM circular buffer
			writeSpeechChunk(samples + offset, config_.chunkSamples);

			if (!vadSpeaking_) {
				vadSpeaking_    = true;
				silenceSamples_ = 0;
				pendingVadMsgs_.push_back({true, 0});
			}
			silenceSamples_ = 0;
		} else {
			silenceSamples_ += config_.chunkSamples;

			// Compute accumulated silence in milliseconds.
			// Use uint64_t intermediate to prevent overflow when
			// silenceSamples_ exceeds ~4.29M (268 s at 16 kHz).
			const uint32_t silenceMs = static_cast<uint32_t>(
				(static_cast<uint64_t>(silenceSamples_) * 1000u) / config_.sampleRateHz);

			if (vadSpeaking_ && silenceMs >= config_.silenceDurationMs) {
				vadSpeaking_    = false;
				silenceSamples_ = 0;
				pendingVadMsgs_.push_back({false, silenceMs});
			}
		}

		offset += config_.chunkSamples;
	}
}

// ---------------------------------------------------------------------------
// writeSpeechChunk()
// ---------------------------------------------------------------------------

void AudioIngestNode::writeSpeechChunk(const float*  samples,
										uint32_t      numSamples)
{
	// Caller must hold stateMtx_.
	auto [slotBase, slotIdx] = shm_->acquireWriteSlot();

	// Write per-slot header
	auto* audioHeader         = reinterpret_cast<ShmAudioHeader*>(slotBase);
	audioHeader->sampleRateHz = config_.sampleRateHz;
	audioHeader->numSamples   = numSamples;
	const uint64_t seq = chunkSeq_++;
	audioHeader->seqNumber    = seq;
	const uint64_t captureTimestampNs = static_cast<uint64_t>(steadyClockNanoseconds());
	audioHeader->timestampNs  = captureTimestampNs;

	// Write PCM data immediately after header
	float* pcmDestination = reinterpret_cast<float*>(slotBase + sizeof(ShmAudioHeader));
	std::memcpy(pcmDestination, samples, numSamples * sizeof(float));

	// Commit: makes the slot visible to the consumer
	shm_->commitWrite();

	OE_LOG_DEBUG("audio_chunk_written: slot={}, samples={}, seq={}", slotIdx, numSamples, seq);

	// Enqueue ZMQ notification for the main thread to publish.
	// ZMQ sockets are NOT thread-safe — never call pubSocket_ from this thread.
	pendingChunks_.push_back({slotIdx, numSamples, captureTimestampNs, seq});
}

// ---------------------------------------------------------------------------
// resetVadAndStartPipeline() — shared by start_audio and push_to_talk
// ---------------------------------------------------------------------------

bool AudioIngestNode::resetVadAndStartPipeline()
{
	{
		std::lock_guard<std::mutex> lk(stateMtx_);
		vad_->resetState();
		vadSpeaking_    = false;
		silenceSamples_ = 0;
	}

	if (!pipeline_->isRunning()) {
		try {
			pipeline_->start();
			return true;
		} catch (const std::exception& ex) {
			OE_LOG_WARN("audio_pipeline_start_failed: error={}", ex.what());
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// handleUiCommand() — dispatch a parsed ui_command on the main thread
// ---------------------------------------------------------------------------

bool AudioIngestNode::handleUiCommand(UiAction action,
                                       const nlohmann::json& cmd)
{
	switch (action) {
	case UiAction::kStartAudio:
		if (resetVadAndStartPipeline()) {
			pipelineActive_.store(true, std::memory_order_release);
		}
		OE_LOG_INFO("audio_pipeline_started");
		return true;

	case UiAction::kStopAudio:
		pipeline_->stop();
		{
			std::lock_guard<std::mutex> lk(stateMtx_);
			vadSpeaking_    = false;
			silenceSamples_ = 0;
		}
		OE_LOG_INFO("audio_pipeline_stopped");
		return true;

	case UiAction::kPushToTalk: {
		const bool pressed = cmd.value("state", false);
		if (pressed) {
			if (resetVadAndStartPipeline()) {
				pipelineActive_.store(true, std::memory_order_release);
			}
		} else {
			pipeline_->stop();
		}
		return true;
	}

	default:
		return false;
	}
}

// ---------------------------------------------------------------------------
// drainPendingPublishes() — called from run() on the main thread only
// ---------------------------------------------------------------------------

void AudioIngestNode::drainPendingPublishes()
{
	// Snapshot and clear the queues under the lock, then publish without it.
	// This minimises the time stateMtx_ is held and keeps ZMQ sends off the
	// GStreamer thread entirely.
	std::vector<PendingChunkMsg> chunks;
	std::vector<PendingVadMsg>   vadMsgs;

	{
		std::lock_guard<std::mutex> lk(stateMtx_);
		chunks.swap(pendingChunks_);
		vadMsgs.swap(pendingVadMsgs_);
	}

	for (const auto& c : chunks) {
		publishChunkMsg(c.slot, c.samples, c.ts, c.seq);
	}
	for (const auto& v : vadMsgs) {
		publishVadMsg(v.speaking, v.silenceMs);
	}
}

// ---------------------------------------------------------------------------
// publishChunkMsg() — called from run() on the main thread only
// ---------------------------------------------------------------------------

void AudioIngestNode::publishChunkMsg(uint32_t slot, uint32_t samples,
									   uint64_t ts, uint64_t seq)
{
	static thread_local nlohmann::json msg = {
		{"v",       kSchemaVersion},
		{"type",    "audio_chunk"},
		{"shm",     "/oe.aud.ingest"},
		{"slot",    uint32_t{0}},
		{"samples", uint32_t{0}},
		{"seq",     uint64_t{0}},
		{"vad",     "speech"},
		{"ts",      int64_t{0}},
	};

	msg["slot"]    = slot;
	msg["samples"] = samples;
	msg["seq"]     = seq;
	msg["ts"]      = static_cast<int64_t>(ts);

	messageRouter_.publish("audio_chunk", msg);
}

// ---------------------------------------------------------------------------
// publishVadMsg() — called from run() on the main thread only
// ---------------------------------------------------------------------------

void AudioIngestNode::publishVadMsg(bool speaking, uint32_t silenceMs)
{
	messageRouter_.publish("vad_status", {
		{"v",          kSchemaVersion},
		{"type",       "vad_status"},
		{"speaking",   speaking},
		{"silence_ms", silenceMs},
	});
	OE_LOG_INFO("vad_status_published: speaking={}, silence_ms={}",
		speaking, silenceMs);
}

