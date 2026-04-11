#include "tts/tts_node.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/oe_tracy.hpp"
#include "gpu/cuda_guard.hpp"
#include "gpu/gpu_diagnostics.hpp"
#include "common/time_utils.hpp"
#include "common/ui_action.hpp"


// ---------------------------------------------------------------------------
// Constructors / Destructor
// ---------------------------------------------------------------------------

TTSNode::TTSNode(const Config& config,
                              std::unique_ptr<TTSInferencer> inferencer)
	: config_(config)
	, inferencer_(std::move(inferencer))
	, messageRouter_(MessageRouter::Config{
		config_.moduleName,
		config_.pubPort,
		kPublisherControlHighWaterMark,
		config_.pollTimeout})
{
}

TTSNode::~TTSNode()
{
	stop();
	// The inferencer destructor calls unloadModel() which frees CUDA/ONNX state.
	// ZMQ sockets are cleaned up by cppzmq RAII destructors.
}

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<TTSNode::Config, std::string> TTSNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requireNonEmpty("onnxModelPath", raw.onnxModelPath);
	v.requireNonEmpty("voiceDir", raw.voiceDir);
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("subLlmPort", raw.subLlmPort);
	v.requirePort("subDaemonPort", raw.subDaemonPort);
	v.requireRangeF("speed", raw.speed, 0.1f, 5.0f);

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// ---------------------------------------------------------------------------
// configureTransport
// ---------------------------------------------------------------------------

tl::expected<void, std::string> TTSNode::configureTransport()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// ── 1. Create TTS SHM segment ─────────────────────────────────────────
	// Producer: create = true.  Destructor will call shm_unlink.
	shmOutput_ = std::make_unique<ShmMapping>(
		config_.shmOutput,
		kTtsShmBytes,
		/*create=*/true);

	// Initialise header in the segment.
	auto* ttsHeader = reinterpret_cast<ShmTtsHeader*>(shmOutput_->bytes());
	*ttsHeader = ShmTtsHeader{};

	// ── 2. Register ZMQ subscriptions via MessageRouter ──────────────────
	messageRouter_.subscribe(config_.subLlmPort, "llm_response", false,
		[this](const nlohmann::json& msg) { handleLlmResponse(msg); });
	messageRouter_.subscribe(config_.subDaemonPort, "module_status", false,
		[this](const nlohmann::json& msg) { handleDaemonMessage(msg); });

	return {};
}

// ---------------------------------------------------------------------------
// loadInferencer
// ---------------------------------------------------------------------------

tl::expected<void, std::string> TTSNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	// ── 1. Load ONNX model ────────────────────────────────────────────────
	auto loadResult = inferencer_->loadModel(config_.onnxModelPath,
	                                      config_.voiceDir);
	if (!loadResult) {
		// INT8 model may fail on some ONNX Runtime builds (ConvInteger
		// unsupported).  Fall back to the FP32 model in the same directory.
		namespace fs = std::filesystem;
		const auto fp32Path =
			(fs::path(config_.onnxModelPath).parent_path() / "model.onnx").string();
		if (config_.onnxModelPath != fp32Path && fs::exists(fp32Path)) {
			OE_LOG_WARN("tts_int8_fallback: int8_error={}, trying fp32={}",
				loadResult.error(), fp32Path);
			config_.onnxModelPath = fp32Path;
			loadResult = inferencer_->loadModel(config_.onnxModelPath,
			                                 config_.voiceDir);
		}
		if (!loadResult) {
			return tl::unexpected(
				std::format("model load failed: {}", loadResult.error()));
		}
	}

	OE_LOG_INFO("tts_initialized: model={}, voice_dir={}, voice={}, shm={}",
	          config_.onnxModelPath, config_.voiceDir, config_.defaultVoice, config_.shmOutput);

	// ── 2. Pre-warm the default voice style tensor ───────────────────────
	// The checklist requires voice styles loaded from .npy at init time to
	// avoid first-synthesis latency.  Run a silent warmup synthesis so the
	// .npy is loaded and cached before the first real llm_response arrives.
	{
		OE_LOG_INFO("tts_warmup_start: voice={}", config_.defaultVoice);
		const auto warmupStart = std::chrono::steady_clock::now();

		auto warmup = inferencer_->synthesize(
			".", config_.defaultVoice, config_.speed);

		const auto warmupMs = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - warmupStart).count();

		if (!warmup) {
			// Non-fatal: log and continue — synthesis will attempt to load on
			// first real sentence; this only means first-sentence latency is higher.
			OE_LOG_WARN("tts_voice_warmup_failed: voice={}, elapsed_ms={}, error={}",
			         config_.defaultVoice, warmupMs, warmup.error());
		} else {
			OE_LOG_INFO("tts_voice_warmed: voice={}, samples={}, elapsed_ms={}",
			          config_.defaultVoice, warmup->size(), warmupMs);
		}
	}

	return {};
}

// ---------------------------------------------------------------------------
// handleLlmResponse
// ---------------------------------------------------------------------------

void TTSNode::handleLlmResponse(const nlohmann::json& msg)
{
	OE_ZONE_SCOPED;
	// Expected fields (all optional for robustness):
	//   "token"             : string — text token emitted by LLM
	//   "finished"          : bool   — true on last token
	//   "sentence_boundary" : bool   — true if this token starts a new sentence

	const std::string token = msg.value("token", std::string{});
	const bool finished     = msg.value("finished", false);
	const bool sentBoundary = msg.value("sentence_boundary", false);

	OE_LOG_DEBUG("tts_llm_token: token_len={}, finished={}, sentence_boundary={}",
	           token.size(), finished, sentBoundary);

	// ── Secondary: LLM sentence_boundary flag ─────────────────────────────
	// The LLM sets sentence_boundary:true on the FIRST token of the NEXT
	// sentence.  Flush the current buffer FIRST so the boundary token
	// lands only in the fresh buffer — never in both old and new.
	if (sentBoundary) {
		if (auto tail = splitter_.flush()) {
			if (!tail->empty()) {
				synthesizeAndPublish(*tail);
			}
		}
		// Now append the boundary token into the fresh (empty) buffer.
		if (!token.empty()) {
			if (auto sentence = splitter_.appendToken(token)) {
				synthesizeAndPublish(*sentence);
			}
		}
		return;
	}

	// ── Primary boundary detection via SentenceSplitter ──────────────────
	if (!token.empty()) {
		if (auto sentence = splitter_.appendToken(token)) {
			synthesizeAndPublish(*sentence);
		}
	}

	// ── Flush on finished ─────────────────────────────────────────────────
	if (finished) {
		if (auto tail = splitter_.flush()) {
			synthesizeAndPublish(*tail);
		}
	}
}

// ---------------------------------------------------------------------------
// handleDaemonMessage
// ---------------------------------------------------------------------------

void TTSNode::handleDaemonMessage(const nlohmann::json& msg)
{
	OE_ZONE_SCOPED;
	// The daemon sends {"action":"flush_tts"} to clear the synthesis queue
	// during the INTERRUPTED state transition (barge-in).
	const auto action = parseUiAction(msg.value("action", std::string{}));
	switch (action) {
	case UiAction::kFlushTts:
		splitter_.reset();
		OE_LOG_INFO("tts_flushed: reason=daemon_flush_tts");
		break;
	default:
		break;
	}
}

// ---------------------------------------------------------------------------
// synthesizeAndPublish
// ---------------------------------------------------------------------------

void TTSNode::synthesizeAndPublish(const std::string& text)
{
	OE_ZONE_SCOPED;
	if (text.empty()) {
		return;
	}

	// ── VRAM pre-flight before inference ──────────────────────────────────
	if (auto check = ensureVramAvailableMiB(kTtsInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("tts_vram_preflight_failed: {}", check.error());
		return;
	}

	const int64_t synthStartNs = steadyClockNanoseconds();

	// Synthesize PCM via the inferencer.
	auto result = inferencer_->synthesize(
		text,
		config_.defaultVoice,
		config_.speed);

	if (!result) {
		OE_LOG_WARN("tts_synthesis_failed: text_len={}, error={}",
		          text.size(), result.error());
		return;
	}

	const std::vector<float>& pcm = *result;

	if (pcm.empty()) {
		OE_LOG_WARN("tts_synthesis_empty_pcm: text_len={}", text.size());
		return;
	}

	if (pcm.size() > kTtsShmMaxSamples) {
		OE_LOG_WARN("tts_synthesis_too_long: samples={}, max_samples={}, text_len={}",
		          pcm.size(), kTtsShmMaxSamples, text.size());
		return;
	}

	// ── Write to SHM ───────────────────────────────────────────────────────
	const int64_t writeNs = steadyClockNanoseconds();

	// Write PCM data FIRST — a reader polling seqNumber must not see new
	// metadata while the sample payload is still being memcpy'd.
	std::memcpy(shmOutput_->bytes() + ShmTtsHeader::dataOffset(),
	            pcm.data(),
	            pcm.size() * sizeof(float));

	// Memory fence ensures PCM is visible before metadata commit
	std::atomic_thread_fence(std::memory_order_release);

	// THEN commit header (readers poll seqNumber to detect new data)
	auto* ttsHeader = reinterpret_cast<ShmTtsHeader*>(shmOutput_->bytes());
	ttsHeader->sampleRateHz = kTtsOutputSampleRateHz;
	ttsHeader->numSamples   = static_cast<uint32_t>(pcm.size());
	ttsHeader->seqNumber    = ++seqNumber_;
	ttsHeader->timestampNs  = static_cast<uint64_t>(writeNs);

	// ── Publish tts_audio ZMQ notification ────────────────────────────────
	messageRouter_.publish("tts_audio", {
		{"v",       kSchemaVersion},
		{"type",    "tts_audio"},
		{"shm",     config_.shmOutput},
		{"samples", static_cast<int>(pcm.size())},
		{"rate",    static_cast<int>(kTtsOutputSampleRateHz)},
		{"seq",     seqNumber_},
	});

	const int64_t synthNs  = steadyClockNanoseconds() - synthStartNs;
	const int64_t synthMs  = synthNs / 1'000'000;
	const double  audioDurationMs =
		static_cast<double>(pcm.size()) /
		static_cast<double>(kTtsOutputSampleRateHz) * 1000.0;
	const double  rtf = (synthMs > 0)
		? (audioDurationMs / static_cast<double>(synthMs))
		: 0.0;

	OE_LOG_INFO("tts_synthesis_done: samples={}, synth_ms={}, rtf={:.2f}, seq={}, text_len={}, "
	            "vram_used={}MiB, host_rss={}KiB",
	          pcm.size(), synthMs, rtf, seqNumber_, text.size(),
	          queryVramMiB().usedMiB, hostRssKiB());
}

