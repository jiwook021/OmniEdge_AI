// conversation_node.cpp -- ConversationNode full implementation

#include "conversation/conversation_node.hpp"

#include <chrono>
#include <format>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "common/zmq_messages.hpp"

#include "common/constants/conversation_constants.hpp"
#include "common/constants/video_constants.hpp"
#include "common/oe_tracy.hpp"
#include "common/oe_shm_helpers.hpp"
#include "common/ui_action.hpp"
#include "common/oe_logger.hpp"
#include "zmq/audio_constants.hpp"
#include "zmq/heartbeat_constants.hpp"
#include "shm/shm_frame_reader.hpp"
#include "shm/shm_circular_buffer.hpp"


namespace {

/// @brief Return true if @p text ends with a sentence-ending punctuation mark.
[[nodiscard]] bool endsSentence(std::string_view text) noexcept
{
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    return (last == '.' || last == '?' || last == '!');
}

} // namespace

// ---------------------------------------------------------------------------
// Config::validate
// ---------------------------------------------------------------------------

tl::expected<ConversationNode::Config, std::string>
ConversationNode::Config::validate(const Config& raw)
{
    ConfigValidator v;

    v.requirePort("pubPort", raw.pubPort);
    v.requirePort("daemonSubPort", raw.daemonSubPort);
    v.requirePort("uiCommandSubPort", raw.uiCommandSubPort);
    v.requireNonEmpty("modelDir", raw.modelDir);
    v.requireNonEmpty("modelVariant", raw.modelVariant);

    if (auto err = v.finish(); !err.empty()) {
        return tl::unexpected(err);
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ConversationNode::ConversationNode(
    const Config& config,
    std::unique_ptr<ConversationInferencer> inferencer)
    : config_(config)
    , inferencer_(std::move(inferencer))
    , messageRouter_(MessageRouter::Config{
        .moduleName  = config.moduleName,
        .pubPort     = config.pubPort,
        .pubHwm      = kPublisherControlHighWaterMark,
        .pollTimeout = config.pollTimeout,
    })
{
}

ConversationNode::~ConversationNode()
{
    if (messageRouter_.isRunning()) {
        stop();
    }
    if (inferencer_) {
        inferencer_->unloadModel();
    }
}

// ---------------------------------------------------------------------------
// configureTransport
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ConversationNode::configureTransport()
{
    OE_ZONE_SCOPED;
    OeLogger::instance().setModule(config_.moduleName);
    OE_LOG_INFO("init_start: model_dir={}, variant={}", config_.modelDir, config_.modelVariant);

    messageRouter_.subscribe(
        config_.daemonSubPort,
        kZmqTopicConversationPrompt,
        /*conflate=*/false,
        [this](const nlohmann::json& msg) { handlePrompt(msg); });

    messageRouter_.subscribe(
        config_.uiCommandSubPort,
        kZmqTopicUiCommand,
        /*conflate=*/false,
        [this](const nlohmann::json& msg) { handleUiCommand(msg); });

    // ── Video conversation: subscribe to toggle commands from daemon ──
    messageRouter_.subscribe(
        config_.daemonSubPort,
        kZmqTopicVideoConversation,
        /*conflate=*/true,
        [this](const nlohmann::json& msg) { handleVideoConversationToggle(msg); });

    // ── Video conversation: open video ingest SHM (read-only) ─────────
    if (inferencer_ && inferencer_->supportsNativeVision()) {
        const std::size_t videoShmSize =
            ShmCircularBuffer<ShmVideoHeader>::segmentSize(
                kCircularBufferSlotCount, kMaxBgr24FrameBytes);
        auto shmResult = oe::shm::openConsumerWithRetry(
            config_.videoShmName, videoShmSize, "conversation_video", /*maxAttempts=*/3);
        if (!shmResult) {
            OE_LOG_WARN("OE-CONV-3011: video SHM init failed ({}), video conversation disabled",
                       shmResult.error());
        } else {
            videoShm_ = std::move(*shmResult);
            // Allocate pinned staging buffer for DMA-capable H2D transfer.
            // Falls back to non-pinned path if CUDA is unavailable (e.g. tests).
            try {
                videoStagingBuffer_ = std::make_unique<PinnedStagingBuffer>(kMaxBgr24FrameBytes);
            } catch (...) {
                OE_LOG_WARN("video_pinned_buffer_unavailable: using non-pinned path");
            }
            OE_LOG_INFO("video_shm_ready: shm={}, size={}", config_.videoShmName, videoShmSize);
        }

        // Screen SHM — opened eagerly since screen_ingest is always-on.
        // If screen_ingest hasn't written yet, the SHM won't exist — that's OK,
        // readLatestBgrFrame returns null view and we fall back to text-only.
        const std::size_t screenShmSize =
            ShmCircularBuffer<ShmVideoHeader>::segmentSize(
                kCircularBufferSlotCount, kMaxBgr24FrameBytes);
        auto screenResult = oe::shm::openConsumerWithRetry(
            config_.screenShmName, screenShmSize, "conversation_screen", /*maxAttempts=*/1);
        if (screenResult) {
            screenShm_ = std::move(*screenResult);
            OE_LOG_INFO("screen_shm_ready: shm={}", config_.screenShmName);
        } else {
            OE_LOG_INFO("screen_shm_not_available_yet: will retry on first screen toggle");
        }
    }

    // ── Native STT: subscribe to audio from AudioIngest ─────────────
    if (inferencer_ && inferencer_->supportsNativeStt()) {
        audioAccumulator_ = std::make_unique<AudioAccumulator>(
            AudioAccumulator::Config{
                .shmName                = config_.audioShmName,
                .maxAccumulationSamples = config_.maxAudioSamples,
            });
        auto initResult = audioAccumulator_->initialize();
        if (!initResult) {
            OE_LOG_WARN("OE-CONV-3010: audio SHM init failed ({}), native STT disabled",
                       initResult.error());
            audioAccumulator_.reset();
        } else {
            messageRouter_.subscribe(
                config_.audioSubPort,
                "audio_chunk",
                /*conflate=*/true,
                [this](const nlohmann::json& msg) { handleAudioChunk(msg); });

            messageRouter_.subscribe(
                config_.audioSubPort,
                "vad_status",
                /*conflate=*/false,
                [this](const nlohmann::json& msg) { handleVadStatus(msg); });

            OE_LOG_INFO("native_stt_enabled: shm={}, audio_port={}",
                       config_.audioShmName, config_.audioSubPort);
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// loadInferencer
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ConversationNode::loadInferencer()
{
    // Wire up heartbeat callback so generate() can keep the daemon watchdog
    // alive while blocking the poll loop for token streaming.
    inferencer_->setHeartbeatCallback([this]() {
        messageRouter_.publishHeartbeat();
    });

    auto loadResult = inferencer_->loadModel(config_.modelDir);
    if (!loadResult) {
        return tl::unexpected(
            std::format("Conversation inferencer load failed: {}", loadResult.error()));
    }

    OE_LOG_INFO("init_complete: inferencer={}, native_stt={}, native_tts={}, native_vision={}",
        inferencer_->name(), inferencer_->supportsNativeStt(),
        inferencer_->supportsNativeTts(), inferencer_->supportsNativeVision());
    OE_LOG_INFO("OE-CONV-1004: TTS sidecar {}",
        inferencer_->supportsNativeTts() ? "not required" : "required");

    return {};
}

// ---------------------------------------------------------------------------
// onPublishReady
// ---------------------------------------------------------------------------

void ConversationNode::onPublishReady()
{
    if (config_.degraded) {
        messageRouter_.publish(kZmqTopicModuleReady, {
            {"v",        kConversationSchemaVersion},
            {"type",     std::string(kZmqTopicModuleReady)},
            {"module",   config_.moduleName},
            {"pid",      static_cast<int>(::getpid())},
            {"degraded", true},
        });
    } else {
        messageRouter_.publishModuleReady();
    }
}

// ---------------------------------------------------------------------------
// onBeforeStop
// ---------------------------------------------------------------------------

void ConversationNode::onBeforeStop() noexcept
{
    if (inferencer_) {
        inferencer_->cancel();
    }
}

// ---------------------------------------------------------------------------
// handlePrompt
// ---------------------------------------------------------------------------

void ConversationNode::handlePrompt(const nlohmann::json& msg)
{
    OE_ZONE_SCOPED;
    if (!msg.contains("type") || msg["type"] != "conversation_prompt") {
        return;
    }

    const auto generationStartTime = std::chrono::steady_clock::now();
    OE_LOG_INFO("prompt_received");

    // The daemon assembles a "messages" array (chat format).
    // Convert to a ChatML prompt string for the tokenizer.
    std::string prompt;
    if (msg.contains("text") && !msg["text"].get<std::string>().empty()) {
        // Legacy: flat text field (tests, direct injection).
        prompt = msg["text"].get<std::string>();
    } else if (msg.contains("messages") && msg["messages"].is_array()) {
        // Gemma-4 chat format: <|turn>role\ncontent<turn|> (token ids 105/106).
        // NOT the Gemma-3 <start_of_turn>/<end_of_turn> pair — those are absent
        // from this tokenizer's vocabulary and tokenize as garbage BPE subwords.
        // Gemma uses "model" for assistant and has no system role — system
        // instructions are prepended to the first user turn.
        std::string systemText;
        bool firstUserEmitted = false;
        for (const auto& m : msg["messages"]) {
            auto role       = m.value("role", "user");
            const auto content = m.value("content", "");
            if (role == "system") {
                systemText += content + "\n";
                continue;
            }
            if (role == "assistant") role = "model";
            std::string turnContent = content;
            if (role == "user" && !firstUserEmitted && !systemText.empty()) {
                turnContent   = systemText + turnContent;
                firstUserEmitted = true;
            } else if (role == "user") {
                firstUserEmitted = true;
            }
            prompt += "<|turn>" + role + "\n" + turnContent + "<turn|>\n";
        }
        // Open the model turn for generation.
        prompt += "<|turn>model\n";
    }
    if (prompt.empty()) {
        OE_LOG_WARN("OE-CONV-3002: conversation_prompt has no text or messages");
        return;
    }

    const int sequenceId = msg.value("sequence_id", 0);
    const GenerationParams params = parseGenerationParams(msg, config_.generationDefaults);

    OE_LOG_DEBUG("prompt_details: prompt_len={}, max_tokens={}, temperature={}, top_p={}, seq_id={}",
        prompt.size(), params.maxTokens, params.temperature, params.topP, sequenceId);

    // Reset sentence boundary state for each new generation.
    prevTokenEndedSentence_ = false;

    bool timeToFirstTokenMeasured = false;
    int outputTokenCount = 0;

    auto tokenCallback =
        [this, &timeToFirstTokenMeasured, &generationStartTime, &sequenceId, &outputTokenCount]
        (std::string_view token, bool /*sentenceBoundary*/, bool done) {
            // Override sentence boundary: use our own tracking for consistency.
            const bool boundary = prevTokenEndedSentence_;
            prevTokenEndedSentence_ = endsSentence(token);

            sendResponseToken(token, done, boundary, sequenceId);
            ++outputTokenCount;

            // Keep heartbeat alive during generation — generate() blocks the
            // poll loop so the normal heartbeat path never fires.
            messageRouter_.publishHeartbeat();

            if (!timeToFirstTokenMeasured && (!token.empty() || done)) {
                timeToFirstTokenMeasured = true;
                const auto firstTokenTime = std::chrono::steady_clock::now();
                const double ttftMs =
                    std::chrono::duration<double, std::milli>(
                        firstTokenTime - generationStartTime).count();
                OE_LOG_INFO("ttft: ttft_ms={:.2f}", ttftMs);
            }
        };

    // Publish heartbeat before entering generate() — covers the TTFT gap
    // (time between prompt dispatch and first token) which blocks the poll loop.
    messageRouter_.publishHeartbeat();

    // ── Video conversation: read latest frame from SHM and call generateWithVideo ──
    tl::expected<void, std::string> result;
    ShmMapping* activeShmPtr = (activeVideoSource_ == VideoSource::kScreen && screenShm_)
        ? screenShm_.get()
        : videoShm_.get();
    if (videoConversationEnabled_ && activeShmPtr &&
        inferencer_->supportsNativeVision()) {
        auto frameView = readLatestBgrFrame(*activeShmPtr);
        if (frameView.data != nullptr && frameView.width > 0 && frameView.height > 0) {
            const std::size_t frameBytes =
                static_cast<std::size_t>(frameView.width) * frameView.height * 3;
            const uint8_t* frameData = frameView.data;

            // Use pinned staging buffer when available (DMA-capable H2D transfer).
            // Falls back to direct SHM pointer when CUDA is unavailable.
            if (videoStagingBuffer_) {
                std::memcpy(videoStagingBuffer_->data(), frameView.data, frameBytes);
                frameData = videoStagingBuffer_->data();
            }
            OE_LOG_DEBUG("video_frame_attached: {}x{}, {} bytes",
                frameView.width, frameView.height, frameBytes);
            result = inferencer_->generateWithVideo(
                prompt,
                std::span<const uint8_t>(frameData, frameBytes),
                frameView.width,
                frameView.height,
                params,
                tokenCallback);
        } else {
            OE_LOG_WARN("OE-CONV-3012: video frame unavailable, falling back to text-only");
            result = inferencer_->generate(prompt, params, tokenCallback);
        }
    } else {
        result = inferencer_->generate(prompt, params, tokenCallback);
    }

    if (!result) {
        OE_LOG_WARN("OE-CONV-4003: generate_error: reason={}", result.error());
        sendResponseToken("", /*done=*/true, /*sentenceBoundary=*/false, sequenceId);
    }

    const auto generationEndTime = std::chrono::steady_clock::now();
    const double totalMs =
        std::chrono::duration<double, std::milli>(
            generationEndTime - generationStartTime).count();
    const double tokensPerSecond = (totalMs > 0.0) ? (outputTokenCount * 1000.0 / totalMs) : 0.0;
    OE_LOG_INFO("OE-CONV-2005: generation_complete: {} tokens in {:.1f}ms ({:.1f} tok/s)",
        outputTokenCount, totalMs, tokensPerSecond);
}

// ---------------------------------------------------------------------------
// handleUiCommand
// ---------------------------------------------------------------------------

void ConversationNode::handleUiCommand(const nlohmann::json& msg)
{
    const auto action = parseUiAction(msg.value("action", std::string{}));
    switch (action) {
    case UiAction::kCancelGeneration:
        OE_LOG_INFO("cancel_requested");
        inferencer_->cancel();
        break;
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
// sendResponseToken
// ---------------------------------------------------------------------------

void ConversationNode::sendResponseToken(
    std::string_view token,
    bool             done,
    bool             sentenceBoundary,
    int              sequenceId)
{
    const bool hasAudio = inferencer_->supportsNativeTts();

    static thread_local ConversationResponseMsg payload;
    payload.token             = std::string(token);
    payload.finished          = done;
    payload.sentence_boundary = sentenceBoundary;
    payload.sequence_id       = sequenceId;
    payload.has_audio         = hasAudio;

    OE_LOG_DEBUG("conversation_token: len={}, done={}, sentence_boundary={}, has_audio={}",
        token.size(), done, sentenceBoundary, hasAudio);

    messageRouter_.publish(kZmqTopicConversationResponse, nlohmann::json(payload));
}

// ---------------------------------------------------------------------------
// parseGenerationParams
// ---------------------------------------------------------------------------

GenerationParams ConversationNode::parseGenerationParams(
    const nlohmann::json& msg, const GenerationParams& defaults)
{
    GenerationParams params = defaults;

    if (msg.contains("generation_params") && msg["generation_params"].is_object()) {
        const auto& gp = msg["generation_params"];
        params.temperature = gp.value("temperature", params.temperature);
        params.topP        = gp.value("top_p", params.topP);
        params.maxTokens   = gp.value("max_tokens", params.maxTokens);
    } else {
        OE_LOG_WARN("OE-CONV-3003: generation_params missing from conversation_prompt: using defaults");
    }

    // Vision: top-level image_path carries a path on disk (written by ws_bridge)
    // for vision-capable conversation models.
    params.imagePath = msg.value("image_path", std::string{});

    // Clamp to valid ranges and log if values were adjusted.
    GenerationParams original = params;
    params.clamp();

    if (params.temperature != original.temperature) {
        OE_LOG_DEBUG("OE-CONV-2004: Generation params clamped: temperature {} -> {}",
            original.temperature, params.temperature);
    }
    if (params.topP != original.topP) {
        OE_LOG_DEBUG("OE-CONV-2004: Generation params clamped: top_p {} -> {}",
            original.topP, params.topP);
    }
    if (params.maxTokens != original.maxTokens) {
        OE_LOG_DEBUG("OE-CONV-2004: Generation params clamped: max_tokens {} -> {}",
            original.maxTokens, params.maxTokens);
    }

    return params;
}

// ---------------------------------------------------------------------------
// Video conversation toggle
// ---------------------------------------------------------------------------

void ConversationNode::handleVideoConversationToggle(const nlohmann::json& msg)
{
    const bool enabled = msg.value("enabled", false);
    const auto source  = msg.value("source", "camera");

    if (enabled && !inferencer_->supportsNativeVision()) {
        OE_LOG_WARN("OE-CONV-3013: video conversation requested but inferencer {} does not support vision",
                   inferencer_->name());
        return;
    }

    videoConversationEnabled_ = enabled;

    if (source == "screen") {
        activeVideoSource_ = VideoSource::kScreen;
        // Lazy retry if screen SHM wasn't available at init.
        if (!screenShm_) {
            const std::size_t screenShmSize =
                ShmCircularBuffer<ShmVideoHeader>::segmentSize(
                    kCircularBufferSlotCount, kMaxBgr24FrameBytes);
            auto result = oe::shm::openConsumerWithRetry(
                config_.screenShmName, screenShmSize,
                "conversation_screen", /*maxAttempts=*/3);
            if (result) {
                screenShm_ = std::move(*result);
                OE_LOG_INFO("screen_shm_opened_on_toggle: shm={}", config_.screenShmName);
            } else {
                OE_LOG_WARN("screen_shm_open_failed: {}", result.error());
            }
        }
    } else {
        activeVideoSource_ = VideoSource::kCamera;
    }

    OE_LOG_INFO("video_source_switched: enabled={}, source={}", enabled, source);
}

// ---------------------------------------------------------------------------
// Native STT handlers
// ---------------------------------------------------------------------------

void ConversationNode::handleAudioChunk(const nlohmann::json& msg)
{
    OE_ZONE_SCOPED;
    if (!audioAccumulator_) return;

    audioAccumulator_->appendFromShm(msg);

    if (audioAccumulator_->isFull()) {
        runTranscription();
    }
}

void ConversationNode::handleVadStatus(const nlohmann::json& msg)
{
    const bool speaking = msg.value("speaking", true);
    if (!speaking && audioAccumulator_ && !audioAccumulator_->empty()) {
        runTranscription();
    }
}

void ConversationNode::runTranscription()
{
    OE_ZONE_SCOPED;
    if (!audioAccumulator_ || audioAccumulator_->empty()) return;

    transcriptionBuffer_.clear();
    audioAccumulator_->flush(transcriptionBuffer_);

    OE_LOG_INFO("native_stt_transcribing: samples={}", transcriptionBuffer_.size());

    auto result = inferencer_->transcribe(
        std::span<const float>(transcriptionBuffer_),
        kSttInputSampleRateHz);

    if (!result) {
        OE_LOG_WARN("OE-CONV-4010: transcription_error: {}", result.error());
        return;
    }

    const auto& text = result.value();
    if (text.empty() || text.find_first_not_of(" \t\n\r") == std::string::npos) {
        OE_LOG_DEBUG("transcription_empty_suppressed");
        return;
    }

    publishTranscription(text);
}

void ConversationNode::publishTranscription(const std::string& text)
{
    messageRouter_.publish("transcription", nlohmann::json{
        {"v",          kConversationSchemaVersion},
        {"type",       "transcription"},
        {"text",       text},
        {"lang",       "en"},
        {"source",     "native_stt"},
        {"confidence", 1.0},
    });
    OE_LOG_INFO("native_transcription_published: text_len={}", text.size());
}

