// conversation_node.cpp -- ConversationNode full implementation

#include "conversation/conversation_node.hpp"

#include <chrono>
#include <format>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "common/constants/conversation_constants.hpp"
#include "common/oe_tracy.hpp"
#include "common/ui_action.hpp"
#include "common/oe_logger.hpp"


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
// onConfigure
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ConversationNode::onConfigure()
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

    return {};
}

// ---------------------------------------------------------------------------
// onLoadInferencer
// ---------------------------------------------------------------------------

tl::expected<void, std::string> ConversationNode::onLoadInferencer()
{
    auto loadResult = inferencer_->loadModel(config_.modelDir);
    if (!loadResult) {
        return tl::unexpected(
            std::format("Conversation inferencer load failed: {}", loadResult.error()));
    }

    OE_LOG_INFO("init_complete: inferencer={}, native_audio={}",
        inferencer_->name(), inferencer_->supportsNativeAudio());
    OE_LOG_INFO("OE-CONV-1004: supportsNativeAudio() = {}, TTS sidecar {}",
        inferencer_->supportsNativeAudio(),
        inferencer_->supportsNativeAudio() ? "not required" : "required");

    return {};
}

// ---------------------------------------------------------------------------
// onPublishReady
// ---------------------------------------------------------------------------

void ConversationNode::onPublishReady()
{
    if (config_.degraded) {
        messageRouter_.publish("module_ready", {
            {"v",        kConversationSchemaVersion},
            {"type",     "module_ready"},
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

    // The daemon assembles the full prompt -- node receives ready-to-process text.
    const std::string prompt = msg.value("text", std::string{});
    if (prompt.empty()) {
        OE_LOG_WARN("OE-CONV-3002: conversation_prompt has empty text field");
        return;
    }

    const int sequenceId = msg.value("sequence_id", 0);
    const GenerationParams params = parseGenerationParams(msg, config_.generationDefaults);

    OE_LOG_DEBUG("prompt_details: prompt_len={}, max_tokens={}, temperature={}, top_p={}, seq_id={}",
        prompt.size(), params.maxTokens, params.temperature, params.topP, sequenceId);

    // Reset sentence boundary state for each new generation.
    prevTokenEndedSentence_ = false;

    bool firstTokenReceived = false;
    int tokenCount = 0;

    auto result = inferencer_->generate(
        prompt,
        params,
        [this, &firstTokenReceived, &generationStartTime, &sequenceId, &tokenCount]
        (std::string_view token, bool /*sentenceBoundary*/, bool done) {
            // Override sentence boundary: use our own tracking for consistency.
            const bool boundary = prevTokenEndedSentence_;
            prevTokenEndedSentence_ = endsSentence(token);

            publishToken(token, done, boundary, sequenceId);
            ++tokenCount;

            if (!firstTokenReceived && (!token.empty() || done)) {
                firstTokenReceived = true;
                const auto firstTokenTime = std::chrono::steady_clock::now();
                const double ttftMs =
                    std::chrono::duration<double, std::milli>(
                        firstTokenTime - generationStartTime).count();
                OE_LOG_INFO("ttft: ttft_ms={:.2f}", ttftMs);
            }
        });

    if (!result) {
        OE_LOG_WARN("OE-CONV-4003: generate_error: reason={}", result.error());
        publishToken("", /*done=*/true, /*sentenceBoundary=*/false, sequenceId);
    }

    const auto generationEndTime = std::chrono::steady_clock::now();
    const double totalMs =
        std::chrono::duration<double, std::milli>(
            generationEndTime - generationStartTime).count();
    const double tokensPerSecond = (totalMs > 0.0) ? (tokenCount * 1000.0 / totalMs) : 0.0;
    OE_LOG_INFO("OE-CONV-2005: generation_complete: {} tokens in {:.1f}ms ({:.1f} tok/s)",
        tokenCount, totalMs, tokensPerSecond);
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
// publishToken
// ---------------------------------------------------------------------------

void ConversationNode::publishToken(
    std::string_view token,
    bool             done,
    bool             sentenceBoundary,
    int              sequenceId)
{
    const bool hasAudio = inferencer_->supportsNativeAudio();

    static thread_local nlohmann::json payload = {
        {"v",                 kConversationSchemaVersion},
        {"type",              kZmqTopicConversationResponse},
        {"token",             ""},
        {"done",              false},
        {"sentence_boundary", false},
        {"sequence_id",       0},
        {"has_audio",         false},
    };
    payload["token"]             = std::string(token);
    payload["done"]              = done;
    payload["sentence_boundary"] = sentenceBoundary;
    payload["sequence_id"]       = sequenceId;
    payload["has_audio"]         = hasAudio;

    OE_LOG_DEBUG("conversation_token: len={}, done={}, sentence_boundary={}, has_audio={}",
        token.size(), done, sentenceBoundary, hasAudio);

    messageRouter_.publish(kZmqTopicConversationResponse, payload);
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

