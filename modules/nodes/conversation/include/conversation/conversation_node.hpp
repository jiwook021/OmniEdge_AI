#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "conversation_inferencer.hpp"
#include "zmq/message_router.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/zmq_constants.hpp"


class ConversationInferencer;  // forward-declare

/**
 * @brief Unified conversation node (STT+LLM+TTS in one process).
 *
 * Subscribes to two ZMQ topics:
 *   - `conversation_prompt` (daemon PUB port 5571) -- fully assembled prompt
 *   - `ui_command`          (WebSocketBridge PUB port 5570) -- cancel_generation
 *
 * Publishes on one ZMQ topic:
 *   - `conversation_response` (own PUB port 5572) -- one message per token
 *
 * Token streaming protocol (v2):
 *   Each token emits a JSON message with:
 *     "done": false, "has_audio": bool, "sentence_boundary": bool
 *   The first token after '.', '?', or '!' carries sentence_boundary=true.
 *   The final message carries "done": true.
 *
 * Supports three model backends:
 *   - Qwen2.5-Omni-7B (has_audio=true, no TTS sidecar)
 *   - Qwen2.5-Omni-3B (has_audio=true, no TTS sidecar)
 *   - Gemma 4 E4B      (has_audio=false, TTS sidecar required)
 *
 * Thread safety:
 *   initialize() and run() must be called from the same thread.
 *   stop() is safe from any thread (including SIGTERM signal handler).
 */
class ConversationNode : public ModuleNodeBase<ConversationNode> {
    friend class ModuleNodeBase<ConversationNode>;

public:
    struct Config {
        std::string moduleName       = "conversation";
        int         pubPort          = kConversationModel;  ///< Own PUB port (5572)
        int         daemonSubPort    = kDaemon;             ///< Daemon PUB port to subscribe (5571)
        int         uiCommandSubPort = kWsBridge;           ///< WS Bridge PUB port to subscribe (5570)
        std::string modelDir;                                      ///< Model weights directory
        std::string modelVariant;                                  ///< "qwen_omni_7b", "qwen_omni_3b", "gemma_e4b"
        int         cudaStreamPriority = kCudaPriorityLlm;     ///< CUDA stream priority (-5)
        std::chrono::milliseconds pollTimeout{kPollTimeoutMs};
        bool        degraded{false};                               ///< true when on fallback engine

        /// INI-configurable generation defaults (fallbacks when frontend omits fields).
        GenerationParams generationDefaults{};

        /// @brief Validate raw config values.
        [[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
    };

    /**
     * @brief Construct with config and injected inferencer.
     *
     * @param config      Populated from YAML in main.cpp.
     * @param inferencer  Conversation inferencer. Pass MockConversationInferencer for tests.
     */
    explicit ConversationNode(const Config& config,
                              std::unique_ptr<ConversationInferencer> inferencer);

    // Non-copyable, non-movable -- owns CUDA context and ZMQ sockets.
    ConversationNode(const ConversationNode&)            = delete;
    ConversationNode& operator=(const ConversationNode&) = delete;

    ~ConversationNode();

    // ---- CRTP hooks (required) ------------------------------------------------

    [[nodiscard]] tl::expected<void, std::string> onConfigure();
    [[nodiscard]] tl::expected<void, std::string> onLoadInferencer();

    // ---- CRTP hooks (optional) ------------------------------------------------

    void onBeforeStop() noexcept;
    void onPublishReady();

    // ---- Accessors ------------------------------------------------------------

    [[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
    [[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

    /**
     * @brief Parse GenerationParams from a conversation_prompt JSON payload.
     *
     * Uses @p defaults for missing fields; clamps to valid ranges.
     * Static pure function -- testable without a node instance.
     *
     * @param msg      JSON message containing optional "generation_params" object.
     * @param defaults Fallback values from INI [generation] section.
     */
    [[nodiscard]] static GenerationParams parseGenerationParams(
        const nlohmann::json& msg,
        const GenerationParams& defaults = GenerationParams{});

private:
    Config                                  config_;
    std::unique_ptr<ConversationInferencer> inferencer_;
    MessageRouter                           messageRouter_;

    // Sentence boundary tracking across the token stream.
    bool prevTokenEndedSentence_{false};

    // ---- Private helpers ---------------------------------------------------

    void handlePrompt(const nlohmann::json& msg);
    void handleUiCommand(const nlohmann::json& msg);

    void publishToken(std::string_view token,
                      bool             done,
                      bool             sentenceBoundary,
                      int              sequenceId);
};

