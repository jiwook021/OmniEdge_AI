#pragma once

#include <deque>
#include <string>

#include <nlohmann/json.hpp>

#include "common/runtime_defaults.hpp"


/// Assembles multi-modal context into a single LLM prompt.
/// Combines system prompt + face identity + scene description + conversation
/// history + current utterance. Token budget managed with sliding-window eviction.
/// NOT thread-safe — called from orchestrator poll loop only.
class PromptAssembler {
public:
    struct Config {
        int maxContextTokens{kPromptMaxContextTokens};
        int systemPromptTokens{kPromptSystemTokens};
        int dynamicContextTokens{kPromptDynamicContextTokens};
        int maxUserTurnTokens{kPromptMaxUserTurnTokens};
        int generationHeadroom{kPromptGenerationHeadroom};
        std::string systemPrompt;
    };

    explicit PromptAssembler(Config config);
    PromptAssembler(const PromptAssembler&)            = delete;
    PromptAssembler& operator=(const PromptAssembler&) = delete;

    void setFaceIdentity(const std::string& name, float confidence);
    void setSceneDescription(const std::string& description);

    /// Build full prompt JSON for the llm_prompt topic.
    [[nodiscard]] nlohmann::json assemble(const std::string& userUtterance);

    void addToHistory(const std::string& userText, const std::string& assistantText);
    void clearHistory();
    [[nodiscard]] std::size_t historySize() const noexcept { return history_.size(); }

private:
    Config config_;
    std::string faceIdentity_;
    std::string sceneDescription_;

    struct ConversationTurn {
        std::string userText;
        std::string assistantText;
    };
    std::deque<ConversationTurn> history_;

    [[nodiscard]] static int estimateTokens(const std::string& text);
    void evictHistory(int availableTokens);
};

