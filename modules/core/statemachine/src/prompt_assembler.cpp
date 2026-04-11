#include "statemachine/prompt_assembler.hpp"

#include <format>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


PromptAssembler::PromptAssembler(Config config)
    : config_(std::move(config))
{
}

void PromptAssembler::setFaceIdentity(const std::string& name, float confidence)
{
    if (name.empty()) {
        faceIdentity_.clear();
        return;
    }
    faceIdentity_ = std::format("The user is {} (confidence: {:.2f}).", name, confidence);
}

void PromptAssembler::setSceneDescription(const std::string& description)
{
    sceneDescription_ = description;
}

nlohmann::json PromptAssembler::assemble(const std::string& userUtterance)
{
    nlohmann::json messages = nlohmann::json::array();

    // System prompt + dynamic context (identity, scene)
    std::string systemContent = config_.systemPrompt;
    if (!faceIdentity_.empty())     systemContent += "\n" + faceIdentity_;
    if (!sceneDescription_.empty()) systemContent += "\nYou can see: " + sceneDescription_;

    messages.push_back({{"role", "system"}, {"content", systemContent}});

    // Conversation history (sliding-window eviction)
    int usedTokens = estimateTokens(systemContent) +
                     estimateTokens(userUtterance) +
                     config_.generationHeadroom;

    if (usedTokens > config_.maxContextTokens) {
        OE_LOG_WARN("prompt_overflow: system+user tokens ({}) exceed context window ({}), truncating",
                    usedTokens, config_.maxContextTokens);
        // Truncate scene description to fit — keep at least system prompt + user utterance
        const int overflowTokens = usedTokens - config_.maxContextTokens;
        const int overflowChars = overflowTokens * 4; // rough estimate (reverse of estimateTokens)
        if (static_cast<int>(systemContent.size()) > overflowChars) {
            systemContent.resize(systemContent.size() - static_cast<std::size_t>(overflowChars));
        }
        usedTokens = estimateTokens(systemContent) + estimateTokens(userUtterance) + config_.generationHeadroom;
    }

    evictHistory(config_.maxContextTokens - usedTokens);

    for (const auto& turn : history_) {
        messages.push_back({{"role", "user"},      {"content", turn.userText}});
        messages.push_back({{"role", "assistant"}, {"content", turn.assistantText}});
    }

    // Current utterance
    messages.push_back({{"role", "user"}, {"content", userUtterance}});

    return {
        {"v",        kSchemaVersion},
        {"type",     "conversation_prompt"},
        {"messages", messages},
    };
}

void PromptAssembler::addToHistory(const std::string& userText, const std::string& assistantText)
{
    history_.push_back({userText, assistantText});
}

void PromptAssembler::clearHistory()
{
    history_.clear();
}

int PromptAssembler::estimateTokens(const std::string& text)
{
    if (text.empty()) return 0;
    return static_cast<int>(text.size()) / 4 + 1;
}

void PromptAssembler::evictHistory(int availableTokens)
{
    int totalHistoryTokens = 0;
    for (const auto& turn : history_) {
        totalHistoryTokens += estimateTokens(turn.userText) +
                              estimateTokens(turn.assistantText);
    }

    const auto before = history_.size();
    while (!history_.empty() && totalHistoryTokens > availableTokens) {
        totalHistoryTokens -= estimateTokens(history_.front().userText) +
                              estimateTokens(history_.front().assistantText);
        history_.pop_front();
    }

    if (history_.size() < before) {
        OE_LOG_INFO("history_evicted: removed={}, remaining={}", before - history_.size(), history_.size());
    }
}

