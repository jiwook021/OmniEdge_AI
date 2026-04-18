#include "statemachine/prompt_assembler.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <system_error>

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
    const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    history_.push_back({userText, assistantText, nowNs});
    persistIfEnabled();
}

void PromptAssembler::clearHistory()
{
    history_.clear();
    persistIfEnabled();
}

nlohmann::json PromptAssembler::historyAsJson() const
{
    nlohmann::json turns = nlohmann::json::array();
    for (const auto& turn : history_) {
        turns.push_back({{"role", "user"},      {"content", turn.userText},      {"ts_ns", turn.tsNs}});
        turns.push_back({{"role", "assistant"}, {"content", turn.assistantText}, {"ts_ns", turn.tsNs}});
    }
    return turns;
}

bool PromptAssembler::saveToFile(const std::string& path) const
{
    if (path.empty()) return false;

    nlohmann::json turns = nlohmann::json::array();
    for (const auto& turn : history_) {
        turns.push_back({
            {"user",      turn.userText},
            {"assistant", turn.assistantText},
            {"ts_ns",     turn.tsNs},
        });
    }
    nlohmann::json doc = {
        {"v",     1},
        {"turns", std::move(turns)},
    };

    std::error_code ec;
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path(), ec);

    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            OE_LOG_WARN("prompt_persist_open_failed: path={}", tmp);
            return false;
        }
        out << doc.dump();
        out.flush();
        if (!out.good()) {
            OE_LOG_WARN("prompt_persist_write_failed: path={}", tmp);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        OE_LOG_WARN("prompt_persist_rename_failed: path={} err={}", path, ec.message());
        return false;
    }
    return true;
}

bool PromptAssembler::loadFromFile(const std::string& path)
{
    if (path.empty()) return false;
    if (!std::filesystem::exists(path)) {
        OE_LOG_INFO("prompt_persist_no_file: path={} (first run)", path);
        return true;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        OE_LOG_WARN("prompt_persist_read_failed: path={}", path);
        return false;
    }

    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const std::exception& ex) {
        OE_LOG_WARN("prompt_persist_parse_failed: path={} err={}", path, ex.what());
        return false;
    }

    if (!doc.contains("turns") || !doc["turns"].is_array()) {
        OE_LOG_WARN("prompt_persist_schema_invalid: path={}", path);
        return false;
    }

    std::deque<ConversationTurn> loaded;
    for (const auto& turn : doc["turns"]) {
        loaded.push_back({
            turn.value("user",      std::string{}),
            turn.value("assistant", std::string{}),
            turn.value("ts_ns",     int64_t{0}),
        });
    }
    history_ = std::move(loaded);
    OE_LOG_INFO("prompt_persist_loaded: path={} turns={}", path, history_.size());
    return true;
}

void PromptAssembler::persistIfEnabled() const
{
    if (!persistPath_.empty()) {
        saveToFile(persistPath_);
    }
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

