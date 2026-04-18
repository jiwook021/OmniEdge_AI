#include "statemachine/session_persistence.hpp"

#include <filesystem>
#include <format>

#include <nlohmann/json.hpp>

#include "common/file.hpp"
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


SessionPersistence::SessionPersistence(Config config)
    : config_(std::move(config))
    , lastSaveTime_(std::chrono::steady_clock::now())
{
}

tl::expected<void, std::string> SessionPersistence::save(const nlohmann::json& state)
{
    auto result = atomicWrite(config_.filePath, state.dump(2));
    if (!result) {
        OE_LOG_ERROR("session_save: {}", result.error());
        return tl::make_unexpected(result.error());
    }

    lastSaveTime_ = std::chrono::steady_clock::now();
    return {};
}

tl::expected<nlohmann::json, std::string> SessionPersistence::load()
{
    if (!exists(config_.filePath))
        return tl::make_unexpected("session file does not exist");

    std::error_code ec;
    auto writeTime = std::filesystem::last_write_time(config_.filePath, ec);
    if (ec)
        return tl::make_unexpected(std::format("cannot stat: {}", ec.message()));

    auto fileAge = std::chrono::duration_cast<std::chrono::seconds>(
        std::filesystem::file_time_type::clock::now() - writeTime);
    if (fileAge >= config_.maxStaleness)
        return tl::make_unexpected(std::format("stale ({}s > {}s)", fileAge.count(), config_.maxStaleness.count()));

    auto textResult = readText(config_.filePath);
    if (!textResult)
        return tl::make_unexpected(textResult.error());

    nlohmann::json state;
    try {
        state = nlohmann::json::parse(*textResult);
    } catch (const nlohmann::json::exception& ex) {
        return tl::make_unexpected(std::format("corrupt JSON: {}", ex.what()));
    }

    OE_LOG_INFO("session_loaded: age={}s", fileAge.count());
    return state;
}

bool SessionPersistence::shouldPeriodicSave() const
{
    return (std::chrono::steady_clock::now() - lastSaveTime_) >= config_.periodicSaveInterval;
}

void SessionPersistence::resetSaveTimer()
{
    lastSaveTime_ = std::chrono::steady_clock::now();
}

