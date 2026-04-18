#pragma once

#include <chrono>
#include <string>

#include <nlohmann/json.hpp>

#include <tl/expected.hpp>
#include "common/runtime_defaults.hpp"


/// Atomic session state reader/writer (write -> rename).
/// NOT thread-safe — called from orchestrator poll loop only.
class SessionPersistence {
public:
    struct Config {
        std::string filePath{"session_state.json"};
        std::chrono::seconds periodicSaveInterval{kSessionSaveIntervalS};
        std::chrono::seconds maxStaleness{kSessionMaxStalenessS};
    };

    explicit SessionPersistence(Config config);
    SessionPersistence(const SessionPersistence&)            = delete;
    SessionPersistence& operator=(const SessionPersistence&) = delete;

    [[nodiscard]] tl::expected<void, std::string> save(const nlohmann::json& state);
    [[nodiscard]] tl::expected<nlohmann::json, std::string> load();
    [[nodiscard]] bool shouldPeriodicSave() const;
    void resetSaveTimer();

private:
    Config config_;
    std::chrono::steady_clock::time_point lastSaveTime_;
};

