#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/string_hash.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — SHM Lease Registry
//
// Daemon-level registry that tracks which POSIX SHM segments belong to
// which module.  When the watchdog detects a module crash, it calls
// cleanupForModule() to shm_unlink all segments the crashed module
// produced — preventing stale /dev/shm/oe.* accumulation.
//
// Without this, a producer crash leaves the SHM segment alive.  The next
// spawn's O_CREAT|O_TRUNC succeeds (overwriting the old data), but
// segments from *other* crash-restart cycles accumulate indefinitely.
// ---------------------------------------------------------------------------


/// Daemon-side SHM segment registry with crash-cleanup lease protocol.
///
/// NOT thread-safe — all calls from orchestrator poll loop (single-threaded).
class ShmRegistry {
public:
    ShmRegistry() = default;

    /// Register a SHM segment name as belonging to a module.
    /// Called during loadModuleConfigFromYaml() or module spawn.
    void registerSegment(std::string_view module, std::string_view shmName);

    /// Unlink all SHM segments owned by a crashed module.
    /// Called from watchdogPoll() before restarting the module.
    void cleanupForModule(std::string_view module);

    /// Read-only access to segments registered for a module.
    [[nodiscard]] const std::vector<std::string>& segmentsForModule(
        std::string_view module) const;

    /// Number of modules with registered segments.
    [[nodiscard]] std::size_t moduleCount() const noexcept { return segments_.size(); }

private:
    std::unordered_map<std::string, std::vector<std::string>, StringHash, std::equal_to<>> segments_;
};

