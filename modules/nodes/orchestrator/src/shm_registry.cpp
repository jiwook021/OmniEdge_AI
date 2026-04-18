#include "orchestrator/shm_registry.hpp"

#include <sys/mman.h>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


static const std::vector<std::string> kEmptySegments;

void ShmRegistry::registerSegment(std::string_view module, std::string_view shmName)
{
    auto it = segments_.find(module);
    if (it == segments_.end()) {
        it = segments_.emplace(std::string(module), std::vector<std::string>{}).first;
    }
    auto& vec = it->second;
    // Avoid duplicates (idempotent registration)
    for (const auto& existing : vec) {
        if (existing == shmName) return;
    }
    vec.emplace_back(shmName);
    OE_LOG_DEBUG("shm_registry: registered segment={} for module={}", shmName, module);
}

void ShmRegistry::cleanupForModule(std::string_view module)
{
    OE_ZONE_SCOPED;
    auto cleanupIt = segments_.find(module);
    if (cleanupIt == segments_.end()) return;

    for (const auto& seg : cleanupIt->second) {
        if (::shm_unlink(seg.c_str()) == 0) {
            OE_LOG_INFO("shm_cleanup: unlinked {} (module {} crashed)", seg, module);
        } else {
            // ENOENT is expected if the module never created the segment
            // (e.g. crashed during init before shm_open).
            OE_LOG_DEBUG("shm_cleanup: shm_unlink({}) failed: {} (module {})",
                         seg, std::strerror(errno), module);
        }
    }
}

const std::vector<std::string>& ShmRegistry::segmentsForModule(
    std::string_view module) const
{
    auto segIt = segments_.find(module);
    if (segIt == segments_.end()) return kEmptySegments;
    return segIt->second;
}

