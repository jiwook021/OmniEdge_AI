#include "statemachine/state_machine.hpp"
#include <array>
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
std::string_view stateName(StateIndex s) noexcept {
    static constexpr std::array<std::string_view, kStateCount> kNames{{
        "idle", "listening", "processing",
        "speaking", "interrupted", "error_recovery",
        "image_transform", "super_resolution",
    }};
    auto idx = static_cast<std::size_t>(s);
    return (idx < kNames.size()) ? kNames[idx] : "unknown";
}
void logTransition(StateIndex from, std::string_view event, StateIndex to) {
    OE_LOG_INFO("fsm: {} + {} -> {}", stateName(from), event, stateName(to));
}
void logUnhandledEvent(StateIndex current, std::string_view event) {
    OE_LOG_WARN("fsm_unhandled: state={}, event={} — no transition defined",
                stateName(current), event);
}
void StateMachine::forceState(StateIndex idx) noexcept {
    switch (idx) {
    case StateIndex::kIdle: state_ = fsm::Idle{}; break;
    case StateIndex::kListening: state_ = fsm::Listening{}; break;
    case StateIndex::kProcessing: state_ = fsm::Processing{}; break;
    case StateIndex::kSpeaking: state_ = fsm::Speaking{}; break;
    case StateIndex::kInterrupted: state_ = fsm::Interrupted{}; break;
    case StateIndex::kErrorRecovery: state_ = fsm::ErrorRecovery{}; break;
    case StateIndex::kImageTransform: state_ = fsm::ImageTransform{}; break;
    case StateIndex::kSuperResolution: state_ = fsm::SuperResolution{}; break;
    }
}
