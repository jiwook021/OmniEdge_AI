#include "graph/edge_state_machine.hpp"

namespace core::graph {

EdgeStateMachine::EdgeStateMachine(std::string edgeId)
    : edgeId_(std::move(edgeId)) {}

EdgeState EdgeStateMachine::current() const { return state_; }

const std::string& EdgeStateMachine::edgeId() const { return edgeId_; }

tl::expected<void, EdgeStateError> EdgeStateMachine::beginConnecting() {
    switch (state_) {
        case EdgeState::kIdle:
            state_ = EdgeState::kConnecting;
            return {};
        default:
            return tl::unexpected(EdgeStateError::kInvalidTransition);
    }
}

tl::expected<void, EdgeStateError> EdgeStateMachine::confirmActive() {
    switch (state_) {
        case EdgeState::kConnecting:
            state_ = EdgeState::kActive;
            return {};
        default:
            return tl::unexpected(EdgeStateError::kInvalidTransition);
    }
}

tl::expected<void, EdgeStateError> EdgeStateMachine::beginDraining() {
    switch (state_) {
        case EdgeState::kActive:
            state_ = EdgeState::kDraining;
            return {};
        default:
            return tl::unexpected(EdgeStateError::kInvalidTransition);
    }
}

tl::expected<void, EdgeStateError> EdgeStateMachine::confirmDisconnected() {
    switch (state_) {
        case EdgeState::kDraining:
        case EdgeState::kConnecting:
            state_ = EdgeState::kDisconnected;
            return {};
        default:
            return tl::unexpected(EdgeStateError::kInvalidTransition);
    }
}

void EdgeStateMachine::forceReset() {
    state_ = EdgeState::kIdle;
}

const char* edgeStateToString(EdgeState state) {
    switch (state) {
        case EdgeState::kIdle:          return "Idle";
        case EdgeState::kConnecting:    return "Connecting";
        case EdgeState::kActive:        return "Active";
        case EdgeState::kDraining:      return "Draining";
        case EdgeState::kDisconnected:  return "Disconnected";
    }
    return "Unknown";
}

} /* namespace core::graph */
