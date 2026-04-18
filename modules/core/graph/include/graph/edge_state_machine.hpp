#pragma once

/*
 * EdgeStateMachine — per-edge lifecycle state tracker.
 *
 * States: Idle → Connecting → Active → Draining → Disconnected
 * Transitions validated via switch; invalid transitions return error.
 */

#include <cstdint>
#include <string>
#include <tl/expected.hpp>

namespace core::graph {

enum class EdgeState : std::uint8_t {
    kIdle = 0,
    kConnecting = 1,
    kActive = 2,
    kDraining = 3,
    kDisconnected = 4
};

static_assert(static_cast<std::uint8_t>(EdgeState::kIdle) == 0);
static_assert(static_cast<std::uint8_t>(EdgeState::kDisconnected) == 4);

enum class EdgeStateError : std::uint8_t {
    kInvalidTransition = 0
};

class EdgeStateMachine {
public:
    EdgeStateMachine() = default;
    explicit EdgeStateMachine(std::string edgeId);

    [[nodiscard]] EdgeState current() const;
    [[nodiscard]] const std::string& edgeId() const;

    [[nodiscard]] tl::expected<void, EdgeStateError> beginConnecting();
    [[nodiscard]] tl::expected<void, EdgeStateError> confirmActive();
    [[nodiscard]] tl::expected<void, EdgeStateError> beginDraining();
    [[nodiscard]] tl::expected<void, EdgeStateError> confirmDisconnected();
    void forceReset();

private:
    std::string edgeId_;
    EdgeState state_{EdgeState::kIdle};
};

/* String representation for logging */
[[nodiscard]] const char* edgeStateToString(EdgeState state);

} /* namespace core::graph */
