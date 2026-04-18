/* tests/core/graph/test_edge_state_machine.cpp — State transition validation */

#include <gtest/gtest.h>
#include "graph/edge_state_machine.hpp"

using core::graph::EdgeState;
using core::graph::EdgeStateError;
using core::graph::EdgeStateMachine;

TEST(EdgeStateMachineTest, test_initialState) {
    EdgeStateMachine sm("e1");
    EXPECT_EQ(sm.current(), EdgeState::kIdle);
    EXPECT_EQ(sm.edgeId(), "e1");
}

TEST(EdgeStateMachineTest, test_happyPath) {
    EdgeStateMachine sm("e1");

    EXPECT_TRUE(sm.beginConnecting().has_value());
    EXPECT_EQ(sm.current(), EdgeState::kConnecting);

    EXPECT_TRUE(sm.confirmActive().has_value());
    EXPECT_EQ(sm.current(), EdgeState::kActive);

    EXPECT_TRUE(sm.beginDraining().has_value());
    EXPECT_EQ(sm.current(), EdgeState::kDraining);

    EXPECT_TRUE(sm.confirmDisconnected().has_value());
    EXPECT_EQ(sm.current(), EdgeState::kDisconnected);
}

TEST(EdgeStateMachineTest, test_crashPath) {
    /* Connecting → Disconnected (process crashed before becoming active) */
    EdgeStateMachine sm("e1");
    EXPECT_TRUE(sm.beginConnecting().has_value());
    EXPECT_TRUE(sm.confirmDisconnected().has_value());
    EXPECT_EQ(sm.current(), EdgeState::kDisconnected);
}

TEST(EdgeStateMachineTest, test_invalidTransitions) {
    EdgeStateMachine sm("e1");

    /* From Idle: only beginConnecting is valid */
    EXPECT_FALSE(sm.confirmActive().has_value());
    EXPECT_FALSE(sm.beginDraining().has_value());
    EXPECT_FALSE(sm.confirmDisconnected().has_value());

    /* From Connecting: only confirmActive and confirmDisconnected are valid */
    sm.beginConnecting();
    EXPECT_FALSE(sm.beginConnecting().has_value());
    EXPECT_FALSE(sm.beginDraining().has_value());

    /* From Active: only beginDraining is valid */
    sm.confirmActive();
    EXPECT_FALSE(sm.beginConnecting().has_value());
    EXPECT_FALSE(sm.confirmActive().has_value());
    EXPECT_FALSE(sm.confirmDisconnected().has_value());

    /* From Draining: only confirmDisconnected is valid */
    sm.beginDraining();
    EXPECT_FALSE(sm.beginConnecting().has_value());
    EXPECT_FALSE(sm.confirmActive().has_value());
    EXPECT_FALSE(sm.beginDraining().has_value());
}

TEST(EdgeStateMachineTest, test_forceResetFromEveryState) {
    /* forceReset works from every state */
    for (int i = 0; i <= 4; ++i) {
        EdgeStateMachine sm("e1");
        switch (static_cast<EdgeState>(i)) {
            case EdgeState::kIdle: break;
            case EdgeState::kConnecting: sm.beginConnecting(); break;
            case EdgeState::kActive: sm.beginConnecting(); sm.confirmActive(); break;
            case EdgeState::kDraining: sm.beginConnecting(); sm.confirmActive(); sm.beginDraining(); break;
            case EdgeState::kDisconnected: sm.beginConnecting(); sm.confirmDisconnected(); break;
        }
        sm.forceReset();
        EXPECT_EQ(sm.current(), EdgeState::kIdle);
    }
}

TEST(EdgeStateMachineTest, test_stateToString) {
    EXPECT_STREQ(core::graph::edgeStateToString(EdgeState::kIdle), "Idle");
    EXPECT_STREQ(core::graph::edgeStateToString(EdgeState::kActive), "Active");
    EXPECT_STREQ(core::graph::edgeStateToString(EdgeState::kDisconnected), "Disconnected");
}
