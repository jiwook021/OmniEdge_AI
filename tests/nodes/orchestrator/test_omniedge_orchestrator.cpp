#include <gtest/gtest.h>

#include "orchestrator/omniedge_daemon.hpp"

namespace {

class OmniEdgeDaemonTest : public ::testing::Test {
protected:
	OmniEdgeDaemon::Config makeConfig()
	{
		OmniEdgeDaemon::Config cfg;
		cfg.configFile          = "";  // No YAML — for unit tests
		cfg.watchdogPollMs      = 100;
		cfg.pubPort             = 25571;  // Non-default port for tests
		cfg.vadSilenceThresholdMs = 800;
		cfg.bargeInEnabled      = true;
		cfg.llmGenerationTimeoutS = 5;
		cfg.moduleReadyTimeoutS = 2;
		cfg.gpuDeviceId         = 0;
		cfg.gpuOverrideProfile  = "";
		cfg.gpuHeadroomMb       = 500;
		cfg.sessionFilePath     = "/tmp/omniedge_test_session.json";
		return cfg;
	}
};

// ── Construction ─────────────────────────────────────────────────────────

TEST_F(OmniEdgeDaemonTest, ConstructsWithConfig)
{
	auto cfg = makeConfig();
	OmniEdgeDaemon daemon(cfg);

	// Starts in IDLE state
	EXPECT_EQ(daemon.currentState(), StateIndex::kIdle);
}

// ── stop ────────────────────────────────────────────────────────────────

TEST_F(OmniEdgeDaemonTest, StopSetsFlag)
{
	auto cfg = makeConfig();
	OmniEdgeDaemon daemon(cfg);

	// stop() should not crash even before initialize()
	daemon.stop();
}

// ── FSM accessor ─────────────────────────────────────────────────────────

TEST_F(OmniEdgeDaemonTest, StateMachineAccessor)
{
	auto cfg = makeConfig();
	OmniEdgeDaemon daemon(cfg);

	// FSM should be accessible and have default state
	auto& fsm = daemon.stateMachine();
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

// ── Module list is initially empty ───────────────────────────────────────

TEST_F(OmniEdgeDaemonTest, ModulesInitiallyEmpty)
{
	auto cfg = makeConfig();
	OmniEdgeDaemon daemon(cfg);

	EXPECT_TRUE(daemon.modules().empty());
}

// ── State machine transitions after forceState ──────────────────────────

TEST_F(OmniEdgeDaemonTest, ForceStateChangesCurrentState)
{
	auto cfg = makeConfig();
	OmniEdgeDaemon daemon(cfg);

	daemon.stateMachine().forceState<fsm::Listening>();
	EXPECT_EQ(daemon.currentState(), StateIndex::kListening);

	daemon.stateMachine().forceState<fsm::ErrorRecovery>();
	EXPECT_EQ(daemon.currentState(), StateIndex::kErrorRecovery);
}

} // namespace
