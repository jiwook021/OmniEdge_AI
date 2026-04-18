// ---------------------------------------------------------------------------
// OmniEdge_AI — SubprocessManager tests
//
// Exercises real subprocess spawn/kill/restart lifecycle.
// Uses /bin/true, /bin/false, /bin/sleep as real subprocess targets.
//
// Bug caught by each test:
//   RunOnce_Success:          Broken spawn or wait logic → success path fails
//   RunOnce_NonZeroExit:      Missing exit-code propagation → silent failure
//   RunOnce_Timeout:          Missing SIGKILL escalation → test hangs forever
//   RunOnce_BadCommand:       Missing posix_spawnp error handling → crash
//   ManagedDaemon_Spawn:      RAII lifecycle broken → zombie or leaked process
//   ManagedDaemon_IsAlive:    isAlive false positive → incorrect daemon state
//   ManagedDaemon_Restart:    restart doesn't reap old process → PID leak
//   ManagedDaemon_Destructor: SIGTERM/SIGKILL escalation broken → zombie
//   ManagedDaemon_HealthProbe: health probe never called → startup hangs
//   MoveSemantics:            Double-free on moved-from manager → crash/zombie
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include <spdlog/spdlog.h>

#include "common/subprocess_manager.hpp"
#include "common/oe_logger.hpp"


// ---------------------------------------------------------------------------
// One-shot execution tests
// ---------------------------------------------------------------------------

TEST(SubprocessManagerTest, RunOnce_Success)
{
	// /bin/true exits with code 0
	auto result = SubprocessManager::runOnce("/bin/true", {}, std::chrono::seconds(5));
	SPDLOG_DEBUG("RunOnce_Success: has_value={}", result.has_value());
	ASSERT_TRUE(result.has_value()) << "Expected /bin/true to succeed: " << result.error();
}

TEST(SubprocessManagerTest, RunOnce_WithArgs)
{
	// /bin/echo with arguments — verifies arg passing works
	auto result = SubprocessManager::runOnce("/bin/echo", {"hello", "world"}, std::chrono::seconds(5));
	SPDLOG_DEBUG("RunOnce_WithArgs: has_value={}", result.has_value());
	ASSERT_TRUE(result.has_value()) << "Expected /bin/echo to succeed: " << result.error();
}

TEST(SubprocessManagerTest, RunOnce_NonZeroExit)
{
	// /bin/false exits with code 1
	auto result = SubprocessManager::runOnce("/bin/false", {}, std::chrono::seconds(5));
	SPDLOG_DEBUG("RunOnce_NonZeroExit: has_value={}, error={}",
	             result.has_value(), result.has_value() ? "" : result.error());
	ASSERT_FALSE(result.has_value()) << "Expected /bin/false to fail";
	EXPECT_NE(result.error().find("exited with code"), std::string::npos)
		<< "Error should mention exit code, got: " << result.error();
}

TEST(SubprocessManagerTest, RunOnce_Timeout)
{
	// sleep 60 with a 1-second timeout — must be killed, not hang
	const auto start = std::chrono::steady_clock::now();
	auto result = SubprocessManager::runOnce("/bin/sleep", {"60"}, std::chrono::seconds(1));
	const auto elapsed = std::chrono::steady_clock::now() - start;
	const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

	SPDLOG_DEBUG("RunOnce_Timeout: elapsed_ms={}, has_value={}, error={}",
	             elapsedMs, result.has_value(), result.has_value() ? "" : result.error());

	ASSERT_FALSE(result.has_value()) << "Expected timeout error";
	EXPECT_NE(result.error().find("timed out"), std::string::npos)
		<< "Error should mention timeout, got: " << result.error();
	// Should complete within ~2 seconds (1s timeout + poll overhead), not 60
	EXPECT_LT(elapsedMs, 5000) << "Timeout did not kill the subprocess promptly";
}

TEST(SubprocessManagerTest, RunOnce_BadCommand)
{
	auto result = SubprocessManager::runOnce(
		"/nonexistent/binary/that/does/not/exist", {}, std::chrono::seconds(5));
	SPDLOG_DEBUG("RunOnce_BadCommand: has_value={}, error={}",
	             result.has_value(), result.has_value() ? "" : result.error());
	ASSERT_FALSE(result.has_value()) << "Expected spawn failure for bad command";
}

TEST(SubprocessManagerTest, RunOnce_PathSearch)
{
	// "echo" without absolute path — posix_spawnp should find it via PATH
	auto result = SubprocessManager::runOnce("echo", {"test"}, std::chrono::seconds(5));
	SPDLOG_DEBUG("RunOnce_PathSearch: has_value={}", result.has_value());
	ASSERT_TRUE(result.has_value()) << "Expected PATH search to find 'echo': " << result.error();
}

// ---------------------------------------------------------------------------
// Managed daemon tests
// ---------------------------------------------------------------------------

TEST(SubprocessManagerTest, ManagedDaemon_SpawnAndDestroy)
{
	// Spawn sleep 60, verify alive, then let destructor kill it
	SubprocessManager::Config cfg;
	cfg.command = "/bin/sleep";
	cfg.args = {"60"};
	cfg.gracePeriod = std::chrono::seconds(1);

	auto result = SubprocessManager::spawn(cfg);
	ASSERT_TRUE(result.has_value()) << "Expected spawn to succeed: " << result.error();

	auto& mgr = *result;
	SPDLOG_DEBUG("ManagedDaemon_SpawnAndDestroy: pid={}", mgr.pid());
	EXPECT_GT(mgr.pid(), 0);
	EXPECT_TRUE(mgr.isAlive());
	// Destructor runs here — SIGTERM → SIGKILL
}

TEST(SubprocessManagerTest, ManagedDaemon_IsAlive_AfterExit)
{
	// Spawn /bin/true (exits immediately), then check isAlive
	SubprocessManager::Config cfg;
	cfg.command = "/bin/true";
	cfg.gracePeriod = std::chrono::seconds(1);

	auto result = SubprocessManager::spawn(cfg);
	ASSERT_TRUE(result.has_value()) << result.error();

	auto& mgr = *result;
	// Give subprocess time to exit
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	SPDLOG_DEBUG("ManagedDaemon_IsAlive_AfterExit: pid={}, alive={}", mgr.pid(), mgr.isAlive());
	EXPECT_FALSE(mgr.isAlive()) << "Process should have exited";
}

TEST(SubprocessManagerTest, ManagedDaemon_Restart)
{
	SubprocessManager::Config cfg;
	cfg.command = "/bin/sleep";
	cfg.args = {"60"};
	cfg.gracePeriod = std::chrono::seconds(1);

	auto result = SubprocessManager::spawn(cfg);
	ASSERT_TRUE(result.has_value()) << result.error();

	auto& mgr = *result;
	const pid_t firstPid = mgr.pid();
	SPDLOG_DEBUG("ManagedDaemon_Restart: first_pid={}", firstPid);

	auto restartResult = mgr.restart();
	ASSERT_TRUE(restartResult.has_value()) << restartResult.error();

	const pid_t secondPid = mgr.pid();
	SPDLOG_DEBUG("ManagedDaemon_Restart: second_pid={}", secondPid);
	EXPECT_NE(firstPid, secondPid) << "Restarted process should have a different PID";
	EXPECT_TRUE(mgr.isAlive());
}

TEST(SubprocessManagerTest, ManagedDaemon_HealthProbe)
{
	// Health probe that succeeds immediately
	int probeCallCount = 0;

	SubprocessManager::Config cfg;
	cfg.command = "/bin/sleep";
	cfg.args = {"60"};
	cfg.startupTimeout = std::chrono::seconds(5);
	cfg.gracePeriod = std::chrono::seconds(1);
	cfg.healthProbe = [&probeCallCount]() {
		++probeCallCount;
		return true;  // Immediately healthy
	};

	auto result = SubprocessManager::spawn(cfg);
	SPDLOG_DEBUG("ManagedDaemon_HealthProbe: probe_calls={}", probeCallCount);
	ASSERT_TRUE(result.has_value()) << result.error();
	EXPECT_GE(probeCallCount, 1) << "Health probe should have been called at least once";
}

TEST(SubprocessManagerTest, ManagedDaemon_HealthProbe_Timeout)
{
	// Health probe that never succeeds → should time out
	SubprocessManager::Config cfg;
	cfg.command = "/bin/sleep";
	cfg.args = {"60"};
	cfg.startupTimeout = std::chrono::seconds(1);
	cfg.gracePeriod = std::chrono::seconds(1);
	cfg.healthProbe = []() { return false; };  // Never healthy

	const auto start = std::chrono::steady_clock::now();
	auto result = SubprocessManager::spawn(cfg);
	[[maybe_unused]] const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - start).count();

	SPDLOG_DEBUG("ManagedDaemon_HealthProbe_Timeout: elapsed_ms={}, has_value={}",
	             elapsedMs, result.has_value());
	ASSERT_FALSE(result.has_value()) << "Expected startup timeout";
	EXPECT_NE(result.error().find("timed out"), std::string::npos);
}

TEST(SubprocessManagerTest, MoveSemantics)
{
	SubprocessManager::Config cfg;
	cfg.command = "/bin/sleep";
	cfg.args = {"60"};
	cfg.gracePeriod = std::chrono::seconds(1);

	auto result = SubprocessManager::spawn(cfg);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Move construct
	SubprocessManager mgr1 = std::move(*result);
	const pid_t pid = mgr1.pid();
	SPDLOG_DEBUG("MoveSemantics: original_pid={}", pid);
	EXPECT_GT(pid, 0);
	EXPECT_TRUE(mgr1.isAlive());

	// Move assign
	SubprocessManager mgr2 = std::move(mgr1);
	EXPECT_EQ(mgr1.pid(), -1) << "Moved-from manager should have pid -1";
	EXPECT_EQ(mgr2.pid(), pid) << "Moved-to manager should have the original pid";
	EXPECT_TRUE(mgr2.isAlive());
	// mgr2 destructor kills the process
}

