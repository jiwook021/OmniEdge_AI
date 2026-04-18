#include <gtest/gtest.h>

#include "orchestrator/module_launcher.hpp"

namespace {

class ModuleLauncherTest : public ::testing::Test {
protected:
	ModuleDescriptor makeDesc(const std::string& name = "test_mod",
							  const std::string& bin  = "/bin/sleep")
	{
		ModuleDescriptor d;
		d.name        = name;
		d.binaryPath  = bin;
		d.args        = {"5"};
		d.maxRestarts = 3;
		d.pid         = -1;
		d.restartCount = 0;
		d.ready       = false;
		return d;
	}
};

// ── spawnModule ──────────────────────────────────────────────────────────

TEST_F(ModuleLauncherTest, SpawnModuleSetsPositivePid)
{
	ModuleLauncher launcher;
	auto desc = makeDesc();
	auto result = launcher.spawnModule(desc);

	ASSERT_TRUE(result.has_value()) << result.error();
	EXPECT_GT(result.value(), 0);
	EXPECT_GT(desc.pid, 0);

	// Clean up
	launcher.stopModule(desc, std::chrono::seconds{2});
}

TEST_F(ModuleLauncherTest, SpawnModuleWithBadBinaryReturnsError)
{
	ModuleLauncher launcher;
	auto desc = makeDesc("bad", "/nonexistent/binary/path");
	auto result = launcher.spawnModule(desc);

	EXPECT_FALSE(result.has_value());
}

// ── stopModule ───────────────────────────────────────────────────────────

TEST_F(ModuleLauncherTest, StopModuleResetsPid)
{
	ModuleLauncher launcher;
	auto desc = makeDesc();
	auto result = launcher.spawnModule(desc);
	ASSERT_TRUE(result.has_value()) << result.error();

	launcher.stopModule(desc, std::chrono::seconds{2});
	EXPECT_EQ(desc.pid, -1);
	EXPECT_FALSE(desc.isRunning());
}

TEST_F(ModuleLauncherTest, StopModuleOnNotRunningIsNoOp)
{
	ModuleLauncher launcher;
	auto desc = makeDesc();
	// Should not crash when stopping a module that was never started
	launcher.stopModule(desc, std::chrono::seconds{1});
	EXPECT_EQ(desc.pid, -1);
}

// ── checkExited ──────────────────────────────────────────────────────────

TEST_F(ModuleLauncherTest, CheckExitedReturnsFalseWhileRunning)
{
	ModuleLauncher launcher;
	auto desc = makeDesc("sleeper", "/bin/sleep");
	desc.args = {"10"};
	auto result = launcher.spawnModule(desc);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Process should still be running
	EXPECT_FALSE(launcher.checkExited(desc));
	EXPECT_TRUE(desc.isRunning());

	launcher.stopModule(desc, std::chrono::seconds{2});
}

TEST_F(ModuleLauncherTest, CheckExitedReturnsTrueAfterProcessEnds)
{
	ModuleLauncher launcher;
	auto desc = makeDesc("fast", "/bin/true");
	desc.args.clear();
	auto result = launcher.spawnModule(desc);
	ASSERT_TRUE(result.has_value()) << result.error();

	// Give it a moment to exit
	std::this_thread::sleep_for(std::chrono::milliseconds{200});

	EXPECT_TRUE(launcher.checkExited(desc));
	EXPECT_FALSE(desc.isRunning());
}

// ── restartModule ────────────────────────────────────────────────────────

TEST_F(ModuleLauncherTest, RestartModuleIncrementsCounter)
{
	ModuleLauncher launcher;
	auto desc = makeDesc("restartable", "/bin/true");
	desc.args.clear();
	desc.maxRestarts = 5;

	// Initial spawn
	auto r1 = launcher.spawnModule(desc);
	ASSERT_TRUE(r1.has_value());

	std::this_thread::sleep_for(std::chrono::milliseconds{200});
	(void)launcher.checkExited(desc);

	// Restart
	auto r2 = launcher.restartModule(desc);
	ASSERT_TRUE(r2.has_value()) << r2.error();
	EXPECT_EQ(desc.restartCount, 1);

	std::this_thread::sleep_for(std::chrono::milliseconds{200});
	(void)launcher.checkExited(desc);
}

TEST_F(ModuleLauncherTest, RestartModuleExceedingMaxReturnsError)
{
	ModuleLauncher launcher;
	auto desc = makeDesc("limited", "/bin/true");
	desc.args.clear();
	desc.maxRestarts = 1;
	desc.restartCount = 1;

	auto result = launcher.restartModule(desc);
	EXPECT_FALSE(result.has_value());
	EXPECT_NE(result.error().find("max"), std::string::npos);
}

TEST_F(ModuleLauncherTest, RestartBackoffIncreasesExponentially)
{
	ModuleLauncher launcher;
	// Use small backoff values to keep the test fast
	launcher.setBackoffConfig(/*baseMs=*/ 50, /*maxMs=*/ 5000);

	auto desc = makeDesc("backoff_test", "/bin/true");
	desc.args.clear();
	desc.maxRestarts = 3;

	// Initial spawn
	auto r0 = launcher.spawnModule(desc);
	ASSERT_TRUE(r0.has_value());
	std::this_thread::sleep_for(std::chrono::milliseconds{100});
	(void)launcher.checkExited(desc);

	// First restart (restartCount=0 → backoff = 50 * 2^0 = 50ms)
	auto t1Start = std::chrono::steady_clock::now();
	auto r1 = launcher.restartModule(desc);
	auto t1Elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - t1Start);
	ASSERT_TRUE(r1.has_value()) << r1.error();
	EXPECT_EQ(desc.restartCount, 1);
	EXPECT_GE(t1Elapsed.count(), 40);  // ~50ms with tolerance

	std::this_thread::sleep_for(std::chrono::milliseconds{100});
	(void)launcher.checkExited(desc);

	// Second restart (restartCount=1 → backoff = 50 * 2^1 = 100ms)
	auto t2Start = std::chrono::steady_clock::now();
	auto r2 = launcher.restartModule(desc);
	auto t2Elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - t2Start);
	ASSERT_TRUE(r2.has_value()) << r2.error();
	EXPECT_EQ(desc.restartCount, 2);
	EXPECT_GE(t2Elapsed.count(), 80);  // ~100ms with tolerance
	// Second restart should take longer than first
	EXPECT_GT(t2Elapsed.count(), t1Elapsed.count());

	std::this_thread::sleep_for(std::chrono::milliseconds{100});
	(void)launcher.checkExited(desc);
}

// ── launchAll ────────────────────────────────────────────────────────────

TEST_F(ModuleLauncherTest, LaunchAllSpawnsMultipleModules)
{
	ModuleLauncher launcher;
	zmq::context_t ctx(1);

	std::vector<ModuleDescriptor> modules;
	for (int i = 0; i < 3; ++i) {
		auto d = makeDesc("mod_" + std::to_string(i), "/bin/sleep");
		d.args = {"10"};
		modules.push_back(std::move(d));
	}

	// Short timeout — modules won't send module_ready, so all should be
	// in the "not ready" set.
	auto notReady = launcher.launchAll(modules, ctx, std::chrono::seconds{1});

	// All modules should have been spawned even if not ready
	for (auto& m : modules) {
		EXPECT_TRUE(m.isRunning()) << m.name << " not running";
		launcher.stopModule(m, std::chrono::seconds{2});
	}
}

// ── ModuleDescriptor queries ─────────────────────────────────────────────

TEST_F(ModuleLauncherTest, IsRunningReflectsPid)
{
	ModuleDescriptor d;
	d.pid = -1;
	EXPECT_FALSE(d.isRunning());
	d.pid = 0;
	EXPECT_FALSE(d.isRunning());
	d.pid = 42;
	EXPECT_TRUE(d.isRunning());
}

} // namespace
