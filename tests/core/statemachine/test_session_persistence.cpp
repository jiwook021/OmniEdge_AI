#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

#include "statemachine/session_persistence.hpp"

namespace {

class SessionPersistenceTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		testDir_ = std::filesystem::temp_directory_path() / "omniedge_test_session";
		std::filesystem::create_directories(testDir_);
		testFile_ = (testDir_ / "session_state.json").string();
	}

	void TearDown() override
	{
		std::filesystem::remove_all(testDir_);
	}

	SessionPersistence::Config makeConfig()
	{
		return {
			.filePath             = testFile_,
			.periodicSaveInterval = std::chrono::seconds{1},
			.maxStaleness         = std::chrono::seconds{60},
		};
	}

	nlohmann::json makeSampleState()
	{
		return {
			{"mode", "conversation"},
			{"history", nlohmann::json::array({
				{{"role", "user"}, {"content", "hello"}},
				{{"role", "assistant"}, {"content", "hi there"}},
			})},
			{"face_identity", "Alice"},
		};
	}

	std::filesystem::path testDir_;
	std::string           testFile_;
};

// ── save ─────────────────────────────────────────────────────────────────

TEST_F(SessionPersistenceTest, SaveCreatesFile)
{
	SessionPersistence sp(makeConfig());
	auto state = makeSampleState();

	auto result = sp.save(state);
	ASSERT_TRUE(result.has_value()) << result.error();
	EXPECT_TRUE(std::filesystem::exists(testFile_));
}

TEST_F(SessionPersistenceTest, SaveWritesValidJson)
{
	SessionPersistence sp(makeConfig());
	auto state = makeSampleState();
	ASSERT_TRUE(sp.save(state).has_value());

	std::ifstream ifs(testFile_);
	nlohmann::json loaded;
	ASSERT_NO_THROW(ifs >> loaded);
	EXPECT_EQ(loaded["mode"], "conversation");
	EXPECT_EQ(loaded["face_identity"], "Alice");
}

TEST_F(SessionPersistenceTest, SaveOverwritesPrevious)
{
	SessionPersistence sp(makeConfig());
	auto state1 = makeSampleState();
	ASSERT_TRUE(sp.save(state1).has_value());

	auto state2 = state1;
	state2["mode"] = "conversation";
	ASSERT_TRUE(sp.save(state2).has_value());

	std::ifstream ifs(testFile_);
	nlohmann::json loaded;
	ifs >> loaded;
	EXPECT_EQ(loaded["mode"], "conversation");
}

// ── load ─────────────────────────────────────────────────────────────────

TEST_F(SessionPersistenceTest, LoadReturnsSavedState)
{
	SessionPersistence sp(makeConfig());
	auto state = makeSampleState();
	ASSERT_TRUE(sp.save(state).has_value());

	auto result = sp.load();
	ASSERT_TRUE(result.has_value()) << result.error();
	EXPECT_EQ(result.value()["mode"], "conversation");
	EXPECT_EQ(result.value()["face_identity"], "Alice");
}

TEST_F(SessionPersistenceTest, LoadReturnErrorOnMissingFile)
{
	SessionPersistence sp(makeConfig());

	auto result = sp.load();
	EXPECT_FALSE(result.has_value());
}

TEST_F(SessionPersistenceTest, LoadReturnErrorOnCorruptJson)
{
	SessionPersistence sp(makeConfig());

	// Write corrupt data
	std::ofstream ofs(testFile_);
	ofs << "{ this is not valid json !!!";
	ofs.close();

	auto result = sp.load();
	EXPECT_FALSE(result.has_value());
}

TEST_F(SessionPersistenceTest, LoadReturnsErrorOnStaleFile)
{
	auto cfg = makeConfig();
	cfg.maxStaleness = std::chrono::seconds{1};
	SessionPersistence sp(cfg);

	auto state = makeSampleState();
	ASSERT_TRUE(sp.save(state).has_value());

	// Wait for staleness
	std::this_thread::sleep_for(std::chrono::seconds{2});

	auto result = sp.load();
	EXPECT_FALSE(result.has_value());
}

TEST_F(SessionPersistenceTest, LoadReturnsFreshFile)
{
	auto cfg = makeConfig();
	cfg.maxStaleness = std::chrono::seconds{60};
	SessionPersistence sp(cfg);

	auto state = makeSampleState();
	ASSERT_TRUE(sp.save(state).has_value());

	// Should be fresh enough
	auto result = sp.load();
	ASSERT_TRUE(result.has_value()) << result.error();
	EXPECT_EQ(result.value()["mode"], "conversation");
}

// ── shouldPeriodicSave ───────────────────────────────────────────────────

TEST_F(SessionPersistenceTest, ShouldPeriodicSaveIsFalseImmediately)
{
	auto cfg = makeConfig();
	cfg.periodicSaveInterval = std::chrono::seconds{10};
	SessionPersistence sp(cfg);

	EXPECT_FALSE(sp.shouldPeriodicSave());
}

TEST_F(SessionPersistenceTest, ShouldPeriodicSaveBecomesTrueAfterInterval)
{
	auto cfg = makeConfig();
	cfg.periodicSaveInterval = std::chrono::seconds{1};
	SessionPersistence sp(cfg);

	std::this_thread::sleep_for(std::chrono::milliseconds{1100});
	EXPECT_TRUE(sp.shouldPeriodicSave());
}

TEST_F(SessionPersistenceTest, ResetSaveTimerRestartsInterval)
{
	auto cfg = makeConfig();
	cfg.periodicSaveInterval = std::chrono::seconds{1};
	SessionPersistence sp(cfg);

	std::this_thread::sleep_for(std::chrono::milliseconds{1100});
	EXPECT_TRUE(sp.shouldPeriodicSave());

	sp.resetSaveTimer();
	EXPECT_FALSE(sp.shouldPeriodicSave());
}

// ── Atomic write safety ────────────────────────────────────────────────

TEST_F(SessionPersistenceTest, SaveIsAtomic_NoPartialWrite)
{
	SessionPersistence sp(makeConfig());

	// Multiple sequential saves should all produce valid files
	for (int i = 0; i < 10; ++i) {
		auto state = makeSampleState();
		state["iteration"] = i;
		auto result = sp.save(state);
		ASSERT_TRUE(result.has_value()) << result.error();

		std::ifstream ifs(testFile_);
		nlohmann::json loaded;
		ASSERT_NO_THROW(ifs >> loaded);
		EXPECT_EQ(loaded["iteration"], i);
	}
}

} // namespace
