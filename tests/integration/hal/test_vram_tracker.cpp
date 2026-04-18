#include <gtest/gtest.h>

#include "common/ini_config.hpp"
#include "vram/vram_tracker.hpp"
#include "vram/vram_thresholds.hpp"
#include "common/runtime_defaults.hpp"

// ---------------------------------------------------------------------------
// VramTracker tests — CPU-only, no CUDA required.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Fixture — registers a standard set of modules before each test
// ---------------------------------------------------------------------------

class VramTrackerTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		// Use kConversation profile priorities (default boot state in five-mode architecture)
		static const IniConfig iniConfig;
		const auto& priorities = iniConfig.profilePriorities(InteractionProfile::kConversation);
		tracker_.registerModuleBudget("background_blur",    kBgBlurMiB,          priorities.at("background_blur"));
		tracker_.registerModuleBudget("face_recognition",   kFaceRecogMiB,       priorities.at("face_recognition"));
		tracker_.registerModuleBudget("conversation_model", kGemmaE4bMiB,       priorities.at("conversation_model"));
		tracker_.registerModuleBudget("tts",                kTtsMiB,             priorities.at("tts"));
		tracker_.registerModuleBudget("audio_denoise",      kDtlnMiB,            priorities.at("audio_denoise"));
		tracker_.registerModuleBudget("sam2",               kSam2MiB,            priorities.at("sam2"));
	}

	VramTracker tracker_;
};

// ---------------------------------------------------------------------------
// registerModuleBudget
// ---------------------------------------------------------------------------

TEST_F(VramTrackerTest, DuplicateRegistrationThrows)
{
	EXPECT_THROW(
		tracker_.registerModuleBudget("background_blur", 500, 0),
		std::runtime_error);
}

TEST_F(VramTrackerTest, EmptyModuleNameThrows)
{
	EXPECT_THROW(
		tracker_.registerModuleBudget("", 100, 0),
		std::invalid_argument);
}

// ---------------------------------------------------------------------------
// markModuleLoaded / markModuleUnloaded
// ---------------------------------------------------------------------------

TEST_F(VramTrackerTest, InitiallyNoModulesLoaded)
{
	EXPECT_EQ(tracker_.totalLoadedMb(), 0u);
}

TEST_F(VramTrackerTest, RecordLoadedIncreasesTotalMb)
{
	tracker_.markModuleLoaded("conversation_model");
	EXPECT_EQ(tracker_.totalLoadedMb(), kGemmaE4bMiB);
}

TEST_F(VramTrackerTest, RecordUnloadedDecreasesTotalMb)
{
	tracker_.markModuleLoaded("conversation_model");
	tracker_.markModuleLoaded("tts");
	tracker_.markModuleUnloaded("conversation_model");

	EXPECT_EQ(tracker_.totalLoadedMb(), kTtsMiB);
}

TEST_F(VramTrackerTest, RecordLoadedUnknownModuleThrows)
{
	EXPECT_THROW(tracker_.markModuleLoaded("nonexistent_module"), std::runtime_error);
}

TEST_F(VramTrackerTest, RecordUnloadedUnknownModuleThrows)
{
	EXPECT_THROW(tracker_.markModuleUnloaded("nonexistent_module"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// evictableCandidates — ordering and filtering
// ---------------------------------------------------------------------------

TEST_F(VramTrackerTest, NoCandidatesWhenNothingLoaded)
{
	EXPECT_TRUE(tracker_.evictableCandidates().empty());
}

TEST_F(VramTrackerTest, ConversationModelNeverEvictedInConversationProfile)
{
	// conversation_model has priority 5 in kConversation — never evictable
	tracker_.markModuleLoaded("conversation_model");
	const auto candidates = tracker_.evictableCandidates();
	for (const auto& c : candidates) {
		EXPECT_NE(c.moduleName, "conversation_model")
			<< "conversation_model must never be evicted in conversation profile";
	}
}

TEST_F(VramTrackerTest, CandidatesSortedByPriorityAscending)
{
	// In kConversation: audio_denoise=1, sam2=2
	tracker_.markModuleLoaded("audio_denoise");  // priority 1
	tracker_.markModuleLoaded("sam2");           // priority 2

	const auto candidates = tracker_.evictableCandidates();
	ASSERT_EQ(candidates.size(), 2u);
	EXPECT_LE(candidates[0].evictPriority, candidates[1].evictPriority);
	EXPECT_EQ(candidates[0].moduleName, "audio_denoise");
}

TEST_F(VramTrackerTest, BusyModuleExcludedFromCandidates)
{
	// Use evictable modules (priority < 5) for this test.
	tracker_.markModuleLoaded("audio_denoise");
	tracker_.markModuleLoaded("sam2");
	tracker_.setIdle("audio_denoise", false);  // busy — must be excluded

	const auto candidates = tracker_.evictableCandidates();
	ASSERT_EQ(candidates.size(), 1u);
	EXPECT_EQ(candidates[0].moduleName, "sam2");
}

TEST_F(VramTrackerTest, UnloadedModuleNotACandidate)
{
	tracker_.markModuleLoaded("background_blur");
	tracker_.markModuleUnloaded("background_blur");  // unloaded → not a candidate

	EXPECT_TRUE(tracker_.evictableCandidates().empty());
}

// ---------------------------------------------------------------------------
// setIdle — state transitions
// ---------------------------------------------------------------------------

TEST_F(VramTrackerTest, SetIdleFalseMarksBusy)
{
	tracker_.markModuleLoaded("audio_denoise");
	tracker_.setIdle("audio_denoise", false);

	// audio_denoise is busy — must not appear as candidate
	const auto candidates = tracker_.evictableCandidates();
	for (const auto& c : candidates) {
		EXPECT_NE(c.moduleName, "audio_denoise");
	}
}

TEST_F(VramTrackerTest, SetIdleUnknownModuleThrows)
{
	EXPECT_THROW(tracker_.setIdle("ghost_module", true), std::runtime_error);
}

// ---------------------------------------------------------------------------
// snapshot
// ---------------------------------------------------------------------------

TEST_F(VramTrackerTest, SnapshotContainsAllRegisteredModules)
{
	const auto snap = tracker_.snapshot();
	EXPECT_EQ(snap.size(), 6u);  // bg_blur, face_recog, conversation_model, tts, audio_denoise, sam2
}

TEST_F(VramTrackerTest, SnapshotReflectsLoadedState)
{
	tracker_.markModuleLoaded("tts");

	const auto snap = tracker_.snapshot();
	bool found = false;
	for (const auto& rec : snap) {
		if (rec.moduleName == "tts") {
			EXPECT_TRUE(rec.isLoaded);
			found = true;
		}
	}
	EXPECT_TRUE(found);
}

