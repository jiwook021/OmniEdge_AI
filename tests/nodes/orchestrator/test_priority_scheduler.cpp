// ---------------------------------------------------------------------------
// test_priority_scheduler.cpp — PriorityScheduler unit tests
//
// Tests: registration, priority ordering, VRAM-budget tiebreak, never-evict,
//        busy skip, preemption, dynamic setPriority, profile switching,
//        unloaded not evictable.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "common/ini_config.hpp"
#include "vram/priority_scheduler.hpp"

namespace {

// Default IniConfig provides built-in profile priorities for tests.
const IniConfig kDefaultIniConfig;


// ── Register and Snapshot ──────────────────────────────────────────────────

TEST(PrioritySchedulerTest, RegisterAndSnapshot)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("a", 0, 100);
	sched.registerModuleBudget("b", 1, 200);
	sched.registerModuleBudget("c", 2, 300);
	sched.registerModuleBudget("d", 5, 400);  // never-evict

	auto snap = sched.snapshot();
	EXPECT_EQ(snap.size(), 4u);
}

// ── Eviction Order Respects Priority ───────────────────────────────────────

TEST(PrioritySchedulerTest, EvictionOrderRespectsPriority)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("high",  3, 300);
	sched.registerModuleBudget("mid",   1, 200);
	sched.registerModuleBudget("low",   0, 100);

	sched.markModuleLoaded("high");
	sched.markModuleLoaded("mid");
	sched.markModuleLoaded("low");

	auto candidates = sched.evictionCandidates();
	ASSERT_GE(candidates.size(), 3u);
	EXPECT_EQ(candidates[0].moduleName, "low");   // priority 0 first
	EXPECT_EQ(candidates[1].moduleName, "mid");    // priority 1 second
	EXPECT_EQ(candidates[2].moduleName, "high");   // priority 3 third
}

// ── Same Priority: Largest VRAM Budget Evicted First ──────────────────────

TEST(PrioritySchedulerTest, SamePriorityLargestBudgetFirst)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("small",  1, 100);
	sched.registerModuleBudget("medium", 1, 500);
	sched.registerModuleBudget("large",  1, 1500);

	sched.markModuleLoaded("small");
	sched.markModuleLoaded("medium");
	sched.markModuleLoaded("large");

	auto candidates = sched.evictionCandidates();
	ASSERT_EQ(candidates.size(), 3u);
	// Within same priority (1), largest VRAM budget is evicted first
	EXPECT_EQ(candidates[0].moduleName, "large");   // 1500 MiB
	EXPECT_EQ(candidates[1].moduleName, "medium");   // 500 MiB
	EXPECT_EQ(candidates[2].moduleName, "small");    // 100 MiB
}

// ── LLM (Priority 5) Never Evicted ────────────────────────────────────────

TEST(PrioritySchedulerTest, LlmNeverEvicted)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("llm", 5, 4300);  // kNeverEvictPriority = 5
	sched.markModuleLoaded("llm");

	auto candidates = sched.evictionCandidates();
	EXPECT_TRUE(candidates.empty());

	auto best = sched.bestEvictionCandidate();
	EXPECT_FALSE(best.has_value());
}

// ── Busy Modules Skipped ───────────────────────────────────────────────────

TEST(PrioritySchedulerTest, BusyModulesSkipped)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("busy_mod", 1, 200);
	sched.markModuleLoaded("busy_mod");
	sched.setIdle("busy_mod", false);

	auto candidates = sched.evictionCandidates();
	EXPECT_TRUE(candidates.empty());
}

// ── Preemption Finds Lower-Priority Target ─────────────────────────────────

TEST(PrioritySchedulerTest, PreemptionFindsLowerPriorityTarget)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("low_pri", 1, 500);
	sched.registerModuleBudget("high_pri", 4, 1000);
	sched.markModuleLoaded("low_pri");
	sched.markModuleLoaded("high_pri");

	// High priority (4) module needs 400 MiB — low priority (1) has 500 MiB
	auto target = sched.findPreemptionTarget(4, 400);
	ASSERT_TRUE(target.has_value());
	EXPECT_EQ(*target, "low_pri");
}

TEST(PrioritySchedulerTest, PreemptionDoesNotEvictSamePriority)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("peer", 2, 500);
	sched.markModuleLoaded("peer");

	// Requester at same priority (2) — should not preempt
	auto target = sched.findPreemptionTarget(2, 400);
	EXPECT_FALSE(target.has_value());
}

// ── Dynamic setPriority Changes Eviction Order ────────────────────────────

TEST(PrioritySchedulerTest, SetPriorityChangesEvictionOrder)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("a", 0, 100);
	sched.registerModuleBudget("b", 2, 200);

	sched.markModuleLoaded("a");
	sched.markModuleLoaded("b");

	// Before: a (priority 0) is first eviction candidate
	auto before = sched.evictionCandidates();
	ASSERT_GE(before.size(), 2u);
	EXPECT_EQ(before[0].moduleName, "a");

	// Raise a's priority above b's
	sched.setPriority("a", 3);

	// After: b (priority 2) is now first eviction candidate
	auto after = sched.evictionCandidates();
	ASSERT_GE(after.size(), 2u);
	EXPECT_EQ(after[0].moduleName, "b");
}

// ── modulePriority Getter ──────────────────────────────────────────────────

TEST(PrioritySchedulerTest, ModulePriorityGetter)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("a", 3, 100);

	EXPECT_EQ(sched.modulePriority("a"), 3);
	EXPECT_EQ(sched.modulePriority("nonexistent"), -1);

	sched.setPriority("a", 1);
	EXPECT_EQ(sched.modulePriority("a"), 1);
}

// ── Profile Application Changes Priorities ─────────────────────────────────

TEST(PrioritySchedulerTest, ApplyProfileChangesPriorities)
{
	PriorityScheduler sched;
	// Register modules with kConversation defaults (default boot profile)
	sched.registerModuleBudget("conversation_model", 5, 3500);
	sched.registerModuleBudget("sam2",               0,  800);
	sched.registerModuleBudget("background_blur",    5,  250);

	EXPECT_EQ(sched.currentProfile(), InteractionProfile::kConversation);

	// Apply kSam2Segmentation profile — sam2 gets priority 5, conversation drops
	auto changed2 = sched.applyProfile(InteractionProfile::kSam2Segmentation,
		kDefaultIniConfig.profilePriorities(InteractionProfile::kSam2Segmentation));

	EXPECT_EQ(sched.currentProfile(), InteractionProfile::kSam2Segmentation);
	EXPECT_FALSE(changed2.empty());
	EXPECT_EQ(sched.modulePriority("sam2"), 5);
	EXPECT_EQ(sched.modulePriority("conversation_model"), 0);
	EXPECT_EQ(sched.modulePriority("background_blur"), 5);
}

// ── Profile Switch: kSam2Segmentation Evicts Conversation Model ─────────────

TEST(PrioritySchedulerTest, Sam2EvictsConversationModel)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("conversation_model", 5, 3500);
	sched.registerModuleBudget("sam2",               0,  800);
	sched.registerModuleBudget("background_blur",    5,  250);
	sched.registerModuleBudget("audio_denoise",      1,   50);

	// Load conversation model and always-on modules
	sched.markModuleLoaded("conversation_model");
	sched.markModuleLoaded("background_blur");
	sched.markModuleLoaded("audio_denoise");

	// Switch to SAM2 profile
	(void)sched.applyProfile(InteractionProfile::kSam2Segmentation,
		kDefaultIniConfig.profilePriorities(InteractionProfile::kSam2Segmentation));

	auto candidates = sched.evictionCandidates();
	// conversation_model should now be priority 0 (evict), audio_denoise priority 0
	// background_blur stays priority 5 (always-on, never evict)
	EXPECT_EQ(sched.modulePriority("conversation_model"), 0);
	EXPECT_EQ(sched.modulePriority("sam2"), 5);
	EXPECT_EQ(sched.modulePriority("background_blur"), 5);

	ASSERT_GE(candidates.size(), 2u);
	// conversation_model (0, 3500 MiB) evicted first (larger budget at same priority)
	EXPECT_EQ(candidates[0].moduleName, "conversation_model");
}

// ── Unloaded Modules Not Evictable ─────────────────────────────────────────

TEST(PrioritySchedulerTest, UnloadedModulesNotEvictable)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("mod", 1, 100);
	// Not loaded — should not appear in candidates
	auto candidates = sched.evictionCandidates();
	EXPECT_TRUE(candidates.empty());

	// Load then unload
	sched.markModuleLoaded("mod");
	sched.markModuleUnloaded("mod");
	candidates = sched.evictionCandidates();
	EXPECT_TRUE(candidates.empty());
}

// ── Best Eviction Candidate ────────────────────────────────────────────────

TEST(PrioritySchedulerTest, BestEvictionCandidateReturnsLowestPri)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("a", 0, 100);
	sched.registerModuleBudget("b", 2, 200);
	sched.markModuleLoaded("a");
	sched.markModuleLoaded("b");

	auto best = sched.bestEvictionCandidate();
	ASSERT_TRUE(best.has_value());
	EXPECT_EQ(best->moduleName, "a");  // priority 0 < 2
}

// ── Re-registration Overwrites ─────────────────────────────────────────────

TEST(PrioritySchedulerTest, ReregistrationOverwrites)
{
	PriorityScheduler sched;
	sched.registerModuleBudget("a", 2, 100);
	EXPECT_EQ(sched.modulePriority("a"), 2);

	sched.registerModuleBudget("a", 4, 500);
	EXPECT_EQ(sched.modulePriority("a"), 4);

	auto snap = sched.snapshot();
	EXPECT_EQ(snap.size(), 1u);  // not duplicated
}

// ── applyProfile Returns Only Changed Modules ──────────────────────────────

TEST(PrioritySchedulerTest, ApplyProfileReturnsOnlyChangedModules)
{
	PriorityScheduler sched;
	// Register with kConversation defaults (boot state)
	sched.registerModuleBudget("conversation_model", 5, 3500);
	sched.registerModuleBudget("sam2",               0,  800);

	// Switch to kSam2Segmentation — conversation_model should change (5→0),
	// sam2 should change (0→5)
	auto changed = sched.applyProfile(InteractionProfile::kSam2Segmentation,
		kDefaultIniConfig.profilePriorities(InteractionProfile::kSam2Segmentation));

	bool convChanged  = false;
	bool sam2Changed  = false;
	for (const auto& [name, pri] : changed) {
		if (name == "conversation_model") convChanged = true;
		if (name == "sam2")               sam2Changed = true;
	}

	EXPECT_TRUE(convChanged);  // conversation_model 5 → 0
	EXPECT_TRUE(sam2Changed);  // sam2 0 → 5
}

} // namespace
