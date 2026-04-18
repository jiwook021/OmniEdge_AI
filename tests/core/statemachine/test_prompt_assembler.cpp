#include <gtest/gtest.h>

#include "statemachine/prompt_assembler.hpp"

namespace {

class PromptAssemblerTest : public ::testing::Test {
protected:
	PromptAssembler::Config makeConfig()
	{
		return {
			.maxContextTokens    = 4096,
			.systemPromptTokens  = 200,
			.dynamicContextTokens = 100,
			.maxUserTurnTokens   = 500,
			.generationHeadroom  = 500,
			.systemPrompt        = "You are OmniEdge AI, a helpful assistant.",
		};
	}
};

TEST_F(PromptAssemblerTest, AssembleProducesValidStructureWithSystemAndUser)
{
	PromptAssembler pa(makeConfig());
	auto result = pa.assemble("What is AI?");

	EXPECT_TRUE(result.contains("v"));
	EXPECT_TRUE(result.contains("type"));
	EXPECT_EQ(result["type"], "conversation_prompt");
	EXPECT_TRUE(result.contains("messages"));
	EXPECT_TRUE(result["messages"].is_array());

	auto& messages = result["messages"];
	ASSERT_GE(messages.size(), 2U);
	EXPECT_EQ(messages.front()["role"], "system");
	EXPECT_EQ(messages.back()["role"], "user");
	EXPECT_EQ(messages.back()["content"], "What is AI?");
}

TEST_F(PromptAssemblerTest, DynamicContextInjectedIntoSystemPrompt)
{
	PromptAssembler pa(makeConfig());
	pa.setFaceIdentity("Alice", 0.96f);
	pa.setSceneDescription("A person sitting at a desk.");

	auto result = pa.assemble("Hi");
	std::string systemContent = result["messages"][0]["content"];
	EXPECT_NE(systemContent.find("Alice"), std::string::npos);
	EXPECT_NE(systemContent.find("0.96"), std::string::npos);
	EXPECT_NE(systemContent.find("A person sitting at a desk."), std::string::npos);
}

TEST_F(PromptAssemblerTest, ClearFaceIdentityRemovesItFromSystem)
{
	PromptAssembler pa(makeConfig());
	pa.setFaceIdentity("Bob", 0.9f);
	pa.setFaceIdentity("", 0.0f);

	auto result = pa.assemble("Hi");
	std::string systemContent = result["messages"][0]["content"];
	EXPECT_EQ(systemContent.find("Bob"), std::string::npos);
}

TEST_F(PromptAssemblerTest, AddToHistoryAppearsInMessages)
{
	PromptAssembler pa(makeConfig());
	pa.addToHistory("How are you?", "I'm doing well.");

	auto result = pa.assemble("Tell me more.");
	auto& messages = result["messages"];

	// System + history(user,assistant) + current user = 4 messages
	ASSERT_EQ(messages.size(), 4U);
	EXPECT_EQ(messages[1]["role"], "user");
	EXPECT_EQ(messages[1]["content"], "How are you?");
	EXPECT_EQ(messages[2]["role"], "assistant");
	EXPECT_EQ(messages[2]["content"], "I'm doing well.");
}

TEST_F(PromptAssemblerTest, ClearHistoryRemovesAllTurns)
{
	PromptAssembler pa(makeConfig());
	pa.addToHistory("Q1", "A1");
	pa.addToHistory("Q2", "A2");
	EXPECT_EQ(pa.historySize(), 2U);

	pa.clearHistory();
	EXPECT_EQ(pa.historySize(), 0U);

	auto result = pa.assemble("Fresh start");
	// System + user only (no history)
	EXPECT_EQ(result["messages"].size(), 2U);
}

TEST_F(PromptAssemblerTest, HistoryEvictionRemovesOldestTurns)
{
	// Use a very small context window to force eviction
	auto cfg = makeConfig();
	cfg.maxContextTokens = 100;
	cfg.generationHeadroom = 20;
	PromptAssembler pa(cfg);

	// Add many turns to exceed budget
	for (int i = 0; i < 20; ++i) {
		pa.addToHistory(
			"This is a long user question number " + std::to_string(i),
			"This is a correspondingly long assistant response " + std::to_string(i));
	}

	auto result = pa.assemble("Current question");
	auto& messages = result["messages"];

	// History should have been trimmed to fit
	EXPECT_LT(messages.size(), 42U);  // 20 turns = 40 messages + system + user = 42
	// But should still contain system and current user
	EXPECT_EQ(messages.front()["role"], "system");
	EXPECT_EQ(messages.back()["role"], "user");
	EXPECT_EQ(messages.back()["content"], "Current question");
}

TEST_F(PromptAssemblerTest, EmptyUtteranceStillAssembles)
{
	PromptAssembler pa(makeConfig());
	auto result = pa.assemble("");

	EXPECT_TRUE(result.contains("messages"));
	EXPECT_EQ(result["messages"].back()["content"], "");
}

} // namespace
