// test_conversation_params.cpp -- Level 1 pure logic tests for GenerationParams
//
// No GPU, no ZMQ, no mocks required.  Tests pure value semantics.
//
// Test coverage:
//   - GenerationParams default values
//   - clamp() on out-of-range values
//   - parseGenerationParams() JSON with all fields
//   - parseGenerationParams() JSON with partial fields (defaults used)
//   - parseGenerationParams() JSON with missing generation_params (all defaults)
//   - Sentence boundary detection edge cases

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "conversation_inferencer.hpp"
#include "conversation/conversation_node.hpp"


// ---------------------------------------------------------------------------
// GenerationParams::clamp() tests
// ---------------------------------------------------------------------------

TEST(GenerationParamsTest, DefaultValuesAreValid)
{
    GenerationParams params;
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.topP, 0.9f);
    EXPECT_EQ(params.maxTokens, 2048);

    GenerationParams original = params;
    params.clamp();
    EXPECT_FLOAT_EQ(params.temperature, original.temperature);
    EXPECT_FLOAT_EQ(params.topP, original.topP);
    EXPECT_EQ(params.maxTokens, original.maxTokens);
}

TEST(GenerationParamsTest, ClampNegativeTemperature)
{
    GenerationParams params;
    params.temperature = -0.5f;
    params.clamp();
    EXPECT_FLOAT_EQ(params.temperature, 0.0f);
}

TEST(GenerationParamsTest, ClampExcessiveTemperature)
{
    GenerationParams params;
    params.temperature = 5.0f;
    params.clamp();
    EXPECT_FLOAT_EQ(params.temperature, 2.0f);
}

TEST(GenerationParamsTest, ClampNegativeTopP)
{
    GenerationParams params;
    params.topP = -0.1f;
    params.clamp();
    EXPECT_FLOAT_EQ(params.topP, 0.0f);
}

TEST(GenerationParamsTest, ClampExcessiveTopP)
{
    GenerationParams params;
    params.topP = 1.5f;
    params.clamp();
    EXPECT_FLOAT_EQ(params.topP, 1.0f);
}

TEST(GenerationParamsTest, ClampTooFewTokens)
{
    GenerationParams params;
    params.maxTokens = 10;
    params.clamp();
    EXPECT_EQ(params.maxTokens, 64);
}

TEST(GenerationParamsTest, ClampTooManyTokens)
{
    GenerationParams params;
    params.maxTokens = 20000;
    params.clamp();
    EXPECT_EQ(params.maxTokens, 8192);
}

TEST(GenerationParamsTest, BoundaryValuesNotClamped)
{
    GenerationParams params;
    params.temperature = 0.0f;
    params.topP = 0.0f;
    params.maxTokens = 64;
    params.clamp();
    EXPECT_FLOAT_EQ(params.temperature, 0.0f);
    EXPECT_FLOAT_EQ(params.topP, 0.0f);
    EXPECT_EQ(params.maxTokens, 64);

    params.temperature = 2.0f;
    params.topP = 1.0f;
    params.maxTokens = 8192;
    params.clamp();
    EXPECT_FLOAT_EQ(params.temperature, 2.0f);
    EXPECT_FLOAT_EQ(params.topP, 1.0f);
    EXPECT_EQ(params.maxTokens, 8192);
}

// ---------------------------------------------------------------------------
// ConversationNode::parseGenerationParams() tests
// ---------------------------------------------------------------------------

TEST(ParseGenerationParamsTest, AllFieldsPresent)
{
    nlohmann::json msg = {
        {"type", "conversation_prompt"},
        {"text", "hello"},
        {"generation_params", {
            {"temperature", 0.3},
            {"top_p", 0.5},
            {"max_tokens", 1024},
        }},
    };
    auto params = ConversationNode::parseGenerationParams(msg);
    EXPECT_FLOAT_EQ(params.temperature, 0.3f);
    EXPECT_FLOAT_EQ(params.topP, 0.5f);
    EXPECT_EQ(params.maxTokens, 1024);
}

TEST(ParseGenerationParamsTest, PartialFieldsUseDefaults)
{
    nlohmann::json msg = {
        {"type", "conversation_prompt"},
        {"text", "hello"},
        {"generation_params", {
            {"temperature", 1.2},
        }},
    };
    auto params = ConversationNode::parseGenerationParams(msg);
    EXPECT_FLOAT_EQ(params.temperature, 1.2f);
    EXPECT_FLOAT_EQ(params.topP, 0.9f);      // default
    EXPECT_EQ(params.maxTokens, 2048);        // default
}

TEST(ParseGenerationParamsTest, MissingGenerationParamsUsesAllDefaults)
{
    nlohmann::json msg = {
        {"type", "conversation_prompt"},
        {"text", "hello"},
    };
    auto params = ConversationNode::parseGenerationParams(msg);
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.topP, 0.9f);
    EXPECT_EQ(params.maxTokens, 2048);
}

TEST(ParseGenerationParamsTest, OutOfRangeValuesClamped)
{
    nlohmann::json msg = {
        {"type", "conversation_prompt"},
        {"text", "hello"},
        {"generation_params", {
            {"temperature", -1.0},
            {"top_p", 2.0},
            {"max_tokens", 99999},
        }},
    };
    auto params = ConversationNode::parseGenerationParams(msg);
    EXPECT_FLOAT_EQ(params.temperature, 0.0f);
    EXPECT_FLOAT_EQ(params.topP, 1.0f);
    EXPECT_EQ(params.maxTokens, 8192);
}

TEST(ParseGenerationParamsTest, NonObjectGenerationParamsUsesDefaults)
{
    nlohmann::json msg = {
        {"type", "conversation_prompt"},
        {"text", "hello"},
        {"generation_params", "invalid"},
    };
    auto params = ConversationNode::parseGenerationParams(msg);
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.topP, 0.9f);
    EXPECT_EQ(params.maxTokens, 2048);
}

// ---------------------------------------------------------------------------
// Config::validate() tests
// ---------------------------------------------------------------------------

TEST(ConversationConfigTest, ValidConfigPasses)
{
    ConversationNode::Config cfg;
    cfg.pubPort          = 5572;
    cfg.daemonSubPort    = 5571;
    cfg.uiCommandSubPort = 5570;
    cfg.modelDir         = "/tmp/model";
    cfg.modelVariant     = "gemma_e4b";

    auto result = ConversationNode::Config::validate(cfg);
    ASSERT_TRUE(result.has_value()) << result.error();
}

TEST(ConversationConfigTest, EmptyModelDirFails)
{
    ConversationNode::Config cfg;
    cfg.pubPort          = 5572;
    cfg.daemonSubPort    = 5571;
    cfg.uiCommandSubPort = 5570;
    cfg.modelDir         = "";
    cfg.modelVariant     = "gemma_e4b";

    auto result = ConversationNode::Config::validate(cfg);
    EXPECT_FALSE(result.has_value());
}

TEST(ConversationConfigTest, EmptyModelVariantFails)
{
    ConversationNode::Config cfg;
    cfg.pubPort          = 5572;
    cfg.daemonSubPort    = 5571;
    cfg.uiCommandSubPort = 5570;
    cfg.modelDir         = "/tmp/model";
    cfg.modelVariant     = "";

    auto result = ConversationNode::Config::validate(cfg);
    EXPECT_FALSE(result.has_value());
}

TEST(ConversationConfigTest, InvalidPortFails)
{
    ConversationNode::Config cfg;
    cfg.pubPort          = 80;  // below 1024
    cfg.daemonSubPort    = 5571;
    cfg.uiCommandSubPort = 5570;
    cfg.modelDir         = "/tmp/model";
    cfg.modelVariant     = "gemma_e4b";

    auto result = ConversationNode::Config::validate(cfg);
    EXPECT_FALSE(result.has_value());
}

