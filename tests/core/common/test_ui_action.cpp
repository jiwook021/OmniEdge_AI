#include <gtest/gtest.h>

#include "common/ui_action.hpp"

// ---------------------------------------------------------------------------
// UiAction enum tests — verify parse/name round-trip for all known actions.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Round-trip: parseUiAction(name) → enum → uiActionName(enum) == name
// ---------------------------------------------------------------------------

struct UiActionRoundTrip {
	std::string_view wire;
	UiAction         expected;
};

class UiActionRoundTripTest
	: public ::testing::TestWithParam<UiActionRoundTrip> {};

TEST_P(UiActionRoundTripTest, ParseAndNameRoundTrip)
{
	const auto& [wire, expected] = GetParam();
	const UiAction parsed = parseUiAction(wire);
	EXPECT_EQ(parsed, expected);
	EXPECT_EQ(uiActionName(parsed), wire);
}

INSTANTIATE_TEST_SUITE_P(AllActions, UiActionRoundTripTest, ::testing::Values(
	UiActionRoundTrip{"start_audio",        UiAction::kStartAudio},
	UiActionRoundTrip{"stop_audio",         UiAction::kStopAudio},
	UiActionRoundTrip{"enable_webcam",      UiAction::kEnableWebcam},
	UiActionRoundTrip{"disable_webcam",     UiAction::kDisableWebcam},
	UiActionRoundTrip{"push_to_talk",       UiAction::kPushToTalk},
	UiActionRoundTrip{"text_input",         UiAction::kTextInput},
	UiActionRoundTrip{"describe_scene",     UiAction::kDescribeScene},
	UiActionRoundTrip{"tts_complete",       UiAction::kTtsComplete},
	UiActionRoundTrip{"switch_mode",        UiAction::kSwitchMode},
	UiActionRoundTrip{"cancel_generation",  UiAction::kCancelGeneration},
	UiActionRoundTrip{"flush_tts",          UiAction::kFlushTts},
	UiActionRoundTrip{"stop_playback",      UiAction::kStopPlayback},
	UiActionRoundTrip{"register_face",      UiAction::kRegisterFace},
	UiActionRoundTrip{"set_image_adjust",   UiAction::kSetImageAdjust},
	UiActionRoundTrip{"set_bg_mode",        UiAction::kSetBgMode},
	UiActionRoundTrip{"update_shapes",      UiAction::kUpdateShapes}
));

