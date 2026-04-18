#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "ws_bridge/websocket_bridge_node.hpp"
#include "common/constants/cv_constants.hpp"
#include "common/constants/video_denoise_constants.hpp"

// ---------------------------------------------------------------------------
// WebSocketBridgeNode Frame Pacing — Unit Tests
//
// Tests verify the frame pacing logic that smooths video delivery to
// the frontend at a fixed cadence regardless of GPU pipeline jitter.
//
// Test categories:
//   1. shouldPace() — static helper for time-based gating
//   2. Relay behavior — skip-to-newest, re-send-previous, pacing gate
//
// These tests use a MockRelay that counts broadcast calls, allowing
// us to verify pacing without real SHM or GPU.
// ---------------------------------------------------------------------------


// ===========================================================================
// 1. shouldPace() — "only allow a send if enough time has elapsed"
// ===========================================================================

TEST(FramePacing, ShouldPace_ReturnsTrueForZeroLastSend)
{
	// First frame ever — lastSend is default (epoch) — should always pace
	std::chrono::steady_clock::time_point epoch{};
	EXPECT_TRUE(WebSocketBridgeNode::shouldPace(epoch, kBlurFramePacingMs));
}

TEST(FramePacing, ShouldPace_ReturnsFalseWhenTooSoon)
{
	auto now = std::chrono::steady_clock::now();
	// Just sent — should NOT pace yet
	EXPECT_FALSE(WebSocketBridgeNode::shouldPace(now, kBlurFramePacingMs));
}

TEST(FramePacing, ShouldPace_ReturnsTrueAfterInterval)
{
	auto pastTime = std::chrono::steady_clock::now()
		- std::chrono::milliseconds{kBlurFramePacingMs + 1};
	EXPECT_TRUE(WebSocketBridgeNode::shouldPace(pastTime, kBlurFramePacingMs));
}

TEST(FramePacing, ShouldPace_WorksWithDenoiseInterval)
{
	auto pastTime = std::chrono::steady_clock::now()
		- std::chrono::milliseconds{kDenoiseFramePacingMs + 1};
	EXPECT_TRUE(WebSocketBridgeNode::shouldPace(pastTime, kDenoiseFramePacingMs));
}

// ===========================================================================
// 2. Constants sanity — "pacing and deadline values are reasonable"
// ===========================================================================

TEST(FramePacingConstants, BlurPacingMatchesDeadline)
{
	// The blur pacing interval should equal the deadline — both target 20fps
	EXPECT_EQ(kBlurFramePacingMs, kBlurDeadlineMs);
	EXPECT_EQ(kBlurFramePacingMs, 50);
}

TEST(FramePacingConstants, DenoisePacingIsWithinDeadline)
{
	// Denoise pacing is 50ms but deadline is 100ms (BasicVSR++ is slower)
	EXPECT_LE(kDenoiseFramePacingMs, kDenoiseDeadlineMs);
	EXPECT_EQ(kDenoiseFramePacingMs, 50);
	EXPECT_EQ(kDenoiseDeadlineMs, 100);
}

// ===========================================================================
// 3. Pacing interval boundary — "50ms interval produces ~20fps"
// ===========================================================================

TEST(FramePacing, PacingIntervalProduces20FPS)
{
	// Simulate 200ms of frame arrivals at ~30fps (every ~33ms)
	// With 50ms pacing, we should get ~4 sends in 200ms (20fps)
	int sendCount = 0;
	auto lastSend = std::chrono::steady_clock::time_point{};

	// Simulate 6 frames arriving at 33ms intervals (0, 33, 66, 99, 132, 165ms)
	for (int i = 0; i < 6; ++i) {
		if (WebSocketBridgeNode::shouldPace(lastSend, kBlurFramePacingMs)) {
			++sendCount;
			lastSend = std::chrono::steady_clock::now();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}

	// At 33ms intervals over 200ms with 50ms pacing: expect ~4 sends
	EXPECT_GE(sendCount, 3) << "Should send at least 3 frames in 200ms at 20fps pacing";
	EXPECT_LE(sendCount, 5) << "Should not send more than 5 frames in 200ms";
}

