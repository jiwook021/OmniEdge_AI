#include <gtest/gtest.h>

#include "statemachine/state_machine.hpp"

namespace {

// ---------------------------------------------------------------------------
// Test transition table — mirrors the daemon's transitions but without
// side effects.  Guards use pointers to test-owned booleans so each test
// controls exactly which path the FSM takes.
// ---------------------------------------------------------------------------

struct TestTransitions {
	bool* hasTranscription = nullptr;
	bool* bargeInEnabled   = nullptr;
	bool* actionCalled     = nullptr;

	// ── Voice conversation lifecycle ────────────────────────────────────

	// User presses push-to-talk button → mic opens, begin capturing audio
	std::optional<State> operator()(fsm::Idle, fsm::PttPress) const {
		if (actionCalled) *actionCalled = true;
		return fsm::Listening{};
	}
	// User sends a text message → skip audio capture, go straight to LLM
	std::optional<State> operator()(fsm::Idle, fsm::TextInput) const {
		return fsm::Processing{};
	}
	// User asks "what do you see?" → LLM processes camera frame directly
	std::optional<State> operator()(fsm::Idle, fsm::DescribeScene) const {
		return fsm::Processing{};
	}

	// User releases PTT: if we captured speech, send it to the LLM;
	// if the mic was empty (accidental tap), return to idle silently
	std::optional<State> operator()(fsm::Listening, fsm::PttRelease) const {
		return *hasTranscription ? State{fsm::Processing{}} : State{fsm::Idle{}};
	}
	// VAD detected end-of-speech silence while listening.
	// Guard: only transition if there's actual speech to process —
	// prevents empty audio from wasting LLM tokens
	std::optional<State> operator()(fsm::Listening, fsm::VadSilence) const {
		if (!*hasTranscription) return std::nullopt;
		return fsm::Processing{};
	}
	// User explicitly cancelled recording (e.g. swipe-away gesture)
	std::optional<State> operator()(fsm::Listening, fsm::PttCancel) const {
		return fsm::Idle{};
	}

	// LLM produced its first sentence → begin TTS playback immediately
	// (streaming: don't wait for full response to start speaking)
	std::optional<State> operator()(fsm::Processing, fsm::LlmFirstSentence) const {
		return fsm::Speaking{};
	}
	// LLM finished but produced no TTS-worthy output (e.g. tool-only response)
	std::optional<State> operator()(fsm::Processing, fsm::LlmComplete) const {
		return fsm::Idle{};
	}
	// LLM didn't respond within deadline — abort and return to idle
	std::optional<State> operator()(fsm::Processing, fsm::LlmTimeout) const {
		return fsm::Idle{};
	}

	// Barge-in: user presses PTT while TTS is playing.
	// If barge-in is enabled → interrupt TTS, prepare to listen again.
	// If disabled (e.g. during critical announcements) → ignore the press.
	std::optional<State> operator()(fsm::Speaking, fsm::PttPress) const {
		return *bargeInEnabled ? State{fsm::Interrupted{}} : State{fsm::Speaking{}};
	}
	// TTS finished playing the full response — conversation turn complete
	std::optional<State> operator()(fsm::Speaking, fsm::TtsComplete) const {
		return fsm::Idle{};
	}

	// TTS audio flushed and stopped after barge-in → re-open mic
	std::optional<State> operator()(fsm::Interrupted, fsm::InterruptDone) const {
		return fsm::Listening{};
	}

	// ── Error recovery (any state can crash) ────────────────────────────

	std::optional<State> operator()(fsm::Idle, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}
	std::optional<State> operator()(fsm::Listening, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}
	std::optional<State> operator()(fsm::Processing, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}
	std::optional<State> operator()(fsm::Speaking, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}
	// Watchdog restarted the crashed module → resume normal operation
	std::optional<State> operator()(fsm::ErrorRecovery, fsm::ModuleRecovered) const {
		return fsm::Idle{};
	}
	// Module didn't recover in time — give up and return to idle anyway
	// (daemon will log the failure; user can retry)
	std::optional<State> operator()(fsm::ErrorRecovery, fsm::RecoveryTimeout) const {
		return fsm::Idle{};
	}

	// ── Vision pipeline states ──────────────────────────────────────────

	// SAM2 interactive segmentation mode
	std::optional<State> operator()(fsm::Idle, fsm::Sam2SegmentationStart) const {
		return fsm::Sam2Segmentation{};
	}
	std::optional<State> operator()(fsm::Sam2Segmentation, fsm::Sam2SegmentationStop) const {
		return fsm::Idle{};
	}
	std::optional<State> operator()(fsm::Sam2Segmentation, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}

	// Vision model (e.g. Gemma-4 describe-what-I-see continuous mode)
	std::optional<State> operator()(fsm::Idle, fsm::VisionModelStart) const {
		return fsm::VisionModel{};
	}
	std::optional<State> operator()(fsm::VisionModel, fsm::VisionModelStop) const {
		return fsm::Idle{};
	}
	std::optional<State> operator()(fsm::VisionModel, fsm::ModuleCrash) const {
		return fsm::ErrorRecovery{};
	}

	// Catch-all: any unhandled (state, event) pair is a no-op
	template <typename S, typename E>
	std::optional<State> operator()(S, E) const { return std::nullopt; }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class StateMachineTest : public ::testing::Test {
protected:
	StateMachine fsm;

	bool hasTranscription_ = true;
	bool bargeInEnabled_   = true;
	bool actionCalled_     = false;

	TestTransitions table() {
		return {&hasTranscription_, &bargeInEnabled_, &actionCalled_};
	}

	template <typename EventT>
	std::optional<StateIndex> dispatch(const EventT& event) {
		return fsm.dispatch(table(), event);
	}
};

// ===========================================================================
// Boot state
// ===========================================================================

TEST_F(StateMachineTest, StartsIdleOnConstruction)
{
	// The daemon always boots into idle — no mic, no LLM, waiting for input
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

// ===========================================================================
// Voice conversation: happy path
// ===========================================================================

TEST_F(StateMachineTest, PttPressOpensTheMicrophone)
{
	auto result = dispatch(fsm::PttPress{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kListening);
	EXPECT_EQ(fsm.currentState(), StateIndex::kListening);
}

TEST_F(StateMachineTest, PttReleaseWithSpeechSendsAudioToLlm)
{
	hasTranscription_ = true;
	fsm.forceState<fsm::Listening>();

	auto result = dispatch(fsm::PttRelease{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kProcessing);
}

TEST_F(StateMachineTest, PttReleaseWithNoSpeechReturnsToIdle)
{
	// Accidental tap: user pressed and released without saying anything.
	// Must NOT send empty audio to the LLM (wastes tokens, confuses model).
	hasTranscription_ = false;
	fsm.forceState<fsm::Listening>();

	auto result = dispatch(fsm::PttRelease{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

TEST_F(StateMachineTest, VadSilenceTriggersProcessingWhenSpeechCaptured)
{
	// Hands-free mode: VAD detected end-of-utterance silence.
	// Only transition if we actually have speech — prevents ghost triggers.
	hasTranscription_ = true;
	fsm.forceState<fsm::Listening>();

	auto result = dispatch(fsm::VadSilence{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kProcessing);
}

TEST_F(StateMachineTest, VadSilenceIgnoredWithoutSpeech)
{
	// Background noise triggered VAD but no actual words were captured.
	// Guard prevents sending empty audio to the LLM.
	hasTranscription_ = false;
	fsm.forceState<fsm::Listening>();

	auto result = dispatch(fsm::VadSilence{});
	EXPECT_FALSE(result.has_value());
	EXPECT_EQ(fsm.currentState(), StateIndex::kListening);
}

TEST_F(StateMachineTest, LlmFirstSentenceStartsStreamingTts)
{
	// Streaming response: don't wait for the full LLM reply — start speaking
	// as soon as the first complete sentence arrives (lower perceived latency).
	fsm.forceState<fsm::Processing>();

	auto result = dispatch(fsm::LlmFirstSentence{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kSpeaking);
}

TEST_F(StateMachineTest, TtsCompleteEndsConversationTurn)
{
	fsm.forceState<fsm::Speaking>();

	auto result = dispatch(fsm::TtsComplete{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

// ===========================================================================
// Voice conversation: alternate inputs
// ===========================================================================

TEST_F(StateMachineTest, TextInputBypassesAudioCapture)
{
	// User typed a message instead of speaking — skip straight to LLM
	auto result = dispatch(fsm::TextInput{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kProcessing);
}

TEST_F(StateMachineTest, DescribeSceneSendsFrameToLlm)
{
	// "What do you see?" — LLM processes the camera frame natively
	// (no separate vision pipeline needed, the multi-modal LLM handles it)
	auto result = dispatch(fsm::DescribeScene{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kProcessing);
}

// ===========================================================================
// Voice conversation: cancellation and timeouts
// ===========================================================================

TEST_F(StateMachineTest, PttCancelAbortsRecording)
{
	// User swiped away or hit cancel — discard captured audio
	fsm.forceState<fsm::Listening>();

	auto result = dispatch(fsm::PttCancel{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

TEST_F(StateMachineTest, LlmCompleteWithNoTtsReturnsToIdle)
{
	// LLM finished but the response was tool-only (no spoken output needed)
	fsm.forceState<fsm::Processing>();

	auto result = dispatch(fsm::LlmComplete{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

TEST_F(StateMachineTest, LlmTimeoutAbortsAndReturnsToIdle)
{
	// Safety valve: LLM didn't respond within the deadline.
	// Return to idle so the user isn't stuck waiting forever.
	fsm.forceState<fsm::Processing>();

	auto result = dispatch(fsm::LlmTimeout{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

// ===========================================================================
// Barge-in: interrupting TTS playback
// ===========================================================================

TEST_F(StateMachineTest, BargeInInterruptsTtsWhenEnabled)
{
	// User wants to cut off the assistant mid-sentence and say something new.
	// TTS audio must be flushed before mic re-opens (Interrupted state handles that).
	bargeInEnabled_ = true;
	fsm.forceState<fsm::Speaking>();

	auto result = dispatch(fsm::PttPress{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kInterrupted);
}

TEST_F(StateMachineTest, PttPressIgnoredDuringSpeakingWhenBargeInDisabled)
{
	// During critical announcements (e.g. error messages), barge-in is
	// disabled so the user hears the full message before responding.
	bargeInEnabled_ = false;
	fsm.forceState<fsm::Speaking>();

	auto result = dispatch(fsm::PttPress{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kSpeaking);
}

TEST_F(StateMachineTest, InterruptDoneReopensTheMicrophone)
{
	// TTS audio has been flushed after barge-in → now safe to listen again
	fsm.forceState<fsm::Interrupted>();

	auto result = dispatch(fsm::InterruptDone{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kListening);
}

// ===========================================================================
// Error recovery: crash in any active state
// ===========================================================================

TEST_F(StateMachineTest, CrashFromIdleEntersRecovery)
{
	auto result = dispatch(fsm::ModuleCrash{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kErrorRecovery);
}

TEST_F(StateMachineTest, CrashFromListeningEntersRecovery)
{
	fsm.forceState<fsm::Listening>();
	EXPECT_EQ(*dispatch(fsm::ModuleCrash{}), StateIndex::kErrorRecovery);
}

TEST_F(StateMachineTest, CrashFromProcessingEntersRecovery)
{
	fsm.forceState<fsm::Processing>();
	EXPECT_EQ(*dispatch(fsm::ModuleCrash{}), StateIndex::kErrorRecovery);
}

TEST_F(StateMachineTest, CrashFromSpeakingEntersRecovery)
{
	fsm.forceState<fsm::Speaking>();
	EXPECT_EQ(*dispatch(fsm::ModuleCrash{}), StateIndex::kErrorRecovery);
}

TEST_F(StateMachineTest, WatchdogRecoveryClearsError)
{
	// Watchdog restarted the crashed module → daemon resumes normal operation
	fsm.forceState<fsm::ErrorRecovery>();

	auto result = dispatch(fsm::ModuleRecovered{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

TEST_F(StateMachineTest, RecoveryTimeoutFallsBackToIdle)
{
	// Module didn't recover in time — return to idle anyway.
	// The daemon logs the failure; user can retry their action.
	fsm.forceState<fsm::ErrorRecovery>();

	auto result = dispatch(fsm::RecoveryTimeout{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

// ===========================================================================
// Five-mode vision pipeline states
// ===========================================================================

TEST_F(StateMachineTest, Sam2SegmentationStartBeginsInteractiveSegmentation)
{
	// User activated SAM2 point-and-click segmentation mode
	auto result = dispatch(fsm::Sam2SegmentationStart{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kSam2Segmentation);
}

TEST_F(StateMachineTest, Sam2SegmentationStopReturnsToIdle)
{
	fsm.forceState<fsm::Sam2Segmentation>();
	EXPECT_EQ(*dispatch(fsm::Sam2SegmentationStop{}), StateIndex::kIdle);
}

TEST_F(StateMachineTest, Sam2SegmentationCrashEntersRecovery)
{
	fsm.forceState<fsm::Sam2Segmentation>();
	EXPECT_EQ(*dispatch(fsm::ModuleCrash{}), StateIndex::kErrorRecovery);
}

TEST_F(StateMachineTest, VisionModelStartBeginsDescribeMode)
{
	// User activated continuous vision model (e.g. Gemma-4 "describe what I see")
	auto result = dispatch(fsm::VisionModelStart{});
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kVisionModel);
}

TEST_F(StateMachineTest, VisionModelStopReturnsToIdle)
{
	fsm.forceState<fsm::VisionModel>();
	EXPECT_EQ(*dispatch(fsm::VisionModelStop{}), StateIndex::kIdle);
}

TEST_F(StateMachineTest, VisionModelCrashEntersRecovery)
{
	fsm.forceState<fsm::VisionModel>();
	EXPECT_EQ(*dispatch(fsm::ModuleCrash{}), StateIndex::kErrorRecovery);
}

// ===========================================================================
// Dropped events: FSM must silently ignore invalid transitions
// ===========================================================================

TEST_F(StateMachineTest, TtsCompleteIgnoredWhileIdle)
{
	// Late TTS_COMPLETE arrives after the conversation already ended.
	// Must not crash or change state — just log a warning and move on.
	auto result = dispatch(fsm::TtsComplete{});
	EXPECT_FALSE(result.has_value());
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

// ===========================================================================
// Action callbacks: transition handlers can trigger side effects
// ===========================================================================

TEST_F(StateMachineTest, TransitionHandlerCallbackIsInvoked)
{
	// The transition table can execute side effects (e.g. start audio capture)
	// when a transition fires. Verify the callback mechanism works.
	actionCalled_ = false;
	(void)dispatch(fsm::PttPress{});
	EXPECT_TRUE(actionCalled_);
}

// ===========================================================================
// forceState: daemon uses this for external resets and test setup
// ===========================================================================

TEST_F(StateMachineTest, ForceStateByTypeChangesState)
{
	fsm.forceState<fsm::Speaking>();
	EXPECT_EQ(fsm.currentState(), StateIndex::kSpeaking);
}

TEST_F(StateMachineTest, ForceStateByIndexChangesState)
{
	fsm.forceState(StateIndex::kErrorRecovery);
	EXPECT_EQ(fsm.currentState(), StateIndex::kErrorRecovery);
}

// ===========================================================================
// Name lookups: stateName / eventName for logging and ZMQ messages
// ===========================================================================

TEST_F(StateMachineTest, StateNamesMatchZmqProtocolStrings)
{
	// These strings are sent over ZMQ to the frontend — they must be stable
	EXPECT_EQ(stateName(StateIndex::kIdle), "idle");
	EXPECT_EQ(stateName(StateIndex::kListening), "listening");
	EXPECT_EQ(stateName(StateIndex::kProcessing), "processing");
	EXPECT_EQ(stateName(StateIndex::kSpeaking), "speaking");
	EXPECT_EQ(stateName(StateIndex::kInterrupted), "interrupted");
	EXPECT_EQ(stateName(StateIndex::kErrorRecovery), "error_recovery");
}

TEST_F(StateMachineTest, EventNamesMatchZmqProtocolStrings)
{
	EXPECT_EQ(eventName<fsm::PttPress>(), "ptt_press");
	EXPECT_EQ(eventName<fsm::PttRelease>(), "ptt_release");
	EXPECT_EQ(eventName<fsm::DescribeScene>(), "describe_scene");
	EXPECT_EQ(eventName<fsm::LlmTimeout>(), "llm_timeout");
	EXPECT_EQ(eventName<fsm::InterruptDone>(), "interrupt_done");
}

TEST_F(StateMachineTest, AnyEventVariantResolvesCorrectName)
{
	AnyEvent event = fsm::ModuleCrash{};
	EXPECT_EQ(eventName(event), "module_crash");
}

// ===========================================================================
// AnyEvent dispatch: deferred/queued events dispatched from variant
// ===========================================================================

TEST_F(StateMachineTest, DeferredEventDispatchesThroughAnyEventVariant)
{
	// The daemon queues events as AnyEvent when they arrive via ZMQ.
	// Verify the variant-based dispatch path works identically.
	fsm.forceState<fsm::ErrorRecovery>();
	AnyEvent event = fsm::ModuleRecovered{};

	auto result = fsm.dispatch(table(), event);
	ASSERT_TRUE(result.has_value());
	EXPECT_EQ(*result, StateIndex::kIdle);
}

// ===========================================================================
// End-to-end flows: full multi-step scenarios
// ===========================================================================

TEST_F(StateMachineTest, FullVoiceConversation_PttToTtsComplete)
{
	// Complete voice conversation: press PTT, speak, LLM responds, TTS plays out
	hasTranscription_ = true;

	// User presses PTT → mic opens
	(void)dispatch(fsm::PttPress{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kListening);

	// User releases PTT with captured speech → send to LLM
	(void)dispatch(fsm::PttRelease{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kProcessing);

	// LLM produces first sentence → start speaking immediately
	(void)dispatch(fsm::LlmFirstSentence{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kSpeaking);

	// TTS finishes playing → back to idle, ready for next turn
	(void)dispatch(fsm::TtsComplete{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

TEST_F(StateMachineTest, BargeInFlow_InterruptTtsAndResumeListening)
{
	// User interrupts the assistant mid-sentence to correct or add context
	bargeInEnabled_ = true;
	fsm.forceState<fsm::Speaking>();

	// User presses PTT while TTS is playing → interrupt
	(void)dispatch(fsm::PttPress{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kInterrupted);

	// TTS audio flushed → mic re-opens for user's correction
	(void)dispatch(fsm::InterruptDone{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kListening);
}

TEST_F(StateMachineTest, DescribeSceneFlow_VisionQueryToCompletion)
{
	// "What do you see?" flow: camera frame goes to multi-modal LLM
	(void)dispatch(fsm::DescribeScene{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kProcessing);

	(void)dispatch(fsm::LlmComplete{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

TEST_F(StateMachineTest, CrashRecoveryFlow_ModuleCrashAndWatchdogRestart)
{
	// A module (e.g. TTS process) crashes mid-processing.
	// Watchdog detects it, restarts the process, signals recovery.
	fsm.forceState<fsm::Processing>();

	(void)dispatch(fsm::ModuleCrash{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kErrorRecovery);

	(void)dispatch(fsm::ModuleRecovered{});
	EXPECT_EQ(fsm.currentState(), StateIndex::kIdle);
}

// ===========================================================================
// Structural validation: detect dead states and unreachable idle
// ===========================================================================

TEST_F(StateMachineTest, NoDeadStates_AllStatesHaveAtLeastOneExit)
{
	// A dead state would trap the daemon — every state must have at least
	// one outgoing transition so the system can always make progress.
	auto states = allStates();
	auto events = allEvents();
	auto handler = table();

	for (std::size_t si = 0; si < kStateCount; ++si) {
		bool hasAny = false;
		std::visit([&](const auto& s) {
			for (const auto& event : events) {
				std::visit([&](const auto& e) {
					if (handler(s, e).has_value()) hasAny = true;
				}, event);
			}
		}, states[si]);

		EXPECT_TRUE(hasAny) << "Dead state detected: "
		                     << stateName(static_cast<StateIndex>(si))
		                     << " has no outgoing transitions";
	}
}

TEST_F(StateMachineTest, ErrorRecoveryCanAlwaysReachIdle)
{
	// Critical safety property: error recovery must never be a sink state.
	// If it can't reach idle, the daemon is permanently stuck after any crash.
	auto events = allEvents();
	auto handler = table();
	auto recovery = State{fsm::ErrorRecovery{}};

	bool canReachIdle = false;
	std::visit([&](const auto& s) {
		for (const auto& event : events) {
			std::visit([&](const auto& e) {
				auto result = handler(s, e);
				if (result && stateIndex(*result) == StateIndex::kIdle)
					canReachIdle = true;
			}, event);
		}
	}, recovery);

	EXPECT_TRUE(canReachIdle)
		<< "kErrorRecovery has no transition that reaches kIdle — "
		   "the daemon would be permanently stuck after any crash";
}

TEST_F(StateMachineTest, StateAndEventCountsMatchVariantSizes)
{
	auto states = allStates();
	auto events = allEvents();

	EXPECT_EQ(states.size(), kStateCount);
	EXPECT_EQ(events.size(), kEventCount);

	for (std::size_t i = 0; i < kStateCount; ++i) {
		EXPECT_EQ(states[i].index(), i);
	}
}

} // namespace
