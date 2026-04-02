#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <variant>


namespace fsm {
struct Idle {};
struct Listening {};
struct Processing {};
struct Speaking {};
struct Interrupted {};
struct ErrorRecovery {};
struct ImageTransform {};
struct SuperResolution {};
} // namespace fsm

using State = std::variant<
    fsm::Idle, fsm::Listening, fsm::Processing,
    fsm::Speaking, fsm::Interrupted, fsm::ErrorRecovery,
    fsm::ImageTransform, fsm::SuperResolution>;

enum class StateIndex : std::size_t {
    kIdle = 0, kListening = 1, kProcessing = 2,
    kSpeaking = 3, kInterrupted = 4, kErrorRecovery = 5,
    kImageTransform = 6, kSuperResolution = 7,
};

inline constexpr std::size_t kStateCount = 8;
inline constexpr std::size_t kEventCount = 19;

[[nodiscard]] constexpr StateIndex stateIndex(const State& s) noexcept {
    return static_cast<StateIndex>(s.index());
}

[[nodiscard]] std::string_view stateName(StateIndex s) noexcept;

[[nodiscard]] inline std::string_view stateName(const State& s) noexcept {
    return stateName(stateIndex(s));
}

namespace fsm {
struct PttPress {};
struct PttRelease {};
struct PttCancel {};
struct DescribeScene {};
struct TextInput {};
struct VadSilence {};
struct LlmFirstSentence {};
struct LlmComplete {};
struct LlmTimeout {};
struct TtsComplete {};
struct ModuleCrash {};
struct ModuleRecovered {};
struct RecoveryTimeout {};
struct InterruptDone {};
struct ImageTransformRequest {};
struct ImageTransformComplete {};
struct ImageTransformTimeout {};
struct SuperResolutionStart {};
struct SuperResolutionStop {};
} // namespace fsm

using AnyEvent = std::variant<
    fsm::PttPress, fsm::PttRelease, fsm::PttCancel,
    fsm::DescribeScene, fsm::TextInput, fsm::VadSilence,
    fsm::LlmFirstSentence, fsm::LlmComplete, fsm::LlmTimeout,
    fsm::TtsComplete,
    fsm::ModuleCrash, fsm::ModuleRecovered, fsm::RecoveryTimeout,
    fsm::InterruptDone,
    fsm::ImageTransformRequest, fsm::ImageTransformComplete,
    fsm::ImageTransformTimeout,
    fsm::SuperResolutionStart, fsm::SuperResolutionStop>;

template <typename T>
[[nodiscard]] constexpr std::string_view eventName() noexcept {
    if constexpr (std::is_same_v<T, fsm::PttPress>) return "ptt_press";
    if constexpr (std::is_same_v<T, fsm::PttRelease>) return "ptt_release";
    if constexpr (std::is_same_v<T, fsm::PttCancel>) return "ptt_cancel";
    if constexpr (std::is_same_v<T, fsm::DescribeScene>) return "describe_scene";
    if constexpr (std::is_same_v<T, fsm::TextInput>) return "text_input";
    if constexpr (std::is_same_v<T, fsm::VadSilence>) return "vad_silence";
    if constexpr (std::is_same_v<T, fsm::LlmFirstSentence>) return "llm_first_sentence";
    if constexpr (std::is_same_v<T, fsm::LlmComplete>) return "llm_complete";
    if constexpr (std::is_same_v<T, fsm::LlmTimeout>) return "llm_timeout";
    if constexpr (std::is_same_v<T, fsm::TtsComplete>) return "tts_complete";
    if constexpr (std::is_same_v<T, fsm::ModuleCrash>) return "module_crash";
    if constexpr (std::is_same_v<T, fsm::ModuleRecovered>) return "module_recovered";
    if constexpr (std::is_same_v<T, fsm::RecoveryTimeout>) return "recovery_timeout";
    if constexpr (std::is_same_v<T, fsm::InterruptDone>) return "interrupt_done";
    if constexpr (std::is_same_v<T, fsm::ImageTransformRequest>) return "image_transform_request";
    if constexpr (std::is_same_v<T, fsm::ImageTransformComplete>) return "image_transform_complete";
    if constexpr (std::is_same_v<T, fsm::ImageTransformTimeout>) return "image_transform_timeout";
    if constexpr (std::is_same_v<T, fsm::SuperResolutionStart>) return "super_resolution_start";
    if constexpr (std::is_same_v<T, fsm::SuperResolutionStop>) return "super_resolution_stop";
    return "unknown_event";
}

[[nodiscard]] inline std::string_view eventName(const AnyEvent& event) noexcept {
    return std::visit([](const auto& e) -> std::string_view {
        return eventName<std::decay_t<decltype(e)>>();
    }, event);
}

[[nodiscard]] inline std::array<AnyEvent, kEventCount> allEvents() noexcept {
    return {{
        fsm::PttPress{}, fsm::PttRelease{}, fsm::PttCancel{},
        fsm::DescribeScene{}, fsm::TextInput{}, fsm::VadSilence{},
        fsm::LlmFirstSentence{}, fsm::LlmComplete{}, fsm::LlmTimeout{},
        fsm::TtsComplete{},
        fsm::ModuleCrash{}, fsm::ModuleRecovered{}, fsm::RecoveryTimeout{},
        fsm::InterruptDone{},
        fsm::ImageTransformRequest{}, fsm::ImageTransformComplete{},
        fsm::ImageTransformTimeout{},
        fsm::SuperResolutionStart{}, fsm::SuperResolutionStop{},
    }};
}

[[nodiscard]] inline std::array<State, kStateCount> allStates() noexcept {
    return {{
        fsm::Idle{}, fsm::Listening{}, fsm::Processing{},
        fsm::Speaking{}, fsm::Interrupted{}, fsm::ErrorRecovery{},
        fsm::ImageTransform{}, fsm::SuperResolution{},
    }};
}

void logTransition(StateIndex from, std::string_view event, StateIndex to);
void logUnhandledEvent(StateIndex current, std::string_view event);

class StateMachine {
public:
    StateMachine() = default;

    [[nodiscard]] StateIndex currentState() const noexcept {
        return stateIndex(state_);
    }

    [[nodiscard]] const State& state() const noexcept { return state_; }

    template <typename Handler, typename EventT>
    [[nodiscard]] std::optional<StateIndex> dispatch(Handler&& handler, const EventT& event) {
        auto prev = currentState();
        auto result = std::visit([&](auto& s) -> std::optional<State> {
            return std::forward<Handler>(handler)(s, event);
        }, state_);
        if (!result) {
            logUnhandledEvent(prev, eventName<EventT>());
            return std::nullopt;
        }
        state_ = std::move(*result);
        auto curr = currentState();
        logTransition(prev, eventName<EventT>(), curr);
        return curr;
    }

    template <typename Handler>
    [[nodiscard]] std::optional<StateIndex> dispatch(Handler&& handler, const AnyEvent& event) {
        return std::visit([&](const auto& e) -> std::optional<StateIndex> {
            return this->dispatch(std::forward<Handler>(handler), e);
        }, event);
    }

    template <typename S>
    void forceState() noexcept { state_ = S{}; }

    void forceState(StateIndex idx) noexcept;

private:
    State state_{fsm::Idle{}};
};

