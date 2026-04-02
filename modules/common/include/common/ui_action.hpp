#pragma once
#include <array>
#include <string_view>

enum class UiAction : uint8_t {
    kStartAudio, kStopAudio, kEnableWebcam, kDisableWebcam, kPushToTalk,
    kTextInput, kDescribeScene, kTtsComplete, kSwitchMode,
    kSelectConversationModel, kSelectTtsModel,
    kUploadImage, kSelectStyle, kStartImageTransform,
    kCancelGeneration, kFlushTts, kStopPlayback,
    kRegisterFace, kSetImageAdjust, kSetBgMode, kUpdateShapes,
    kDescribeUploadedImage, kToggleFaceFilter, kSelectFilter,
    kToggleVideoDenoise, kToggleAudioDenoise,
    kToggleStt, kToggleTts, kToggleFaceRecognition, kToggleBackgroundBlur,
    kToggleVlm, kVlmCapture,
    kUnknown,
};

inline constexpr std::array kUiActionMap = {
    std::pair{UiAction::kStartAudio,             std::string_view{"start_audio"}},
    std::pair{UiAction::kStopAudio,              std::string_view{"stop_audio"}},
    std::pair{UiAction::kEnableWebcam,           std::string_view{"enable_webcam"}},
    std::pair{UiAction::kDisableWebcam,          std::string_view{"disable_webcam"}},
    std::pair{UiAction::kPushToTalk,             std::string_view{"push_to_talk"}},
    std::pair{UiAction::kTextInput,              std::string_view{"text_input"}},
    std::pair{UiAction::kDescribeScene,          std::string_view{"describe_scene"}},
    std::pair{UiAction::kTtsComplete,            std::string_view{"tts_complete"}},
    std::pair{UiAction::kSwitchMode,             std::string_view{"switch_mode"}},
    std::pair{UiAction::kSelectConversationModel,std::string_view{"select_conversation_model"}},
    std::pair{UiAction::kSelectTtsModel,         std::string_view{"select_tts_model"}},
    std::pair{UiAction::kUploadImage,            std::string_view{"upload_image"}},
    std::pair{UiAction::kSelectStyle,            std::string_view{"select_style"}},
    std::pair{UiAction::kStartImageTransform,    std::string_view{"start_image_transform"}},
    std::pair{UiAction::kCancelGeneration,       std::string_view{"cancel_generation"}},
    std::pair{UiAction::kFlushTts,               std::string_view{"flush_tts"}},
    std::pair{UiAction::kStopPlayback,           std::string_view{"stop_playback"}},
    std::pair{UiAction::kRegisterFace,           std::string_view{"register_face"}},
    std::pair{UiAction::kSetImageAdjust,         std::string_view{"set_image_adjust"}},
    std::pair{UiAction::kSetBgMode,              std::string_view{"set_bg_mode"}},
    std::pair{UiAction::kUpdateShapes,           std::string_view{"update_shapes"}},
    std::pair{UiAction::kDescribeUploadedImage,  std::string_view{"describe_uploaded_image"}},
    std::pair{UiAction::kToggleFaceFilter,       std::string_view{"toggle_face_filter"}},
    std::pair{UiAction::kSelectFilter,           std::string_view{"select_filter"}},
    std::pair{UiAction::kToggleVideoDenoise,     std::string_view{"toggle_video_denoise"}},
    std::pair{UiAction::kToggleAudioDenoise,     std::string_view{"toggle_audio_denoise"}},
    std::pair{UiAction::kToggleStt,              std::string_view{"toggle_stt"}},
    std::pair{UiAction::kToggleTts,              std::string_view{"toggle_tts"}},
    std::pair{UiAction::kToggleFaceRecognition,  std::string_view{"toggle_face_recognition"}},
    std::pair{UiAction::kToggleBackgroundBlur,   std::string_view{"toggle_background_blur"}},
    std::pair{UiAction::kToggleVlm,              std::string_view{"toggle_vlm"}},
    std::pair{UiAction::kVlmCapture,             std::string_view{"vlm_capture"}},
};

[[nodiscard]] constexpr UiAction parseUiAction(std::string_view s) noexcept {
    for (auto& [e, name] : kUiActionMap)
        if (s == name) return e;
    return UiAction::kUnknown;
}

[[nodiscard]] constexpr std::string_view uiActionName(UiAction a) noexcept {
    for (auto& [e, name] : kUiActionMap)
        if (e == a) return name;
    return "unknown";
}

