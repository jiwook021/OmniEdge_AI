#pragma once
#include <array>
#include <string_view>

enum class UiAction : uint8_t {
    kStartAudio, kStopAudio, kEnableWebcam, kDisableWebcam, kPushToTalk,
    kTextInput, kDescribeScene, kTtsComplete, kSwitchMode,
    kSelectConversationModel, kSelectTtsModel,
    kCancelGeneration, kFlushTts, kStopPlayback,
    kRegisterFace, kSetImageAdjust, kSetBgMode, kUpdateShapes,
    kDescribeUploadedImage, kToggleFaceFilter, kSelectFilter,
    kToggleVideoDenoise, kToggleAudioDenoise,
    kToggleStt, kToggleTts, kToggleFaceRecognition, kToggleBackgroundBlur,
    kToggleVideoConversation,
    kSetSam2BgMode, kSetSam2BgColor, kSetSam2BgImage,
    kSam2StopTracking,
    kSelectConversationSource,
    kToggleSecurityMode, kSecurityVlmAnalyze,
    kSecurityListRecordings, kSecurityListEvents,
    kSecurityUpdateClasses, kSecuritySetStyle,
    kSecuritySetRoi,
    kToggleBeauty, kSetBeautySkin, kSetBeautyShape,
    kSetBeautyLight, kSetBeautyBg, kSetBeautyPreset,
    kSwitchBootMode,
    kSnapshot,
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
    std::pair{UiAction::kToggleVideoConversation,std::string_view{"toggle_video_conversation"}},
    std::pair{UiAction::kSetSam2BgMode,         std::string_view{"set_sam2_bg_mode"}},
    std::pair{UiAction::kSetSam2BgColor,        std::string_view{"set_sam2_bg_color"}},
    std::pair{UiAction::kSetSam2BgImage,        std::string_view{"set_sam2_bg_image"}},
    std::pair{UiAction::kSam2StopTracking,      std::string_view{"sam2_stop_tracking"}},
    std::pair{UiAction::kSelectConversationSource, std::string_view{"select_conversation_source"}},
    std::pair{UiAction::kToggleSecurityMode,      std::string_view{"toggle_security_mode"}},
    std::pair{UiAction::kSecurityVlmAnalyze,      std::string_view{"security_vlm_analyze"}},
    std::pair{UiAction::kSecurityListRecordings,   std::string_view{"security_list_recordings"}},
    std::pair{UiAction::kSecurityListEvents,       std::string_view{"security_list_events"}},
    std::pair{UiAction::kSecurityUpdateClasses,    std::string_view{"security_update_classes"}},
    std::pair{UiAction::kSecuritySetStyle,         std::string_view{"security_set_style"}},
    std::pair{UiAction::kSecuritySetRoi,           std::string_view{"security_set_roi"}},
    std::pair{UiAction::kToggleBeauty,             std::string_view{"toggle_beauty"}},
    std::pair{UiAction::kSetBeautySkin,            std::string_view{"set_beauty_skin"}},
    std::pair{UiAction::kSetBeautyShape,           std::string_view{"set_beauty_shape"}},
    std::pair{UiAction::kSetBeautyLight,           std::string_view{"set_beauty_light"}},
    std::pair{UiAction::kSetBeautyBg,              std::string_view{"set_beauty_bg"}},
    std::pair{UiAction::kSetBeautyPreset,          std::string_view{"set_beauty_preset"}},
    std::pair{UiAction::kSwitchBootMode,           std::string_view{"switch_boot_mode"}},
    std::pair{UiAction::kSnapshot,                 std::string_view{"snapshot"}},
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

