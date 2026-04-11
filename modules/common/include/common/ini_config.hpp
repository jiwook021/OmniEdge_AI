#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — INI Configuration Loader
//
// Reads config/omniedge.ini at daemon startup using the inih library
// (INIReader). Modules set to 0 in [modules] are excluded from launch_order
// entirely — no process spawned, no VRAM reserved.
//
// Crash protection settings control memory watchdog behavior and segfault
// restart limits to prevent runaway modules from crashing WSL.
//
// VRAM budgets, tier thresholds, inference headroom, and interaction profile
// priorities are all configurable via INI sections.  Compile-time defaults
// from vram_thresholds.hpp and runtime_defaults.hpp serve as fallbacks when
// an INI key is absent.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

#include "INIReader.h"

#include "common/oe_logger.hpp"
#include "common/runtime_defaults.hpp"
#include "vram/vram_thresholds.hpp"


/// Crash protection configuration parsed from [crash_protection] section.
struct CrashProtectionConfig {
    int  maxSegfaultRestarts{kMaxSegfaultRestarts};
    int  memoryCheckIntervalMs{kMemoryCheckIntervalMs};
    int  maxUnresponsiveChecks{kMaxUnresponsiveChecks};
    bool killProcessGroup{true};
};

/// VRAM limit configuration parsed from [vram_limits] section.
struct VramLimitsConfig {
    std::size_t maxTotalVramMb{kVramWatchdogMaxMb};         ///< Hard cap — never exceed
    int         vramCheckIntervalMs{kVramWatchdogCheckMs};  ///< Watchdog poll interval
    std::size_t warningThresholdMb{kVramWatchdogWarningMb}; ///< Log warning above this
    std::size_t criticalThresholdMb{kVramWatchdogCriticalMb}; ///< Start evicting above this
};

/// Daemon timing configuration parsed from [daemon] section.
struct DaemonTimingConfig {
    int watchdogPollMs{kWatchdogPollMs};
    int moduleReadyTimeoutS{kModuleReadyTimeoutS};
    int maxModuleRestarts{kMaxModuleRestarts};
    int stopGracePeriodS{kStopGracePeriodS};
    int llmGenerationTimeoutS{kLlmGenerationTimeoutS};
    int imageTransformTimeoutS{kImageTransformTimeoutS};
    int vadSilenceThresholdMs{static_cast<int>(kVadSilenceDurationMs)};
};

/// Heartbeat configuration parsed from [heartbeat] section.
struct HeartbeatConfig {
    int intervalMs{kHeartbeatIntervalMs};
    int timeoutMs{kHeartbeatTimeoutMs};
    int ttlMs{kHeartbeatTtlMs};
};

/// Prompt assembly configuration parsed from [prompt] section.
struct PromptConfig {
    int maxContextTokens{kPromptMaxContextTokens};
    int systemPromptTokens{kPromptSystemTokens};
    int dynamicContextTokens{kPromptDynamicContextTokens};
    int maxUserTurnTokens{kPromptMaxUserTurnTokens};
    int generationHeadroom{kPromptGenerationHeadroom};
};

/// Session persistence configuration parsed from [session] section.
struct SessionConfig {
    int periodicSaveIntervalS{kSessionSaveIntervalS};
    int maxStalenessS{kSessionMaxStalenessS};
};

/// VRAM tier selection thresholds parsed from [vram_tiers] section.
struct VramTierConfig {
    std::size_t ultraThresholdMiB{kUltraTierThresholdMiB};
    std::size_t standardThresholdMiB{kStandardTierThresholdMiB};
    std::size_t balancedThresholdMiB{kBalancedTierThresholdMiB};
    std::size_t headroomMiB{kHeadroomMiB};
    std::size_t pressureWarningFreeMiB{kPressureWarningMiB};
    std::size_t pressureCriticalFreeMiB{kPressureCriticalMiB};
};

/// Generation parameter defaults parsed from [generation] section.
/// Used as fallback values when the frontend omits generation_params fields.
struct GenerationDefaultsConfig {
    float temperature{kGenerationDefaultTemperature};
    float topP{kGenerationDefaultTopP};
    int   maxTokens{kGenerationDefaultMaxTokens};
};

/// Computer vision quality/threshold configuration parsed from [cv] section.
struct CvConfig {
    float faceRecognitionThreshold{kFaceRecognitionThreshold};
    float segmentationConfidenceThreshold{kSegmentationConfidenceThreshold};
    int   jpegEncodingQuality{kJpegEncodingQuality};
    float backgroundBlurSigma{kBackgroundBlurSigma};
    int   faceDetectionFrameSubsample{static_cast<int>(kFaceDetectionFrameSubsample)};
};

/// Parses omniedge.ini via INIReader and provides module-enable queries + tunable config.
///
/// Header-only — no .cpp needed. Parse once at startup, query during launch.
/// All config structs default to compile-time values from runtime_defaults.hpp
/// and vram_thresholds.hpp; INI sections override them when present.
class IniConfig {
public:
    IniConfig() = default;

    /// Load and parse the INI file. Call once from OmniEdgeDaemon::parseConfig().
    /// Returns false if the file cannot be opened (all modules default to enabled).
    [[nodiscard]] bool loadFromFile(std::string_view path) {
        std::string pathStr{path};
        INIReader reader{pathStr};

        if (reader.ParseError() < 0) {
            OE_LOG_WARN("ini_config_file_not_found: path={} — all modules enabled by default", path);
            return false;
        }
        if (reader.ParseError() > 0) {
            OE_LOG_WARN("ini_config_parse_error: path={}, line={}", path, reader.ParseError());
        }

        // ── [modules] section ───────────────────────────────────────────
        for (const auto& key : reader.Keys("modules")) {
            bool enabled = reader.GetBoolean("modules", key, true);
            moduleEnabled_[key] = enabled;
            OE_LOG_INFO("ini_config: module={}, enabled={}", key, enabled);
        }

        // ── [memory_limits_mb] section ──────────────────────────────────
        for (const auto& key : reader.Keys("memory_limits_mb")) {
            auto limitMb = static_cast<std::size_t>(
                reader.GetUnsigned("memory_limits_mb", key, 0));
            memoryLimitsMb_[key] = limitMb;
        }

        // ── [crash_protection] section ──────────────────────────────────
        crashProtection_.maxSegfaultRestarts = static_cast<int>(
            reader.GetInteger("crash_protection", "max_segfault_restarts",
                              crashProtection_.maxSegfaultRestarts));
        crashProtection_.memoryCheckIntervalMs = static_cast<int>(
            reader.GetInteger("crash_protection", "memory_check_interval_ms",
                              crashProtection_.memoryCheckIntervalMs));
        crashProtection_.maxUnresponsiveChecks = static_cast<int>(
            reader.GetInteger("crash_protection", "max_unresponsive_checks",
                              crashProtection_.maxUnresponsiveChecks));
        crashProtection_.killProcessGroup =
            reader.GetBoolean("crash_protection", "kill_process_group",
                              crashProtection_.killProcessGroup);

        // ── [vram_limits] section ───────────────────────────────────────
        vramLimits_.maxTotalVramMb = static_cast<std::size_t>(
            reader.GetUnsigned("vram_limits", "max_total_vram_mb",
                               static_cast<unsigned long>(vramLimits_.maxTotalVramMb)));
        vramLimits_.vramCheckIntervalMs = static_cast<int>(
            reader.GetInteger("vram_limits", "vram_check_interval_ms",
                              vramLimits_.vramCheckIntervalMs));
        vramLimits_.warningThresholdMb = static_cast<std::size_t>(
            reader.GetUnsigned("vram_limits", "vram_warning_threshold_mb",
                               static_cast<unsigned long>(vramLimits_.warningThresholdMb)));
        vramLimits_.criticalThresholdMb = static_cast<std::size_t>(
            reader.GetUnsigned("vram_limits", "vram_critical_threshold_mb",
                               static_cast<unsigned long>(vramLimits_.criticalThresholdMb)));

        // ── [vram_tiers] section ────────────────────────────────────────
        vramTiers_.ultraThresholdMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_tiers", "ultra_threshold_mb",
                               static_cast<unsigned long>(vramTiers_.ultraThresholdMiB)));
        vramTiers_.standardThresholdMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_tiers", "standard_threshold_mb",
                               static_cast<unsigned long>(vramTiers_.standardThresholdMiB)));
        vramTiers_.balancedThresholdMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_tiers", "balanced_threshold_mb",
                               static_cast<unsigned long>(vramTiers_.balancedThresholdMiB)));
        vramTiers_.headroomMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_tiers", "headroom_mb",
                               static_cast<unsigned long>(vramTiers_.headroomMiB)));
        vramTiers_.pressureWarningFreeMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_limits", "pressure_warning_free_mb",
                               static_cast<unsigned long>(vramTiers_.pressureWarningFreeMiB)));
        vramTiers_.pressureCriticalFreeMiB = static_cast<std::size_t>(
            reader.GetUnsigned("vram_limits", "pressure_critical_free_mb",
                               static_cast<unsigned long>(vramTiers_.pressureCriticalFreeMiB)));

        // ── [vram_budgets] section ──────────────────────────────────────
        for (const auto& key : reader.Keys("vram_budgets")) {
            auto budgetMiB = static_cast<std::size_t>(
                reader.GetUnsigned("vram_budgets", key, 0));
            vramBudgetsMiB_[key] = budgetMiB;
        }

        // ── [vram_inference_headroom] section ───────────────────────────
        for (const auto& key : reader.Keys("vram_inference_headroom")) {
            auto headroomMiB = static_cast<std::size_t>(
                reader.GetUnsigned("vram_inference_headroom", key, 0));
            vramInferenceHeadroomMiB_[key] = headroomMiB;
        }

        // ── [profile_*] sections — interaction profile priorities ───────
        auto parseProfileSection = [&](const std::string& section,
                                       std::unordered_map<std::string, int>& out) {
            for (const auto& key : reader.Keys(section)) {
                out[key] = static_cast<int>(
                    reader.GetInteger(section, key, 0));
            }
        };
        parseProfileSection("profile_conversation",      profileConversation_);
        parseProfileSection("profile_super_resolution",   profileSuperResolution_);
        parseProfileSection("profile_image_transform",    profileImageTransform_);
        parseProfileSection("profile_sam2_segmentation", profileSam2Segmentation_);

        // ── [daemon] section ────────────────────────────────────────────
        daemonTiming_.watchdogPollMs = static_cast<int>(
            reader.GetInteger("daemon", "watchdog_poll_ms",
                              daemonTiming_.watchdogPollMs));
        daemonTiming_.moduleReadyTimeoutS = static_cast<int>(
            reader.GetInteger("daemon", "module_ready_timeout_s",
                              daemonTiming_.moduleReadyTimeoutS));
        daemonTiming_.maxModuleRestarts = static_cast<int>(
            reader.GetInteger("daemon", "max_module_restarts",
                              daemonTiming_.maxModuleRestarts));
        daemonTiming_.stopGracePeriodS = static_cast<int>(
            reader.GetInteger("daemon", "stop_grace_period_s",
                              daemonTiming_.stopGracePeriodS));
        // ── [timeouts] section ─────────────────────────────────────────
        daemonTiming_.llmGenerationTimeoutS = static_cast<int>(
            reader.GetInteger("timeouts", "llm_generation_s",
                              daemonTiming_.llmGenerationTimeoutS));
        daemonTiming_.imageTransformTimeoutS = static_cast<int>(
            reader.GetInteger("timeouts", "image_transform_s",
                              daemonTiming_.imageTransformTimeoutS));
        daemonTiming_.vadSilenceThresholdMs = static_cast<int>(
            reader.GetInteger("timeouts", "vad_silence_ms",
                              daemonTiming_.vadSilenceThresholdMs));

        // ── [heartbeat] section ─────────────────────────────────────────
        heartbeat_.intervalMs = static_cast<int>(
            reader.GetInteger("heartbeat", "interval_ms", heartbeat_.intervalMs));
        heartbeat_.timeoutMs = static_cast<int>(
            reader.GetInteger("heartbeat", "timeout_ms", heartbeat_.timeoutMs));
        heartbeat_.ttlMs = static_cast<int>(
            reader.GetInteger("heartbeat", "ttl_ms", heartbeat_.ttlMs));

        // ── [prompt] section ────────────────────────────────────────────
        prompt_.maxContextTokens = static_cast<int>(
            reader.GetInteger("prompt", "max_context_tokens",
                              prompt_.maxContextTokens));
        prompt_.systemPromptTokens = static_cast<int>(
            reader.GetInteger("prompt", "system_prompt_tokens",
                              prompt_.systemPromptTokens));
        prompt_.dynamicContextTokens = static_cast<int>(
            reader.GetInteger("prompt", "dynamic_context_tokens",
                              prompt_.dynamicContextTokens));
        prompt_.maxUserTurnTokens = static_cast<int>(
            reader.GetInteger("prompt", "max_user_turn_tokens",
                              prompt_.maxUserTurnTokens));
        prompt_.generationHeadroom = static_cast<int>(
            reader.GetInteger("prompt", "generation_headroom",
                              prompt_.generationHeadroom));

        // ── [session] section ───────────────────────────────────────────
        session_.periodicSaveIntervalS = static_cast<int>(
            reader.GetInteger("session", "periodic_save_interval_s",
                              session_.periodicSaveIntervalS));
        session_.maxStalenessS = static_cast<int>(
            reader.GetInteger("session", "max_staleness_s",
                              session_.maxStalenessS));

        // ── [cv] section ────────────────────────────────────────────────
        cv_.faceRecognitionThreshold = static_cast<float>(
            reader.GetReal("cv", "face_recognition_threshold",
                           cv_.faceRecognitionThreshold));
        cv_.segmentationConfidenceThreshold = static_cast<float>(
            reader.GetReal("cv", "segmentation_confidence_threshold",
                           cv_.segmentationConfidenceThreshold));
        cv_.jpegEncodingQuality = static_cast<int>(
            reader.GetInteger("cv", "jpeg_encoding_quality",
                              cv_.jpegEncodingQuality));
        cv_.backgroundBlurSigma = static_cast<float>(
            reader.GetReal("cv", "background_blur_sigma",
                           cv_.backgroundBlurSigma));
        cv_.faceDetectionFrameSubsample = static_cast<int>(
            reader.GetInteger("cv", "face_detection_frame_subsample",
                              cv_.faceDetectionFrameSubsample));

        // ── [generation] section ────────────────────────────────────
        generation_.temperature = static_cast<float>(
            reader.GetReal("generation", "temperature",
                           generation_.temperature));
        generation_.topP = static_cast<float>(
            reader.GetReal("generation", "top_p",
                           generation_.topP));
        generation_.maxTokens = static_cast<int>(
            reader.GetInteger("generation", "max_tokens",
                              generation_.maxTokens));

        loaded_ = true;
        OE_LOG_INFO("ini_config_loaded: path={}, modules_configured={}, "
                   "memory_limits={}, vram_budgets={}, profiles_loaded={}",
                   path, moduleEnabled_.size(), memoryLimitsMb_.size(),
                   vramBudgetsMiB_.size(),
                   !profileConversation_.empty() || !profileSuperResolution_.empty()
                       || !profileImageTransform_.empty());
        return true;
    }

    /// Returns true if the module should be launched. If the INI file was not
    /// loaded or the module is not listed, defaults to true (enabled).
    [[nodiscard]] bool isModuleEnabled(std::string_view moduleName) const {
        if (!loaded_) return true;
        auto it = moduleEnabled_.find(std::string(moduleName));
        if (it == moduleEnabled_.end()) return true;  // unlisted = enabled
        return it->second;
    }

    /// Returns the RSS memory limit in MB for a module, or 0 if no limit set.
    [[nodiscard]] std::size_t memoryLimitMb(std::string_view moduleName) const {
        auto it = memoryLimitsMb_.find(std::string(moduleName));
        if (it == memoryLimitsMb_.end()) return 0;
        return it->second;
    }

    /// Crash protection settings.
    [[nodiscard]] const CrashProtectionConfig& crashProtection() const noexcept {
        return crashProtection_;
    }

    /// VRAM limit settings.
    [[nodiscard]] const VramLimitsConfig& vramLimits() const noexcept {
        return vramLimits_;
    }

    /// Daemon timing settings (watchdog, module lifecycle, generation timeouts).
    [[nodiscard]] const DaemonTimingConfig& daemonTiming() const noexcept {
        return daemonTiming_;
    }

    /// Heartbeat settings (interval, timeout, TTL).
    [[nodiscard]] const HeartbeatConfig& heartbeat() const noexcept {
        return heartbeat_;
    }

    /// Prompt assembly token budget settings.
    [[nodiscard]] const PromptConfig& prompt() const noexcept {
        return prompt_;
    }

    /// Session persistence settings.
    [[nodiscard]] const SessionConfig& session() const noexcept {
        return session_;
    }

    /// Computer vision quality/threshold settings.
    [[nodiscard]] const CvConfig& cv() const noexcept {
        return cv_;
    }

    /// Generation parameter defaults (fallbacks when frontend omits fields).
    [[nodiscard]] const GenerationDefaultsConfig& generation() const noexcept {
        return generation_;
    }

    /// VRAM tier selection thresholds.
    [[nodiscard]] const VramTierConfig& vramTiers() const noexcept {
        return vramTiers_;
    }

    /// Per-module VRAM budget in MiB.  Falls back to the given compile-time
    /// default if the module key is not present in [vram_budgets].
    [[nodiscard]] std::size_t vramBudgetMiB(std::string_view module,
                                            std::size_t fallback) const {
        auto it = vramBudgetsMiB_.find(std::string(module));
        return it != vramBudgetsMiB_.end() ? it->second : fallback;
    }

    /// Per-module inference headroom in MiB.  Falls back to the given
    /// compile-time default if the module key is not in [vram_inference_headroom].
    [[nodiscard]] std::size_t vramInferenceHeadroomMiB(
        std::string_view module, std::size_t fallback) const {
        auto it = vramInferenceHeadroomMiB_.find(std::string(module));
        return it != vramInferenceHeadroomMiB_.end() ? it->second : fallback;
    }

    /// Interaction profile priority map for the given profile.
    /// Pre-initialised with compile-time defaults; overridden by
    /// [profile_conversation], [profile_super_resolution],
    /// [profile_image_transform], or [profile_sam2_segmentation]
    /// INI sections when present.
    [[nodiscard]] const std::unordered_map<std::string, int>&
    profilePriorities(InteractionProfile profile) const noexcept {
        switch (profile) {
        case InteractionProfile::kSuperResolution:  return profileSuperResolution_;
        case InteractionProfile::kImageTransform:   return profileImageTransform_;
        case InteractionProfile::kSam2Segmentation: return profileSam2Segmentation_;
        case InteractionProfile::kConversation:
        default:                                    return profileConversation_;
        }
    }

    /// Whether the INI file was successfully loaded.
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }

private:
    bool loaded_{false};
    std::unordered_map<std::string, bool>        moduleEnabled_;
    std::unordered_map<std::string, std::size_t> memoryLimitsMb_;
    CrashProtectionConfig                        crashProtection_;
    VramLimitsConfig                             vramLimits_;
    VramTierConfig                               vramTiers_;
    DaemonTimingConfig                           daemonTiming_;
    HeartbeatConfig                              heartbeat_;
    PromptConfig                                 prompt_;
    SessionConfig                                session_;
    CvConfig                                     cv_;
    GenerationDefaultsConfig                     generation_;
    std::unordered_map<std::string, std::size_t> vramBudgetsMiB_;
    std::unordered_map<std::string, std::size_t> vramInferenceHeadroomMiB_;
    // Default profile priorities — overridden by [profile_*] INI sections.
    // Priority 5 = never evict, 0 = evict first.
    std::unordered_map<std::string, int> profileConversation_{
        {"conversation_model", 5}, {"tts", 4}, {"audio_denoise", 1},
        {"background_blur", 5}, {"face_recognition", 5},
        {"super_resolution", 0}, {"image_transform", 0}, {"sam2", 2},
        {"llm", 0}, {"stt", 0}, {"video_denoise", 0},
    };
    std::unordered_map<std::string, int> profileSuperResolution_{
        {"super_resolution", 5}, {"background_blur", 5}, {"face_recognition", 5},
        {"conversation_model", 0}, {"tts", 0}, {"audio_denoise", 0},
        {"image_transform", 0}, {"sam2", 0}, {"llm", 0}, {"stt", 0},
    };
    std::unordered_map<std::string, int> profileImageTransform_{
        {"image_transform", 5}, {"background_blur", 5}, {"face_recognition", 5},
        {"conversation_model", 0}, {"super_resolution", 0}, {"tts", 0},
        {"audio_denoise", 0}, {"sam2", 0}, {"llm", 0}, {"stt", 0},
    };
    std::unordered_map<std::string, int> profileSam2Segmentation_{
        {"sam2", 5}, {"background_blur", 5}, {"face_recognition", 5},
        {"conversation_model", 0}, {"super_resolution", 0}, {"tts", 0},
        {"audio_denoise", 0}, {"image_transform", 0}, {"llm", 0}, {"stt", 0},
    };
};

