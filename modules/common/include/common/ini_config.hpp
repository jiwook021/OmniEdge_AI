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

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "INIReader.h"

#include "common/oe_logger.hpp"
#include "common/runtime_defaults.hpp"
#include "common/constants/ingest_constants.hpp"
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
    int vadSilenceThresholdMs{static_cast<int>(kVadSilenceDurationMs)};
    int restartBaseBackoffMs{static_cast<int>(kRestartBaseBackoffMs)};
    int restartMaxBackoffMs{static_cast<int>(kRestartMaxBackoffMs)};
    bool bargeInEnabled{true};
    std::string defaultMode{"simple_llm"};
    std::vector<std::string> launchOrder;
    std::vector<std::string> subEndpoints;
};

/// Per-mode launch orders parsed from [boot_modes] section.
struct BootModesConfig {
    std::unordered_map<std::string, std::vector<std::string>> modes;
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

/// Graph-based pipeline orchestrator configuration parsed from [orchestrator_graph] section.
struct OrchestratorGraphConfig {
    bool        shadowMode{true};
    std::size_t drainTimeoutMs{kOrchestratorDrainTimeoutMs};
    std::size_t connectRetryCount{10};
    std::size_t connectRetryIntervalMs{500};
};

/// Filesystem paths parsed from [paths] section.
struct PathsConfig {
    std::string modelsRoot{"/opt/omniedge/models"};
};

/// Text-to-speech (Kokoro) configuration parsed from [tts] section.
/// Relative file paths are resolved against [paths] models_root.
struct TtsConfig {
    std::string onnxModel{"kokoro-onnx/onnx/model_int8.onnx"};
    std::string voiceDir{"kokoro/voices"};
    std::string defaultVoice{"af_heart"};
    float       speed{1.0F};
    std::string shmOutput{"/oe.aud.tts"};
    int         cudaStreamPriority{0};
};

/// Video ingest (V4L2 → SHM) configuration parsed from [video_ingest] section.
struct VideoIngestConfig {
    std::string v4l2Device{"/dev/video0"};
    int         frameWidth{1920};
    int         frameHeight{1080};
    int         zmqSendHighWaterMark{2};
    std::string shmInput{"/oe.vid.ingest"};
};

/// Audio ingest configuration parsed from [audio_ingest] section.
/// VAD parameters use the vad_* key prefix (flattened sub-section).
struct AudioIngestConfig {
    std::string audioSource{"auto"};
    std::string windowsHostIp{"172.17.0.1"};
    int         tcpPort{kAudioIngestTcpPort};
    int         sampleRateHz{16000};
    int         chunkSamples{512};
    int         zmqSendHighWaterMark{4};
    std::string vadModelPath{"vad/silero_vad.onnx"};
    float       vadSpeechThreshold{0.5F};
    int         vadSilenceDurationMs{800};
};

/// Screen ingest (TCP from Windows host) configuration parsed from [screen_ingest] section.
struct ScreenIngestConfig {
    std::string windowsHostIp{"host.docker.internal"};
    int         tcpPort{kScreenIngestTcpPort};
    int         healthTimeoutSec{5};
    std::string screenCaptureExe;
};

/// Audio denoise (DTLN two-stage ONNX) configuration parsed from [audio_denoise] section.
/// Relative file paths are resolved against [paths] models_root.
struct AudioDenoiseConfig {
    std::string modelStage1{"dtln/dtln_1.onnx"};
    std::string modelStage2{"dtln/dtln_2.onnx"};
    std::string shmInput{"/oe.aud.ingest"};
    std::string shmOutput{"/oe.aud.denoise"};
    int         cudaStreamPriority{0};
};

/// WebSocket bridge configuration parsed from [websocket_bridge] section.
/// ZMQ PUB port is read from [ports] websocket_bridge; HTTP port from [ports] ws_http.
struct WebsocketBridgeConfig {
    int         wsPort{8080};
    std::string frontendDir{"/opt/omniedge/share/frontend"};
};

/// Face recognition node + model-fetch configuration parsed from [face_recognition].
/// Model-fetch fields (model_repo/file, auto_download, cache_dir) are read
/// directly by OnnxFaceRecogInferencer::loadModel(); mirrored here for
/// completeness and explicit defaults.
struct FaceRecognitionConfig {
    std::string modelPackPath{"face_models/scrfd_auraface"};
    std::string shmInput{"/oe.vid.ingest"};
    float       recognitionThreshold{0.45F};
    std::string facesDb{"./data/known_faces.sqlite"};
    int         frameSubsample{3};
    int         cudaStreamPriority{0};
};

/// Background blur configuration parsed from [background_blur] section.
struct BackgroundBlurConfig {
    std::string enginePath{"bg_blur/mediapipe_selfie_seg.onnx"};
    int         inputSize{256};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.cv.blur.jpeg"};
    std::string shmOutputBgr{"/oe.cv.blur.bgr"};
    std::string inputTopic{"video_frame"};
    int         jpegQuality{85};
    int         blurKernelSize{51};
    float       blurSigma{25.0F};
    int         cudaStreamPriority{0};
    std::string outputFormat{"jpeg"};
};

/// Face filter (AR) configuration parsed from [face_filter] section.
struct FaceFilterConfig {
    std::string modelPath{"facemesh/face_landmarks_detector.onnx"};
    std::string filterManifest{"assets/filters/manifest.json"};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.cv.facefilter.jpeg"};
    int         jpegQuality{85};
    std::string activeFilter{};
    bool        enabled{false};
    int         cudaStreamPriority{0};
};

/// SAM2 segmentation configuration parsed from [sam2] section.
struct Sam2Config {
    std::string encoderModel{"sam2/sam2_hiera_tiny_encoder.onnx"};
    std::string decoderModel{"sam2/sam2_hiera_tiny_decoder.onnx"};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.cv.sam2.mask"};
    int         jpegQuality{90};
    bool        enabled{false};
    int         cudaStreamPriority{0};
};

/// Video denoise (BasicVSR++) configuration parsed from [video_denoise] section.
struct VideoDenoiseConfig {
    std::string onnxModel{"basicvsrpp/basicvsrpp_denoise.onnx"};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.vid.denoise"};
    int         temporalWindow{5};
    int         cudaStreamPriority{-1};
};

/// Security camera configuration parsed from [security_camera] section.
/// target_classes is comma-separated in INI.
struct SecurityCameraConfig {
    std::string enginePath{"security/yolox_nano.onnx"};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.cv.security.jpeg"};
    int         jpegQuality{85};
    float       confidenceThreshold{0.5F};
    int         detectionIntervalMs{100};
    int         eventCooldownMs{3000};
    std::string recordingDir{"recordings/"};
    int         segmentDurationMin{5};
    int         recordingFps{15};
    int         recordingBitrate{2000000};
    std::vector<std::string> targetClasses{"person", "backpack", "suitcase"};
};

/// LP agent loop (YOLO event → clip → Gemma VLM → incident) — [security_vlm].
struct SecurityVlmConfig {
    std::string modelDir{"gemma/gemma-3-4b-it"};  // TODO: confirm HF repo after model selection
    std::string scriptPathOverride{};             // empty → auto-resolve vision_generate.py
    int         prerollSec{3};
    int         postrollSec{2};
    int         frameSamples{6};
    int         ingestFps{5};
    int         jpegDownscale{336};
    int         jpegQuality{80};
    int         startupTimeoutSec{180};
    int         requestTimeoutMs{60000};
    int         idleUnloadSec{120};
    int         maxPendingEvents{8};
    std::string logDir{"logs/"};
    std::string promptTemplate{};                 // empty → use built-in LP prompt
};

/// Speech-to-text (Whisper) configuration parsed from [stt] and
/// [stt.hallucination_filter] sections.
struct SttConfig {
    std::string encoderEngineDir{"whisper/encoder_fp16"};
    std::string decoderEngineDir{"whisper/decoder_fp16"};
    std::string tokenizerDir{"whisper/tokenizer"};
    std::string shmInput{"/oe.aud.ingest"};
    float       noSpeechProbThreshold{0.6F};
    float       minAvgLogprob{-1.0F};
    int         maxConsecutiveRepeats{3};
};

/// Conversation (LLM) node configuration parsed from [conversation] section.
struct ConversationConfig {
    std::string shmInputAudio{"/oe.aud.ingest"};
    std::string shmInputText{"/oe.conv.text"};
    std::string defaultVariant{"gemma_e4b"};
};

/// Beauty mode configuration parsed from [beauty] section.
struct BeautyConfig {
    std::string modelPath{"facemesh/face_landmarks_detector.onnx"};
    std::string shmInput{"/oe.vid.ingest"};
    std::string shmOutput{"/oe.cv.beauty.jpeg"};
    int         jpegQuality{85};
    bool        enabled{true};
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
        parseProfileSection("profile_sam2_segmentation", profileSam2Segmentation_);
        parseProfileSection("profile_vision_model",      profileVisionModel_);

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
        daemonTiming_.defaultMode =
            reader.Get("daemon", "default_mode", daemonTiming_.defaultMode);
        daemonTiming_.restartBaseBackoffMs = static_cast<int>(
            reader.GetInteger("daemon", "restart_base_backoff_ms",
                              daemonTiming_.restartBaseBackoffMs));
        daemonTiming_.restartMaxBackoffMs = static_cast<int>(
            reader.GetInteger("daemon", "restart_max_backoff_ms",
                              daemonTiming_.restartMaxBackoffMs));
        daemonTiming_.bargeInEnabled =
            reader.GetBoolean("daemon", "barge_in_enabled",
                              daemonTiming_.bargeInEnabled);
        daemonTiming_.launchOrder =
            splitCsv(reader.Get("daemon", "launch_order", ""));
        daemonTiming_.subEndpoints =
            splitCsv(reader.Get("daemon", "zmq_sub", ""));

        // ── [boot_modes] section ───────────────────────────────────────
        for (const auto& mode : reader.Keys("boot_modes")) {
            bootModes_.modes[mode] =
                splitCsv(reader.Get("boot_modes", mode, ""));
        }
        // ── [timeouts] section ─────────────────────────────────────────
        daemonTiming_.llmGenerationTimeoutS = static_cast<int>(
            reader.GetInteger("timeouts", "llm_generation_s",
                              daemonTiming_.llmGenerationTimeoutS));
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

        // ── [paths] section ─────────────────────────────────────────
        paths_.modelsRoot =
            reader.Get("paths", "models_root", paths_.modelsRoot);

        // ── [ports] section ─────────────────────────────────────────
        for (const auto& key : reader.Keys("ports")) {
            auto port = static_cast<int>(
                reader.GetInteger("ports", key, 0));
            ports_[key] = port;
        }

        // ── [tts] section ───────────────────────────────────────────
        tts_.onnxModel =
            reader.Get("tts", "onnx_model", tts_.onnxModel);
        tts_.voiceDir =
            reader.Get("tts", "voice_dir", tts_.voiceDir);
        tts_.defaultVoice =
            reader.Get("tts", "default_voice", tts_.defaultVoice);
        tts_.speed = static_cast<float>(
            reader.GetReal("tts", "speed", tts_.speed));
        tts_.shmOutput =
            reader.Get("tts", "shm_output", tts_.shmOutput);
        tts_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("tts", "cuda_stream_priority",
                              tts_.cudaStreamPriority));

        // ── [video_ingest] section ──────────────────────────────────
        videoIngest_.v4l2Device =
            reader.Get("video_ingest", "v4l2_device", videoIngest_.v4l2Device);
        videoIngest_.frameWidth = static_cast<int>(
            reader.GetInteger("video_ingest", "frame_width",
                              videoIngest_.frameWidth));
        videoIngest_.frameHeight = static_cast<int>(
            reader.GetInteger("video_ingest", "frame_height",
                              videoIngest_.frameHeight));
        videoIngest_.zmqSendHighWaterMark = static_cast<int>(
            reader.GetInteger("video_ingest", "zmq_sndhwm",
                              videoIngest_.zmqSendHighWaterMark));
        videoIngest_.shmInput =
            reader.Get("video_ingest", "shm_input", videoIngest_.shmInput);

        // ── [audio_ingest] section ──────────────────────────────────
        audioIngest_.audioSource =
            reader.Get("audio_ingest", "audio_source", audioIngest_.audioSource);
        audioIngest_.windowsHostIp =
            reader.Get("audio_ingest", "windows_host_ip",
                       audioIngest_.windowsHostIp);
        audioIngest_.tcpPort = static_cast<int>(
            reader.GetInteger("audio_ingest", "tcp_port",
                              audioIngest_.tcpPort));
        audioIngest_.sampleRateHz = static_cast<int>(
            reader.GetInteger("audio_ingest", "sample_rate_hz",
                              audioIngest_.sampleRateHz));
        audioIngest_.chunkSamples = static_cast<int>(
            reader.GetInteger("audio_ingest", "chunk_samples",
                              audioIngest_.chunkSamples));
        audioIngest_.zmqSendHighWaterMark = static_cast<int>(
            reader.GetInteger("audio_ingest", "zmq_sndhwm",
                              audioIngest_.zmqSendHighWaterMark));
        audioIngest_.vadModelPath =
            reader.Get("audio_ingest", "vad_model_path",
                       audioIngest_.vadModelPath);
        audioIngest_.vadSpeechThreshold = static_cast<float>(
            reader.GetReal("audio_ingest", "vad_speech_threshold",
                           audioIngest_.vadSpeechThreshold));
        audioIngest_.vadSilenceDurationMs = static_cast<int>(
            reader.GetInteger("audio_ingest", "vad_silence_duration_ms",
                              audioIngest_.vadSilenceDurationMs));

        // ── [screen_ingest] section ─────────────────────────────────
        screenIngest_.windowsHostIp =
            reader.Get("screen_ingest", "windows_host_ip",
                       screenIngest_.windowsHostIp);
        screenIngest_.tcpPort = static_cast<int>(
            reader.GetInteger("screen_ingest", "tcp_port",
                              screenIngest_.tcpPort));
        screenIngest_.healthTimeoutSec = static_cast<int>(
            reader.GetInteger("screen_ingest", "health_timeout_sec",
                              screenIngest_.healthTimeoutSec));
        screenIngest_.screenCaptureExe =
            reader.Get("screen_ingest", "screen_capture_exe",
                       screenIngest_.screenCaptureExe);

        // ── [audio_denoise] section ─────────────────────────────────
        audioDenoise_.modelStage1 =
            reader.Get("audio_denoise", "model_stage1",
                       audioDenoise_.modelStage1);
        audioDenoise_.modelStage2 =
            reader.Get("audio_denoise", "model_stage2",
                       audioDenoise_.modelStage2);
        audioDenoise_.shmInput =
            reader.Get("audio_denoise", "shm_input",
                       audioDenoise_.shmInput);
        audioDenoise_.shmOutput =
            reader.Get("audio_denoise", "shm_output",
                       audioDenoise_.shmOutput);
        audioDenoise_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("audio_denoise", "cuda_stream_priority",
                              audioDenoise_.cudaStreamPriority));

        // ── [websocket_bridge] section ──────────────────────────────
        wsBridge_.frontendDir =
            reader.Get("websocket_bridge", "frontend_dir",
                       wsBridge_.frontendDir);
        wsBridge_.wsPort = static_cast<int>(
            reader.GetInteger("websocket_bridge", "ws_port", wsBridge_.wsPort));

        // ── [face_recognition] section (node-level knobs) ───────────
        faceRecognition_.modelPackPath =
            reader.Get("face_recognition", "model_pack_path",
                       faceRecognition_.modelPackPath);
        faceRecognition_.shmInput =
            reader.Get("face_recognition", "shm_input",
                       faceRecognition_.shmInput);
        faceRecognition_.recognitionThreshold = static_cast<float>(
            reader.GetReal("face_recognition", "recognition_threshold",
                           faceRecognition_.recognitionThreshold));
        faceRecognition_.facesDb =
            reader.Get("face_recognition", "faces_db",
                       faceRecognition_.facesDb);
        faceRecognition_.frameSubsample = static_cast<int>(
            reader.GetInteger("face_recognition", "frame_subsample",
                              faceRecognition_.frameSubsample));
        faceRecognition_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("face_recognition", "cuda_stream_priority",
                              faceRecognition_.cudaStreamPriority));

        // ── [background_blur] section ───────────────────────────────
        backgroundBlur_.enginePath =
            reader.Get("background_blur", "engine_path",
                       backgroundBlur_.enginePath);
        backgroundBlur_.inputSize = static_cast<int>(
            reader.GetInteger("background_blur", "input_size",
                              backgroundBlur_.inputSize));
        backgroundBlur_.shmInput =
            reader.Get("background_blur", "shm_input",
                       backgroundBlur_.shmInput);
        backgroundBlur_.shmOutput =
            reader.Get("background_blur", "shm_output",
                       backgroundBlur_.shmOutput);
        backgroundBlur_.shmOutputBgr =
            reader.Get("background_blur", "shm_output_bgr",
                       backgroundBlur_.shmOutputBgr);
        backgroundBlur_.inputTopic =
            reader.Get("background_blur", "input_topic",
                       backgroundBlur_.inputTopic);
        backgroundBlur_.jpegQuality = static_cast<int>(
            reader.GetInteger("background_blur", "jpeg_quality",
                              backgroundBlur_.jpegQuality));
        backgroundBlur_.blurKernelSize = static_cast<int>(
            reader.GetInteger("background_blur", "blur_kernel_size",
                              backgroundBlur_.blurKernelSize));
        backgroundBlur_.blurSigma = static_cast<float>(
            reader.GetReal("background_blur", "blur_sigma",
                           backgroundBlur_.blurSigma));
        backgroundBlur_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("background_blur", "cuda_stream_priority",
                              backgroundBlur_.cudaStreamPriority));
        backgroundBlur_.outputFormat =
            reader.Get("background_blur", "output_format",
                       backgroundBlur_.outputFormat);

        // ── [face_filter] section ───────────────────────────────────
        faceFilter_.modelPath =
            reader.Get("face_filter", "model_path", faceFilter_.modelPath);
        faceFilter_.filterManifest =
            reader.Get("face_filter", "filter_manifest",
                       faceFilter_.filterManifest);
        faceFilter_.shmInput =
            reader.Get("face_filter", "shm_input", faceFilter_.shmInput);
        faceFilter_.shmOutput =
            reader.Get("face_filter", "shm_output", faceFilter_.shmOutput);
        faceFilter_.jpegQuality = static_cast<int>(
            reader.GetInteger("face_filter", "jpeg_quality",
                              faceFilter_.jpegQuality));
        faceFilter_.activeFilter =
            reader.Get("face_filter", "active_filter",
                       faceFilter_.activeFilter);
        faceFilter_.enabled =
            reader.GetBoolean("face_filter", "enabled", faceFilter_.enabled);
        faceFilter_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("face_filter", "cuda_stream_priority",
                              faceFilter_.cudaStreamPriority));

        // ── [sam2] section ──────────────────────────────────────────
        sam2_.encoderModel =
            reader.Get("sam2", "encoder_model", sam2_.encoderModel);
        sam2_.decoderModel =
            reader.Get("sam2", "decoder_model", sam2_.decoderModel);
        sam2_.shmInput =
            reader.Get("sam2", "shm_input", sam2_.shmInput);
        sam2_.shmOutput =
            reader.Get("sam2", "shm_output", sam2_.shmOutput);
        sam2_.jpegQuality = static_cast<int>(
            reader.GetInteger("sam2", "jpeg_quality", sam2_.jpegQuality));
        sam2_.enabled =
            reader.GetBoolean("sam2", "enabled", sam2_.enabled);
        sam2_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("sam2", "cuda_stream_priority",
                              sam2_.cudaStreamPriority));

        // ── [video_denoise] section ─────────────────────────────────
        videoDenoise_.onnxModel =
            reader.Get("video_denoise", "onnx_model",
                       videoDenoise_.onnxModel);
        videoDenoise_.shmInput =
            reader.Get("video_denoise", "shm_input",
                       videoDenoise_.shmInput);
        videoDenoise_.shmOutput =
            reader.Get("video_denoise", "shm_output",
                       videoDenoise_.shmOutput);
        videoDenoise_.temporalWindow = static_cast<int>(
            reader.GetInteger("video_denoise", "temporal_window",
                              videoDenoise_.temporalWindow));
        videoDenoise_.cudaStreamPriority = static_cast<int>(
            reader.GetInteger("video_denoise", "cuda_stream_priority",
                              videoDenoise_.cudaStreamPriority));

        // ── [security_camera] section ───────────────────────────────
        securityCamera_.enginePath =
            reader.Get("security_camera", "engine_path",
                       securityCamera_.enginePath);
        securityCamera_.shmInput =
            reader.Get("security_camera", "shm_input",
                       securityCamera_.shmInput);
        securityCamera_.shmOutput =
            reader.Get("security_camera", "shm_output",
                       securityCamera_.shmOutput);
        securityCamera_.jpegQuality = static_cast<int>(
            reader.GetInteger("security_camera", "jpeg_quality",
                              securityCamera_.jpegQuality));
        securityCamera_.confidenceThreshold = static_cast<float>(
            reader.GetReal("security_camera", "confidence_threshold",
                           securityCamera_.confidenceThreshold));
        securityCamera_.detectionIntervalMs = static_cast<int>(
            reader.GetInteger("security_camera", "detection_interval_ms",
                              securityCamera_.detectionIntervalMs));
        securityCamera_.eventCooldownMs = static_cast<int>(
            reader.GetInteger("security_camera", "event_cooldown_ms",
                              securityCamera_.eventCooldownMs));
        securityCamera_.recordingDir =
            reader.Get("security_camera", "recording_dir",
                       securityCamera_.recordingDir);
        securityCamera_.segmentDurationMin = static_cast<int>(
            reader.GetInteger("security_camera", "segment_duration_min",
                              securityCamera_.segmentDurationMin));
        securityCamera_.recordingFps = static_cast<int>(
            reader.GetInteger("security_camera", "recording_fps",
                              securityCamera_.recordingFps));
        securityCamera_.recordingBitrate = static_cast<int>(
            reader.GetInteger("security_camera", "recording_bitrate",
                              securityCamera_.recordingBitrate));
        {
            std::string csv = reader.Get("security_camera", "target_classes", "");
            if (!csv.empty()) {
                securityCamera_.targetClasses = splitCsv(csv);
            }
        }

        // ── [security_vlm] section ──────────────────────────────────
        securityVlm_.modelDir =
            reader.Get("security_vlm", "model_dir", securityVlm_.modelDir);
        securityVlm_.scriptPathOverride =
            reader.Get("security_vlm", "script_path", securityVlm_.scriptPathOverride);
        securityVlm_.prerollSec = static_cast<int>(
            reader.GetInteger("security_vlm", "preroll_sec", securityVlm_.prerollSec));
        securityVlm_.postrollSec = static_cast<int>(
            reader.GetInteger("security_vlm", "postroll_sec", securityVlm_.postrollSec));
        securityVlm_.frameSamples = static_cast<int>(
            reader.GetInteger("security_vlm", "frame_samples", securityVlm_.frameSamples));
        securityVlm_.ingestFps = static_cast<int>(
            reader.GetInteger("security_vlm", "ingest_fps", securityVlm_.ingestFps));
        securityVlm_.jpegDownscale = static_cast<int>(
            reader.GetInteger("security_vlm", "jpeg_downscale", securityVlm_.jpegDownscale));
        securityVlm_.jpegQuality = static_cast<int>(
            reader.GetInteger("security_vlm", "jpeg_quality", securityVlm_.jpegQuality));
        securityVlm_.startupTimeoutSec = static_cast<int>(
            reader.GetInteger("security_vlm", "startup_timeout_sec",
                              securityVlm_.startupTimeoutSec));
        securityVlm_.requestTimeoutMs = static_cast<int>(
            reader.GetInteger("security_vlm", "request_timeout_ms",
                              securityVlm_.requestTimeoutMs));
        securityVlm_.idleUnloadSec = static_cast<int>(
            reader.GetInteger("security_vlm", "idle_unload_sec",
                              securityVlm_.idleUnloadSec));
        securityVlm_.maxPendingEvents = static_cast<int>(
            reader.GetInteger("security_vlm", "max_pending_events",
                              securityVlm_.maxPendingEvents));
        securityVlm_.logDir =
            reader.Get("security_vlm", "log_dir", securityVlm_.logDir);
        securityVlm_.promptTemplate =
            reader.Get("security_vlm", "prompt_template", securityVlm_.promptTemplate);

        // ── [stt] section ───────────────────────────────────────────
        stt_.encoderEngineDir =
            reader.Get("stt", "encoder_engine", stt_.encoderEngineDir);
        stt_.decoderEngineDir =
            reader.Get("stt", "decoder_engine", stt_.decoderEngineDir);
        stt_.tokenizerDir =
            reader.Get("stt", "tokenizer_dir", stt_.tokenizerDir);
        stt_.shmInput =
            reader.Get("stt", "shm_input", stt_.shmInput);
        stt_.noSpeechProbThreshold = static_cast<float>(
            reader.GetReal("stt.hallucination_filter", "no_speech_prob_threshold",
                           stt_.noSpeechProbThreshold));
        stt_.minAvgLogprob = static_cast<float>(
            reader.GetReal("stt.hallucination_filter", "min_avg_logprob",
                           stt_.minAvgLogprob));
        stt_.maxConsecutiveRepeats = static_cast<int>(
            reader.GetInteger("stt.hallucination_filter", "max_consecutive_repeats",
                              stt_.maxConsecutiveRepeats));

        // ── [conversation] section ──────────────────────────────────
        conversation_.shmInputAudio =
            reader.Get("conversation", "shm_input_audio",
                       conversation_.shmInputAudio);
        conversation_.shmInputText =
            reader.Get("conversation", "shm_input_text",
                       conversation_.shmInputText);
        conversation_.defaultVariant =
            reader.Get("conversation", "default_variant",
                       conversation_.defaultVariant);

        // ── [beauty] section ────────────────────────────────────────
        beauty_.modelPath =
            reader.Get("beauty", "model_path", beauty_.modelPath);
        beauty_.shmInput =
            reader.Get("beauty", "shm_input", beauty_.shmInput);
        beauty_.shmOutput =
            reader.Get("beauty", "shm_output", beauty_.shmOutput);
        beauty_.jpegQuality = static_cast<int>(
            reader.GetInteger("beauty", "jpeg_quality", beauty_.jpegQuality));
        beauty_.enabled =
            reader.GetBoolean("beauty", "enabled", beauty_.enabled);

        // ── [orchestrator_graph] section ────────────────────────────
        orchestratorGraph_.shadowMode =
            reader.GetBoolean("orchestrator_graph", "shadow_mode",
                              orchestratorGraph_.shadowMode);
        orchestratorGraph_.drainTimeoutMs = static_cast<std::size_t>(
            reader.GetUnsigned("orchestrator_graph", "drain_timeout_ms",
                               static_cast<unsigned long>(orchestratorGraph_.drainTimeoutMs)));
        orchestratorGraph_.connectRetryCount = static_cast<std::size_t>(
            reader.GetUnsigned("orchestrator_graph", "connect_retry_count",
                               static_cast<unsigned long>(orchestratorGraph_.connectRetryCount)));
        orchestratorGraph_.connectRetryIntervalMs = static_cast<std::size_t>(
            reader.GetUnsigned("orchestrator_graph", "connect_retry_interval_ms",
                               static_cast<unsigned long>(orchestratorGraph_.connectRetryIntervalMs)));

        loaded_ = true;
        OE_LOG_INFO("ini_config_loaded: path={}, modules_configured={}, "
                   "memory_limits={}, vram_budgets={}, profiles_loaded={}",
                   path, moduleEnabled_.size(), memoryLimitsMb_.size(),
                   vramBudgetsMiB_.size(),
                   !profileConversation_.empty());
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

    /// Per-mode launch orders (simple_llm, full_conversation, …).
    [[nodiscard]] const BootModesConfig& bootModes() const noexcept {
        return bootModes_;
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
    /// [profile_conversation], [profile_sam2_segmentation],
    /// or [profile_vision_model] INI sections when present.
    [[nodiscard]] const std::unordered_map<std::string, int>&
    profilePriorities(InteractionProfile profile) const noexcept {
        switch (profile) {
        case InteractionProfile::kSam2Segmentation: return profileSam2Segmentation_;
        case InteractionProfile::kVisionModel:      return profileVisionModel_;
        case InteractionProfile::kConversation:
        default:                                    return profileConversation_;
        }
    }

    /// Graph-based pipeline orchestrator settings.
    [[nodiscard]] const OrchestratorGraphConfig& orchestratorGraph() const noexcept {
        return orchestratorGraph_;
    }

    /// Filesystem paths (models_root, …).
    [[nodiscard]] const PathsConfig& paths() const noexcept { return paths_; }

    /// Text-to-speech (Kokoro) configuration.
    [[nodiscard]] const TtsConfig& tts() const noexcept { return tts_; }

    /// Video ingest (V4L2) configuration.
    [[nodiscard]] const VideoIngestConfig& videoIngest() const noexcept {
        return videoIngest_;
    }

    /// Audio ingest (PulseAudio/ALSA/TCP) configuration.
    [[nodiscard]] const AudioIngestConfig& audioIngest() const noexcept {
        return audioIngest_;
    }

    /// Screen ingest (TCP from Windows) configuration.
    [[nodiscard]] const ScreenIngestConfig& screenIngest() const noexcept {
        return screenIngest_;
    }

    /// Audio denoise (DTLN) configuration.
    [[nodiscard]] const AudioDenoiseConfig& audioDenoise() const noexcept {
        return audioDenoise_;
    }

    /// WebSocket bridge configuration.
    [[nodiscard]] const WebsocketBridgeConfig& websocketBridge() const noexcept {
        return wsBridge_;
    }

    /// Face recognition node-level configuration.
    [[nodiscard]] const FaceRecognitionConfig& faceRecognition() const noexcept {
        return faceRecognition_;
    }

    /// Background blur node configuration.
    [[nodiscard]] const BackgroundBlurConfig& backgroundBlur() const noexcept {
        return backgroundBlur_;
    }

    /// Face filter (AR) configuration.
    [[nodiscard]] const FaceFilterConfig& faceFilter() const noexcept {
        return faceFilter_;
    }

    /// SAM2 segmentation configuration.
    [[nodiscard]] const Sam2Config& sam2() const noexcept { return sam2_; }

    /// Video denoise (BasicVSR++) configuration.
    [[nodiscard]] const VideoDenoiseConfig& videoDenoise() const noexcept {
        return videoDenoise_;
    }

    /// Security camera configuration.
    [[nodiscard]] const SecurityCameraConfig& securityCamera() const noexcept {
        return securityCamera_;
    }

    [[nodiscard]] const SecurityVlmConfig& securityVlm() const noexcept {
        return securityVlm_;
    }

    /// Beauty mode configuration.
    [[nodiscard]] const BeautyConfig& beauty() const noexcept { return beauty_; }

    /// Speech-to-text (Whisper) configuration.
    [[nodiscard]] const SttConfig& stt() const noexcept { return stt_; }

    /// Conversation (LLM) node configuration.
    [[nodiscard]] const ConversationConfig& conversation() const noexcept {
        return conversation_;
    }

    /// ZMQ PUB port for a module from [ports]. Returns fallback if the key
    /// is absent.
    [[nodiscard]] int port(std::string_view moduleName, int fallback) const {
        auto it = ports_.find(std::string(moduleName));
        return it != ports_.end() ? it->second : fallback;
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
    BootModesConfig                              bootModes_;
    HeartbeatConfig                              heartbeat_;
    PromptConfig                                 prompt_;
    SessionConfig                                session_;
    CvConfig                                     cv_;
    GenerationDefaultsConfig                     generation_;
    OrchestratorGraphConfig                      orchestratorGraph_;
    std::unordered_map<std::string, std::size_t> vramBudgetsMiB_;
    std::unordered_map<std::string, std::size_t> vramInferenceHeadroomMiB_;
    PathsConfig                                  paths_;
    TtsConfig                                    tts_;
    VideoIngestConfig                            videoIngest_;
    AudioIngestConfig                            audioIngest_;
    ScreenIngestConfig                           screenIngest_;
    AudioDenoiseConfig                           audioDenoise_;
    WebsocketBridgeConfig                        wsBridge_;
    FaceRecognitionConfig                        faceRecognition_;
    BackgroundBlurConfig                         backgroundBlur_;
    FaceFilterConfig                             faceFilter_;
    Sam2Config                                   sam2_;
    VideoDenoiseConfig                           videoDenoise_;
    SecurityCameraConfig                         securityCamera_;
    SecurityVlmConfig                            securityVlm_;
    BeautyConfig                                 beauty_;
    SttConfig                                    stt_;
    ConversationConfig                           conversation_;
    std::unordered_map<std::string, int>         ports_;

    // Split a comma-separated string into trimmed tokens; used for list-typed
    // INI values (e.g. security_camera.target_classes).
    [[nodiscard]] static std::vector<std::string> splitCsv(std::string_view s) {
        std::vector<std::string> out;
        std::size_t start = 0;
        while (start < s.size()) {
            std::size_t comma = s.find(',', start);
            std::size_t end = (comma == std::string_view::npos) ? s.size() : comma;
            std::size_t b = start;
            std::size_t e = end;
            while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
            while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
            if (e > b) out.emplace_back(s.substr(b, e - b));
            if (comma == std::string_view::npos) break;
            start = comma + 1;
        }
        return out;
    }
    // Default profile priorities — overridden by [profile_*] INI sections.
    // Priority 5 = never evict, 0 = evict first.
    std::unordered_map<std::string, int> profileConversation_{
        {"conversation_model", 5}, {"tts", 4}, {"audio_denoise", 1},
        {"background_blur", 5}, {"face_recognition", 5}, {"sam2", 2},
        {"stt", 0}, {"video_denoise", 0},
    };
    std::unordered_map<std::string, int> profileSam2Segmentation_{
        {"sam2", 5}, {"background_blur", 5}, {"face_recognition", 5},
        {"conversation_model", 0}, {"tts", 0},
        {"audio_denoise", 0}, {"stt", 0},
    };
    std::unordered_map<std::string, int> profileVisionModel_{
        {"conversation_model", 5}, {"background_blur", 0}, {"face_recognition", 5},
        {"tts", 0}, {"audio_denoise", 0}, {"sam2", 0},
        {"stt", 0},
    };
};

