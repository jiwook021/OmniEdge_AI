#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// OmniEdge_AI — VRAM Compile-Time Fallback Defaults
//
// All values in MiB (mebibytes).
//
// RUNTIME CONFIGURATION:  The authoritative source for VRAM budgets, tier
// thresholds, inference headroom, and interaction profile priorities is
// config/omniedge.ini (sections [vram_budgets], [vram_tiers],
// [vram_inference_headroom], [profile_*]).  The constants below serve as
// compile-time fallbacks when an INI key is absent.
//
// GPU Mode Architecture — Four Mutually Exclusive Modes:
//   Mode 1 — Conversation (default): Qwen2.5-Omni-7B/3B or Gemma 4 E4B
//   Mode 2 — Super Resolution:       BasicVSR++ temporal video enhancement
//   Mode 3 — Image Transform:        Stable Diffusion + ControlNet
//
// Priority-based GPU allocation:
//   Priorities are dynamic — shifted by InteractionProfile when the active
//   GPU mode changes.  Eviction walks from lowest priority upward.  Within
//   the same priority level, the module with the largest VRAM budget is
//   evicted first.  Priority 5 = never evicted.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Tier selection thresholds (usable VRAM after headroom)
// ---------------------------------------------------------------------------

inline constexpr std::size_t kUltraTierThresholdMiB    = 15'500;
inline constexpr std::size_t kStandardTierThresholdMiB = 11'000;
inline constexpr std::size_t kBalancedTierThresholdMiB =  7'000;

/// VRAM kept free for OS + fragmentation headroom.
inline constexpr std::size_t kHeadroomMiB = 500;

// ---------------------------------------------------------------------------
// VRAM total budget — hard cap enforced by VramGate.
// 11 GB usable out of 12 GB physical (1 GB reserved for OS/WDDM/WSL2).
// ---------------------------------------------------------------------------

inline constexpr std::size_t kVramBudgetMiB = 11'264;  ///< 11 GB

// ---------------------------------------------------------------------------
// VRAM pressure thresholds for GpuProfiler
// ---------------------------------------------------------------------------

inline constexpr std::size_t kPressureWarningMiB  = 1500;
inline constexpr std::size_t kPressureCriticalMiB =  500;

// ---------------------------------------------------------------------------
// Per-module VRAM budgets — standard tier (12 GB RTX PRO 3000)
// ---------------------------------------------------------------------------

// ── Always-on modules (resident across all modes) ──────────────────────────
inline constexpr std::size_t kBgBlurMiB     =  250;   ///< YOLOv8-seg TRT engine (416×416 input)
inline constexpr std::size_t kFaceRecogMiB  =  500;   ///< Maximum face recog budget cap

// ── Face recognition model variant budgets ────────────────────────────────
inline constexpr std::size_t kFaceRecogScrfdAdaFace101MiB    = 350;  ///< SCRFD-10G + AdaFace IR-101
inline constexpr std::size_t kFaceRecogScrfdAdaFace50MiB     = 250;  ///< SCRFD-10G + AdaFace IR-50
inline constexpr std::size_t kFaceRecogScrfdMobileFaceNetMiB =  50;  ///< SCRFD-2.5G + MobileFaceNet

// ── Conversation mode models (mutually exclusive within mode) ──────────────
inline constexpr std::size_t kQwenOmni7bMiB = 3500;   ///< Qwen2.5-Omni-7B 4-bit GPTQ/AWQ — unified STT+LLM+TTS
                                                       ///< ~3.5 GB with on-demand weight loading + CPU offload
inline constexpr std::size_t kQwenOmni3bMiB = 2000;   ///< Qwen2.5-Omni-3B — fits comfortably at higher precision
inline constexpr std::size_t kGemmaE4bMiB   = 3000;   ///< Gemma 4 E4B — native audio+image input; no audio output
inline constexpr std::size_t kTtsMiB        =  100;   ///< TTS ONNX (Kokoro 82M, Dynamic INT8) — spawned alongside Gemma

// ── Legacy separate-module budgets (kept for backward compatibility) ───────
inline constexpr std::size_t kLlmMiB        = 3440;   ///< Standalone LLM (Qwen 2.5 7B TRT-LLM) — deprecated, use conversation models
inline constexpr std::size_t kSttMiB        = 1500;   ///< Standalone STT encoder+decoder — deprecated, Qwen-Omni handles natively

// ── Super Resolution mode ──────────────────────────────────────────────────
inline constexpr std::size_t kBasicVsrppMiB =  1500;  ///< BasicVSR++ ONNX temporal video enhancement

// ── Image Transform mode ───────────────────────────────────────────────────
inline constexpr std::size_t kStableDiffusionMiB = 4000;  ///< Stable Diffusion + ControlNet/IP-Adapter style transfer

// ── CV extension (always-on, lightweight) ──────────────────────────────────
inline constexpr std::size_t kFaceFilterMiB =  100;   ///< FaceMesh V2 ONNX + texture buffers

// ── SAM2 (on-demand, user-triggered segmentation) ─────────────────────────
inline constexpr std::size_t kSam2MiB       =  800;   ///< SAM2 Tiny ONNX image encoder + mask decoder

// ── Audio enhancement (optional toggle in Conversation mode) ───────────────
inline constexpr std::size_t kDtlnMiB       =    50;  ///< DTLN two-stage audio denoiser

// ---------------------------------------------------------------------------
// Per-module inference headroom — minimum free VRAM required before starting
// an inference pass.  Conservative estimates of peak activation memory beyond
// loaded model weights (workspace, KV cache growth, intermediate tensors).
// Used by ensureVramAvailableMiB() pre-flight checks in hot-path code.
// ---------------------------------------------------------------------------

// Always-on
inline constexpr std::size_t kBgBlurInferenceHeadroomMiB    =  64;  ///< YOLOv8-Seg TRT workspace (416×416)
inline constexpr std::size_t kFaceRecogInferenceHeadroomMiB = 128;  ///< SCRFD + AdaFace detection workspace

// Conversation models
inline constexpr std::size_t kQwenOmni7bInferenceHeadroomMiB = 512; ///< Unified STT+LLM+TTS activations + KV cache growth
inline constexpr std::size_t kQwenOmni3bInferenceHeadroomMiB = 300; ///< Smaller model, less activation memory
inline constexpr std::size_t kGemmaE4bInferenceHeadroomMiB   = 400; ///< Audio+image encoder activations + decoder workspace
inline constexpr std::size_t kTtsInferenceHeadroomMiB       =  64;  ///< ONNX Runtime intermediate tensors (Kokoro)

// Legacy separate-module headroom (kept for backward compatibility)
inline constexpr std::size_t kLlmInferenceHeadroomMiB       = 256;  ///< Standalone LLM KV cache growth + TRT workspace
inline constexpr std::size_t kSttInferenceHeadroomMiB       = 200;  ///< Standalone STT encoder+decoder activations

// Super Resolution / Image Transform
inline constexpr std::size_t kBasicVsrppInferenceHeadroomMiB= 256;  ///< Temporal-window activations
inline constexpr std::size_t kStableDiffusionInferenceHeadroomMiB = 512; ///< Diffusion U-Net + ControlNet activations

// CV extension
inline constexpr std::size_t kFaceFilterInferenceHeadroomMiB = 64; ///< FaceMesh V2 ONNX workspace (192x192 input)

// SAM2
inline constexpr std::size_t kSam2InferenceHeadroomMiB     = 256;  ///< SAM2 ViT encoder activations + mask decoder workspace

// Audio enhancement
inline constexpr std::size_t kDtlnInferenceHeadroomMiB     =  32;  ///< DTLN two-stage ONNX workspace (tiny model)

// ---------------------------------------------------------------------------
// Priority threshold — modules at or above this are NEVER evicted.
// ---------------------------------------------------------------------------

inline constexpr int kNeverEvictPriority = 5;

/// Minimum priority for auto-launching on-demand modules at boot.
/// Modules at or above this threshold in the active profile are spawned
/// during daemon initialization.
inline constexpr int kAutoLaunchPriorityThreshold = 3;

// ---------------------------------------------------------------------------
// Interaction Profiles — GPU Mode Priorities
//
// The system operates in four mutually exclusive GPU inference modes.
// Each mode has a priority profile that determines eviction order when
// VRAM is scarce.  On mode switch, the daemon applies the new profile,
// evicts all heavy processes from the previous mode, verifies VRAM is
// freed, then spawns the new mode's processes.
//
// Always-on modules (background_blur, face_recognition) have priority 5
// in ALL profiles — they are never evicted regardless of active mode.
//
// Priority scale: 0 = evict first, 5 = never evict.
// Within the same priority, largest VRAM budget is evicted first.
// ---------------------------------------------------------------------------

enum class InteractionProfile : uint8_t {
    kConversation,    ///< Default at boot: conversation model active
    kSuperResolution, ///< BasicVSR++ temporal video enhancement
    kImageTransform,  ///< Stable Diffusion + ControlNet/IP-Adapter
};

/// Convert InteractionProfile to a human-readable name.
[[nodiscard]] inline constexpr const char* profileName(InteractionProfile p) noexcept {
    switch (p) {
    case InteractionProfile::kConversation:    return "conversation";
    case InteractionProfile::kSuperResolution: return "super_resolution";
    case InteractionProfile::kImageTransform:  return "image_transform";
    }
    return "unknown";
}

// Profile priority maps and profilePriorities() have been moved to
// config/omniedge.ini ([profile_conversation], [profile_super_resolution],
// [profile_image_transform]).  IniConfig::profilePriorities() is now the
// single runtime accessor, with built-in defaults matching the former
// compile-time maps.

