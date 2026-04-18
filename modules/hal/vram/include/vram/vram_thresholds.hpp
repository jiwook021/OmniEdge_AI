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
// GPU Mode Architecture — Five Mutually Exclusive Modes (match InteractionProfile enum):
//   Mode 1 — Conversation (default): Gemma-4 E2B or E4B (native STT + vision, text out)
//   Mode 2 — SAM2 Segmentation:      On-demand interactive mask generation
//   Mode 3 — Vision Model:           On-demand multimodal vision+audio analysis
//   Mode 4 — Security Camera:        YOLOX detection + NVENC recording + on-demand VLM
//   Mode 5 — Beauty:                 Real-time face beautification (FaceMesh + CUDA ISP)
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
inline constexpr std::size_t kBgBlurMiB     =  150;   ///< MediaPipe Selfie Segmentation ONNX (256×256 input, ~462 KB fp32)
inline constexpr std::size_t kFaceRecogMiB  =  500;   ///< Maximum face recog budget cap

// ── Face recognition model variant budgets ────────────────────────────────
inline constexpr std::size_t kFaceRecogScrfdAuraFaceMiB = 350;  ///< SCRFD-10G + AuraFace v1 (glintr100)

// ── Conversation mode models (mutually exclusive within mode) ──────────────
inline constexpr std::size_t kGemmaE2bMiB   = 2500;   ///< Gemma-4 E2B (2B effective) — native audio+image input
inline constexpr std::size_t kGemmaE4bMiB   = 3000;   ///< Gemma-4 E4B (4B effective, default) — native audio+image input
inline constexpr std::size_t kTtsMiB        =  100;   ///< TTS ONNX (Kokoro 82M, Dynamic INT8) — spawned alongside Gemma

// ── Video denoise (optional toggle) ───────────────────────────────────────
inline constexpr std::size_t kBasicVsrppMiB =  1500;  ///< BasicVSR++ ONNX temporal video enhancement

// ── CV extension (always-on, lightweight) ──────────────────────────────────
inline constexpr std::size_t kFaceFilterMiB =  100;   ///< FaceMesh V2 ONNX + texture buffers

// ── SAM2 (on-demand, user-triggered segmentation) ─────────────────────────
inline constexpr std::size_t kSam2MiB       =  800;   ///< SAM2 Tiny ONNX image encoder + mask decoder

// ── Audio enhancement (optional toggle in Conversation mode) ───────────────
inline constexpr std::size_t kDtlnMiB       =    50;  ///< DTLN two-stage audio denoiser

// ── Security Camera mode (YOLOX-Nano detection + NVENC recording) ─────────
inline constexpr std::size_t kSecurityCameraMiB = 150;  ///< YOLOX-Nano ONNX (~3.9 MB) + ONNX Runtime CUDA EP arena

// ── Beauty mode (real-time face beautification) ───────────────────────────
inline constexpr std::size_t kBeautyMiB = 250;  ///< FaceMesh V2 ONNX + bilateral filter workspace + TPS warp buffers

// ---------------------------------------------------------------------------
// Per-module inference headroom — minimum free VRAM required before starting
// an inference pass.  Conservative estimates of peak activation memory beyond
// loaded model weights (workspace, KV cache growth, intermediate tensors).
// Used by ensureVramAvailableMiB() pre-flight checks in hot-path code.
// ---------------------------------------------------------------------------

// Always-on
inline constexpr std::size_t kBgBlurInferenceHeadroomMiB    =  64;  ///< MediaPipe Selfie Seg ONNX workspace (256×256, CUDA EP arena)
inline constexpr std::size_t kFaceRecogInferenceHeadroomMiB = 128;  ///< SCRFD + AuraFace detection workspace

// Conversation models
inline constexpr std::size_t kGemmaE2bInferenceHeadroomMiB   = 300; ///< Smaller variant — audio+image encoder activations
inline constexpr std::size_t kGemmaE4bInferenceHeadroomMiB   = 400; ///< Audio+image encoder activations + decoder workspace
inline constexpr std::size_t kTtsInferenceHeadroomMiB       =  64;  ///< ONNX Runtime intermediate tensors (Kokoro)

// STT (standalone binary, used by stt_node.cpp VRAM pre-flight)
inline constexpr std::size_t kSttInferenceHeadroomMiB       = 200;  ///< STT encoder+decoder activations

// Video denoise (optional toggle)
inline constexpr std::size_t kBasicVsrppInferenceHeadroomMiB= 256;  ///< Temporal-window activations

// CV extension
inline constexpr std::size_t kFaceFilterInferenceHeadroomMiB = 64; ///< FaceMesh V2 ONNX workspace (192x192 input)

// SAM2
inline constexpr std::size_t kSam2InferenceHeadroomMiB     = 256;  ///< SAM2 ViT encoder activations + mask decoder workspace

// Audio enhancement
inline constexpr std::size_t kDtlnInferenceHeadroomMiB     =  32;  ///< DTLN two-stage ONNX workspace (tiny model)

// Security Camera
inline constexpr std::size_t kSecurityCameraInferenceHeadroomMiB = 64; ///< YOLOX-Nano ONNX workspace (416×416, CUDA EP arena)

// Beauty
inline constexpr std::size_t kBeautyInferenceHeadroomMiB = 64; ///< FaceMesh V2 ONNX workspace (192×192 input) + kernel buffers

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
// The system operates in five mutually exclusive GPU inference modes.
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
    kConversation,       ///< Default at boot: Gemma-4 conversation model active
    kSam2Segmentation,   ///< SAM2 interactive segmentation
    kVisionModel,        ///< Multimodal VLM — on-demand vision+audio analysis
    kSecurity,           ///< Security camera: YOLO detection + NVENC recording + on-demand VLM
    kBeauty,             ///< Real-time face beautification: FaceMesh V2 + bilateral filter + TPS warp
};

/// Convert InteractionProfile to a human-readable name.
[[nodiscard]] inline constexpr const char* profileName(InteractionProfile p) noexcept {
    switch (p) {
    case InteractionProfile::kConversation:      return "conversation";
    case InteractionProfile::kSam2Segmentation:  return "sam2_segmentation";
    case InteractionProfile::kVisionModel:       return "vision_model";
    case InteractionProfile::kSecurity:          return "security";
    case InteractionProfile::kBeauty:            return "beauty";
    }
    return "unknown";
}

// Profile priority maps live in config/omniedge.ini as [profile_conversation],
// [profile_vision_model], [profile_sam2_segmentation], [profile_security],
// [profile_beauty] sections. IniConfig::profilePriorities() is the single
// runtime accessor.

