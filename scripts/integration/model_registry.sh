#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Shared Model Registry
#
# Centralises model path definitions used by runtime verification and
# model tests. Source this file and call declare_model_registry() to
# populate the REQUIRED_MODELS and PREFLIGHT_MODELS arrays.
#
# Usage:
#   source scripts/integration/model_registry.sh
#   declare_model_registry        # populates arrays based on GPU tier
#   check_model_list REQUIRED_MODELS
#
# NOTE: This file is a library meant to be sourced, NOT executed directly.
#       Requires common.sh to be sourced first (for resolve_gpu_tier, OE_MODELS_DIR).
# =============================================================================

# Guard against double-sourcing
[[ -n "${_OE_MODEL_REGISTRY_SOURCED:-}" ]] && return 0
readonly _OE_MODEL_REGISTRY_SOURCED=1

# Description: Populate REQUIRED_MODELS and PREFLIGHT_MODELS arrays
# Args: none (reads OE_MODELS_DIR, OE_ENGINES_DIR from environment)
# Sets globals:
#   REQUIRED_MODELS  — full list for system launch verification
#   PREFLIGHT_MODELS — subset for headless model pipeline tests
#   IS_BLACKWELL     — 1 if GPU is Blackwell generation, 0 otherwise
#
# Each entry is "Label:path" — compatible with check_model_list().
declare_model_registry() {
    local gpu_tier
    gpu_tier=$(resolve_gpu_tier)
    IS_BLACKWELL=0
    [[ "$gpu_tier" == "blackwell" ]] && IS_BLACKWELL=1

    # -- Conversation model (Gemma-4 via HF transformers) -------------------
    # Gemma-4 runs as a Python subprocess using HuggingFace transformers —
    # no TRT-LLM engine build. Default variant is E4B; E2B is the fallback.
    local conv_model_label="Conversation model (Gemma-4 E4B)"
    local conv_model_dir="$OE_MODELS_DIR/gemma-4-e4b"

    # -- Full model list for system launch ----------------------------------
    REQUIRED_MODELS=(
        "${conv_model_label}:${conv_model_dir}"
        "BgBlur (MediaPipe Selfie Seg):$OE_MODELS_DIR/bg_blur/mediapipe_selfie_seg.onnx"
        "VAD (Silero):$OE_MODELS_DIR/silero_vad.onnx"
        "TTS ONNX (Kokoro):$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
        "TTS voices:$OE_MODELS_DIR/kokoro/voices"
        "BasicVSR++ (SR):$OE_MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx"
        "STT engines:$OE_ENGINES_DIR/whisper-turbo"
        "STT tokenizer:$OE_MODELS_DIR/whisper-large-v3-turbo/vocab.json"
        "FaceRecog:$OE_MODELS_DIR/face_models/scrfd_auraface"
    )

    # -- Preflight subset for headless model pipeline tests -----------------
    PREFLIGHT_MODELS=(
        "${conv_model_label}:${conv_model_dir}"
        "Kokoro TTS:$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
        "BasicVSR++ SR:$OE_MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx"
        "MediaPipe Selfie Seg BgBlur:$OE_MODELS_DIR/bg_blur/mediapipe_selfie_seg.onnx"
        "Silero VAD:$OE_MODELS_DIR/silero_vad.onnx"
        "Whisper STT:$OE_ENGINES_DIR/whisper-turbo/encoder"
    )
}

# Description: Get the Gemma-4 conversation model directory for test discovery
# Args: none
# Returns: prints the model directory path
#
# Searches for E4B then E2B variant directories.
get_conversation_model_dir() {
    for name in gemma-4-e4b gemma-4-e2b; do
        if [[ -d "$OE_MODELS_DIR/$name" ]]; then
            echo "$OE_MODELS_DIR/$name"
            return 0
        fi
    done
    return 1
}

# Description: Get the Kokoro ONNX model path for test discovery
# Args: none
# Returns: prints the ONNX model file path
get_kokoro_onnx_model() {
    for name in model_int8.onnx kokoro-v1_0.onnx model.onnx; do
        if [[ -f "$OE_MODELS_DIR/kokoro-onnx/onnx/$name" ]]; then
            echo "$OE_MODELS_DIR/kokoro-onnx/onnx/$name"
            return 0
        fi
    done
    return 1
}

# Description: Get the Kokoro voice directory for test discovery
# Args: none
# Returns: prints the voice directory path (prefers .npy files)
get_kokoro_voice_dir() {
    local voice_dir="$OE_MODELS_DIR/kokoro/voices"

    # Check primary location for .npy files
    if [[ -d "$voice_dir" ]] && ls "$voice_dir"/*.npy >/dev/null 2>&1; then
        echo "$voice_dir"
        return 0
    fi

    # Fallback: kokoro-npy/voices/ has .npy files
    local alt_dir="$OE_MODELS_DIR/kokoro-npy/voices"
    if [[ -d "$alt_dir" ]] && ls "$alt_dir"/*.npy >/dev/null 2>&1; then
        echo "$alt_dir"
        return 0
    fi

    # Last resort: return primary even without .npy (may have .bin/.pt)
    if [[ -d "$voice_dir" ]]; then
        echo "$voice_dir"
        return 0
    fi

    return 1
}
