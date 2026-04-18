#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Shared Model Registry
#
# Centralises model path definitions used by run_all.sh, verify.sh, and
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

    # -- LLM engine and tokenizer (GPU-tier dependent) ----------------------
    local llm_engine_label llm_engine_path llm_tokenizer_path

    if [[ "$IS_BLACKWELL" -eq 1 ]]; then
        llm_engine_label="LLM engine (Qwen2.5-7B nvfp4)"
        llm_engine_path="$OE_ENGINES_DIR/qwen2.5-7b-nvfp4/rank0.engine"
        llm_tokenizer_path="$OE_MODELS_DIR/qwen2.5-7b-instruct/tokenizer.json"
    else
        llm_engine_label="LLM engine (Qwen2.5-7B AWQ)"
        llm_engine_path="$OE_ENGINES_DIR/qwen2.5-7b-awq/rank0.engine"
        llm_tokenizer_path="$OE_MODELS_DIR/qwen2.5-7b-instruct-awq/tokenizer.json"
    fi

    # -- Full model list for system launch ----------------------------------
    REQUIRED_MODELS=(
        "${llm_engine_label}:${llm_engine_path}"
        "LLM tokenizer:${llm_tokenizer_path}"
        "BgBlur (YOLO):$OE_ENGINES_DIR/yolov8n-seg.engine"
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
        "${llm_engine_label}:${llm_engine_path}"
        "Kokoro TTS:$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
        "BasicVSR++ SR:$OE_MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx"
        "YOLOv8 BgBlur:$OE_ENGINES_DIR/yolov8n-seg.engine"
        "Silero VAD:$OE_MODELS_DIR/silero_vad.onnx"
        "Whisper STT:$OE_ENGINES_DIR/whisper-turbo/encoder"
    )
}

# Description: Get the LLM engine directory for test discovery
# Args: none (uses REQUIRED_MODELS)
# Returns: prints the engine directory path (without /rank0.engine suffix)
#
# Searches for nvfp4 then AWQ engine directories.
get_llm_engine_dir() {
    for name in qwen2.5-7b-nvfp4 qwen2.5-7b-awq; do
        if [[ -d "$OE_ENGINES_DIR/$name" ]]; then
            echo "$OE_ENGINES_DIR/$name"
            return 0
        fi
    done
    return 1
}

# Description: Get the LLM tokenizer directory for test discovery
# Args: none
# Returns: prints the tokenizer directory path
#
# Searches for instruct then instruct-awq tokenizer directories.
get_llm_tokenizer_dir() {
    for name in qwen2.5-7b-instruct qwen2.5-7b-instruct-awq; do
        if [[ -d "$OE_MODELS_DIR/$name" ]] && [[ -f "$OE_MODELS_DIR/$name/tokenizer.json" ]]; then
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
