#!/usr/bin/env bash
# scripts/quantize/build_trt_engines.sh — Build Whisper TRT engine for OmniEdge_AI
#
# Runs the offline conversion pipeline for the one remaining TRT-LLM consumer:
#   1. Whisper Large-v3-Turbo: INT8 SmoothQuant → TRT-LLM engine
#
# The conversation model (Gemma-4 E2B / E4B) runs via HuggingFace transformers
# — it does not use TRT-LLM and therefore has no entry here.
#
# Prerequisites:
#   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
#
# Docker alternative (recommended for clean environment):
#   docker run --rm -it --ipc host --gpus all \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v $(pwd)/models:/models \
#     nvcr.io/nvidia/tensorrt-llm/release:latest
#
# Environment variables:
#   OE_MODELS_DIR  — Root model directory (default: $HOME/omniedge_models)
#   SKIP_WHISPER   — Set to 1 to skip Whisper engine build

set -euo pipefail

# shellcheck source=../integration/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../integration/common.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OE_MODELS_DIR="${OE_MODELS_DIR:-$HOME/omniedge_models}"
ENGINE_DIR="$OE_MODELS_DIR/trt_engines"

mkdir -p "$ENGINE_DIR"

echo "=== OmniEdge_AI TensorRT Engine Build Pipeline ==="
echo "Models dir : $OE_MODELS_DIR"
echo "Engine dir : $ENGINE_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Description: Verify that trtllm-build is installed and on PATH.
# Exits with an error message if the tool is not found.
check_trtllm_build() {
    if ! PATH="$HOME/.local/bin:$PATH" command -v trtllm-build &>/dev/null; then
        echo "ERROR: trtllm-build not found."
        echo "Install: pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
        exit 1
    fi
}

# Description: Verify that trtexec is installed and on PATH.
# Exits with an error message if the tool is not found.
check_trtexec() {
    if ! command -v trtexec &>/dev/null; then
        echo "ERROR: trtexec not found. Install TensorRT >= 10.x."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. Whisper Large-v3-Turbo → INT8 TRT-LLM engine
# ---------------------------------------------------------------------------

# Description: Build the Whisper Large-v3-Turbo INT8 TensorRT-LLM engine.
# Converts HuggingFace weights to a TRT-LLM checkpoint via SmoothQuant,
# then builds encoder + decoder TRT engines. Skips each step if artefacts
# already exist.
# Globals: reads OE_MODELS_DIR, ENGINE_DIR
build_whisper() {
    echo ""
    echo "--- Whisper Large-v3-Turbo INT8 TRT-LLM engine ---"
    check_trtllm_build

    local WHISPER_DIR="$OE_MODELS_DIR/whisper-large-v3-turbo"
    local CKPT_DIR="$OE_MODELS_DIR/trt_ckpt/whisper-turbo"
    local OUT_DIR="$ENGINE_DIR/whisper-turbo"

    if [ ! -d "$WHISPER_DIR" ] || [ -z "$(ls -A "$WHISPER_DIR" 2>/dev/null)" ]; then
        echo "ERROR: Whisper weights not found at $WHISPER_DIR"
        echo "  Run: hf download openai/whisper-large-v3-turbo --local-dir $WHISPER_DIR"
        exit 1
    fi

    # Step 1: Convert to TRT-LLM checkpoint (INT8 SmoothQuant)
    if [ ! -d "$CKPT_DIR" ] || [ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
        local TRTLLM_ROOT="${TRTLLM_ROOT:-$HOME/TensorRT-LLM}"
        local CONVERT_SCRIPT="$TRTLLM_ROOT/examples/models/core/whisper/convert_checkpoint.py"
        if [ ! -f "$CONVERT_SCRIPT" ]; then
            echo "ERROR: convert_checkpoint.py not found at $CONVERT_SCRIPT"
            echo "  Set TRTLLM_ROOT to your TensorRT-LLM installation directory."
            exit 1
        fi
        echo "Converting Whisper to TRT-LLM checkpoint (INT8)..."
        python3 "$CONVERT_SCRIPT" \
            --model_dir  "$WHISPER_DIR" \
            --output_dir "$CKPT_DIR" \
            --dtype      float16 \
            --quant_algo W8A8_SQ_PER_CHANNEL \
            --tp_size    1
    else
        echo "Checkpoint already exists at $CKPT_DIR, skipping conversion."
    fi

    # Step 2: Build encoder + decoder engines
    if [ ! -d "$OUT_DIR" ] || [ -z "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "Building Whisper TRT engines..."
        PATH="$HOME/.local/bin:$PATH" trtllm-build \
            --checkpoint_dir        "$CKPT_DIR" \
            --output_dir            "$OUT_DIR" \
            --gemm_plugin           float16 \
            --bert_attention_plugin float16 \
            --max_batch_size        1 \
            --max_input_len         1500 \
            --max_seq_len           448
        echo "PASS: Whisper TRT engine → $OUT_DIR"
    else
        echo "Engine already exists at $OUT_DIR, skipping build."
    fi
}

# ---------------------------------------------------------------------------
# Main — skip Whisper build via SKIP_WHISPER=1
# ---------------------------------------------------------------------------
if [[ "${SKIP_WHISPER:-0}" != "1" ]]; then
    build_whisper
fi

echo ""
echo "=== All TRT engines built successfully ==="
echo "Engine directory: $ENGINE_DIR"
ls -lh "$ENGINE_DIR/" 2>/dev/null || true
