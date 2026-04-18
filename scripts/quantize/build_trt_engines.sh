#!/usr/bin/env bash
# scripts/quantize/build_trt_engines.sh — Build all TensorRT engines for OmniEdge_AI
#
# Runs the full offline conversion pipeline:
#   1. Qwen 2.5 7B: TRT-LLM INT4-AWQ checkpoint → TRT engine
#   2. Whisper Large-v3-Turbo: INT8 SmoothQuant → TRT-LLM engine
#
# Prerequisites:
#   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
#   pip install auto-awq autoawq-kernels
#   pip install optimum[onnxruntime-gpu]
#
# Docker alternative (recommended for clean environment):
#   docker run --rm -it --ipc host --gpus all \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v $(pwd)/models:/models \
#     nvcr.io/nvidia/tensorrt-llm/release:latest
#
# Environment variables:
#   OE_MODELS_DIR  — Root model directory (default: $HOME/omniedge_models)
#   SKIP_QWEN      — Set to 1 to skip Qwen engine build
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
# 1. Qwen 2.5 7B → TRT-LLM INT4-AWQ engine
# ---------------------------------------------------------------------------

# Description: Build the Qwen 2.5 7B INT4-AWQ TensorRT-LLM engine.
# Converts HuggingFace AWQ weights to a TRT-LLM checkpoint, then builds
# the TRT engine. Skips each step if artefacts already exist.
# Globals: reads OE_MODELS_DIR, ENGINE_DIR, SCRIPT_DIR
build_qwen() {
    echo "--- [1/3] Qwen 2.5 7B INT4-AWQ TRT-LLM engine ---"
    check_trtllm_build

    local QWEN_AWQ="$OE_MODELS_DIR/qwen2.5-7b-instruct-awq"
    local CKPT_DIR="$OE_MODELS_DIR/trt_ckpt/qwen2.5-7b-awq"
    local OUT_DIR="$ENGINE_DIR/qwen2.5-7b-awq"

    if [ ! -d "$QWEN_AWQ" ] || [ -z "$(ls -A "$QWEN_AWQ" 2>/dev/null)" ]; then
        echo "ERROR: AWQ weights not found at $QWEN_AWQ"
        echo "  Run: python3 $SCRIPT_DIR/quantize_qwen_awq.py"
        echo "  Or:  hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir $QWEN_AWQ"
        exit 1
    fi

    # Step 1: Convert to TRT-LLM checkpoint via Python API
    # Validate by checking for config.json (dir existing alone is not sufficient)
    if [ ! -f "$CKPT_DIR/config.json" ]; then
        echo "Converting to TRT-LLM checkpoint (QWenForCausalLM Python API)..."
        rm -rf "$CKPT_DIR"
        python3 - <<PYEOF
import sys
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.quantization import QuantConfig, QuantAlgo

quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
model = QWenForCausalLM.from_hugging_face(
    "$QWEN_AWQ",
    dtype="float16",
    quant_config=quant_config,
    use_autoawq=True,
)
model.save_checkpoint("$CKPT_DIR")
print("Checkpoint saved to $CKPT_DIR")
PYEOF
        if [ $? -ne 0 ]; then
            echo "ERROR: Checkpoint conversion failed. See errors above."
            exit 1
        fi
    else
        echo "Checkpoint already valid at $CKPT_DIR, skipping conversion."
    fi

    # Step 2: Build TRT engine
    if [ ! -d "$OUT_DIR" ] || [ -z "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "Building TRT engine..."
        PATH="$HOME/.local/bin:$PATH" trtllm-build \
            --checkpoint_dir          "$CKPT_DIR" \
            --output_dir              "$OUT_DIR" \
            --gemm_plugin             float16 \
            --gpt_attention_plugin    float16 \
            --max_batch_size          1 \
            --max_input_len           2048 \
            --max_seq_len             4096 \
            --kv_cache_type           paged \
            --use_paged_context_fmha  enable \
            --max_num_tokens          4096
        echo "PASS: Qwen TRT engine → $OUT_DIR"
    else
        echo "Engine already exists at $OUT_DIR, skipping build."
    fi
}

# ---------------------------------------------------------------------------
# 2. Whisper Large-v3-Turbo → INT8 TRT-LLM engine
# ---------------------------------------------------------------------------

# Description: Build the Whisper Large-v3-Turbo INT8 TensorRT-LLM engine.
# Converts HuggingFace weights to a TRT-LLM checkpoint via SmoothQuant,
# then builds encoder + decoder TRT engines. Skips each step if artefacts
# already exist.
# Globals: reads OE_MODELS_DIR, ENGINE_DIR
build_whisper() {
    echo ""
    echo "--- [2/3] Whisper Large-v3-Turbo INT8 TRT-LLM engine ---"
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
# Main — skip individual models via SKIP_QWEN=1, SKIP_WHISPER=1, etc.
# ---------------------------------------------------------------------------
if [[ "${SKIP_QWEN:-0}" != "1" ]]; then
    build_qwen
fi

if [[ "${SKIP_WHISPER:-0}" != "1" ]]; then
    build_whisper
fi

echo ""
echo "=== All TRT engines built successfully ==="
echo "Engine directory: $ENGINE_DIR"
ls -lh "$ENGINE_DIR/" 2>/dev/null || true
