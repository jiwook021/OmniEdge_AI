#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Install Cross-Model Benchmark Variants
#
# Downloads alternative model architectures and sizes for cross-model
# benchmarking with scripts/benchmark/benchmark_models.py --quant-sweep.
#
# Usage:
#   bash scripts/install_cross_models.sh              # Install all
#   bash scripts/install_cross_models.sh --llm        # LLM variants only
#   bash scripts/install_cross_models.sh --stt        # STT variants only
#   bash scripts/install_cross_models.sh --tts        # TTS variants only
#   bash scripts/install_cross_models.sh --cv           # CV variants only
#   bash scripts/install_cross_models.sh --conversation # Conversation variants only
#   bash scripts/install_cross_models.sh --no-engines   # Download only, skip TRT builds
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source shared utilities (colours, helpers, download_hf, etc.)
source "$SCRIPT_DIR/common.sh"

# ── Flags ─────────────────────────────────────────────────────────────────────
INSTALL_LLM=0
INSTALL_STT=0
INSTALL_TTS=0
INSTALL_CV=0
INSTALL_CONVERSATION=0
BUILD_ENGINES=1
INSTALL_ALL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)        INSTALL_LLM=1; INSTALL_ALL=0 ;;
        --stt)        INSTALL_STT=1; INSTALL_ALL=0 ;;
        --tts)        INSTALL_TTS=1; INSTALL_ALL=0 ;;
        --cv)           INSTALL_CV=1;           INSTALL_ALL=0 ;;
        --conversation) INSTALL_CONVERSATION=1; INSTALL_ALL=0 ;;
        --no-engines) BUILD_ENGINES=0 ;;
        --help|-h)
            echo "Usage: bash scripts/install_cross_models.sh [--llm] [--stt] [--tts] [--cv] [--conversation] [--no-engines]"
            exit 0 ;;
        *) echo -e "${RED}Unknown flag: $1${NC}"; exit 1 ;;
    esac
    shift
done

if [[ "$INSTALL_ALL" -eq 1 ]]; then
    INSTALL_LLM=1; INSTALL_STT=1; INSTALL_TTS=1; INSTALL_CV=1; INSTALL_CONVERSATION=1
fi

mkdir -p "$OE_MODELS_DIR" "$OE_ENGINES_DIR"
ERRORS=0

# Pre-flight checks
if ! command -v huggingface-cli &>/dev/null && ! command -v hf &>/dev/null; then
    echo -e "${RED}ERROR:${NC} huggingface-cli not found. Install with: pip install huggingface-hub[cli]" >&2
    exit 1
fi

if [[ "$BUILD_ENGINES" -eq 1 ]]; then
    if ! command -v trtllm-build &>/dev/null; then
        echo -e "${YELLOW}WARNING:${NC} trtllm-build not found in PATH."
        echo -e "         TRT-LLM engine builds will be skipped."
        echo -e "         Install TensorRT-LLM: pip install tensorrt-llm"
        BUILD_ENGINES=0
    fi
fi

# download_hf is provided by common.sh

# Description: Build a TRT-LLM engine for a Qwen AWQ model
# Args: $1 - checkpoint dir, $2 - engine output dir, $3 - HF weights dir, $4 - description
# Returns: 0 on success, 1 on build failure (also increments ERRORS)
build_qwen_engine() {
    local CKPT_DIR="$1"
    local ENGINE_DIR="$2"
    local HF_DIR="$3"
    local DESC="$4"

    if [[ "$BUILD_ENGINES" -eq 0 ]]; then
        echo -e "  ${YELLOW}SKIP${NC}    Engine build for $DESC (--no-engines)"
        return 0
    fi

    if dir_exists "$ENGINE_DIR"; then
        echo -e "  ${GREEN}EXISTS${NC}  Engine: $DESC"
        return 0
    fi

    if ! dir_exists "$HF_DIR"; then
        echo -e "  ${YELLOW}SKIP${NC}    Engine build for $DESC (weights not downloaded)"
        return 0
    fi

    echo -e "  ${CYAN}BUILD${NC}   TRT-LLM engine: $DESC"

    # Step 1: Convert checkpoint (use_autoawq=True required for AWQ weights)
    if ! dir_exists "$CKPT_DIR"; then
        mkdir -p "$CKPT_DIR"
        python3 "$PROJECT_ROOT/scripts/helpers/convert_trtllm_checkpoint.py" \
            --model-class QWenForCausalLM \
            --hf-dir "$HF_DIR" \
            --output-dir "$CKPT_DIR" \
            --quant-algo W4A16_AWQ \
            --use-autoawq \
        || { record_error "Checkpoint conversion: $DESC"; return 1; }
    fi

    # Step 2: Build engine
    mkdir -p "$ENGINE_DIR"
    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 4096 \
        --kv_cache_type paged \
        --use_paged_context_fmha enable \
        --max_num_tokens 4096 \
    || { record_error "Engine build: $DESC"; return 1; }

    echo -e "  ${GREEN}OK${NC}      Engine: $DESC → $ENGINE_DIR"
}

# Description: Build a TRT-LLM engine for non-Qwen models (Llama, Phi, etc.)
# Args: $1 - TRT-LLM model class, $2 - checkpoint dir, $3 - engine dir,
#        $4 - HF weights dir, $5 - description
# Returns: 0 on success; increments ERRORS on failure
build_generic_engine() {
    local MODEL_CLASS="$1"
    local CKPT_DIR="$2"
    local ENGINE_DIR="$3"
    local HF_DIR="$4"
    local DESC="$5"

    if [[ "$BUILD_ENGINES" -eq 0 ]]; then
        echo -e "  ${YELLOW}SKIP${NC}    Engine build for $DESC (--no-engines)"
        return 0
    fi

    if dir_exists "$ENGINE_DIR"; then
        echo -e "  ${GREEN}EXISTS${NC}  Engine: $DESC"
        return 0
    fi

    if ! dir_exists "$HF_DIR"; then
        echo -e "  ${YELLOW}SKIP${NC}    Engine build for $DESC (weights not downloaded)"
        return 0
    fi

    echo -e "  ${CYAN}BUILD${NC}   TRT-LLM engine: $DESC"

    mkdir -p "$CKPT_DIR" "$ENGINE_DIR"
    python3 "$PROJECT_ROOT/scripts/helpers/convert_trtllm_checkpoint.py" \
        --model-class "$MODEL_CLASS" \
        --hf-dir "$HF_DIR" \
        --output-dir "$CKPT_DIR" \
        --quant-algo W4A16_AWQ \
        --use-autoawq \
    && trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --gemm_plugin float16 \
        --gpt_attention_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 4096 \
        --kv_cache_type paged \
        --use_paged_context_fmha enable \
        --max_num_tokens 4096 \
    && echo -e "  ${GREEN}OK${NC}      Engine: $DESC → $ENGINE_DIR" \
    || record_error "Engine: $DESC"
}

echo -e "${BOLD}OmniEdge_AI — Cross-Model Variant Installer${NC}"
echo -e "  Models dir:  $OE_MODELS_DIR"
echo -e "  Engines dir: $OE_ENGINES_DIR"
echo ""

# =============================================================================
#  LLM VARIANTS
# =============================================================================
if [[ "$INSTALL_LLM" -eq 1 ]]; then
    echo -e "${BOLD}=== LLM Variants ===${NC}"
    echo ""

    # ── Qwen 2.5 3B AWQ ──────────────────────────────────────────────────
    download_hf "Qwen/Qwen2.5-3B-Instruct-AWQ" \
        "$OE_MODELS_DIR/qwen2.5-3b-instruct-awq" \
        "Qwen 2.5 3B AWQ (~2 GB)" \
    || record_error "Download Qwen 2.5 3B AWQ"

    build_qwen_engine \
        "$OE_MODELS_DIR/trt_ckpt/qwen2.5-3b-awq" \
        "$OE_ENGINES_DIR/qwen2.5-3b-awq" \
        "$OE_MODELS_DIR/qwen2.5-3b-instruct-awq" \
        "Qwen 2.5 3B AWQ"

    # ── Qwen 2.5 1.5B AWQ ────────────────────────────────────────────────
    download_hf "Qwen/Qwen2.5-1.5B-Instruct-AWQ" \
        "$OE_MODELS_DIR/qwen2.5-1.5b-instruct-awq" \
        "Qwen 2.5 1.5B AWQ (~1.2 GB)" \
    || record_error "Download Qwen 2.5 1.5B AWQ"

    build_qwen_engine \
        "$OE_MODELS_DIR/trt_ckpt/qwen2.5-1.5b-awq" \
        "$OE_ENGINES_DIR/qwen2.5-1.5b-awq" \
        "$OE_MODELS_DIR/qwen2.5-1.5b-instruct-awq" \
        "Qwen 2.5 1.5B AWQ"

    # ── Llama 3.1 8B AWQ ─────────────────────────────────────────────────
    download_hf "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
        "$OE_MODELS_DIR/llama-3.1-8b-instruct-awq" \
        "Llama 3.1 8B AWQ (~4.5 GB)" \
    || record_error "Download Llama 3.1 8B AWQ"

    build_generic_engine "LLaMAForCausalLM" \
        "$OE_MODELS_DIR/trt_ckpt/llama-3.1-8b-awq" \
        "$OE_ENGINES_DIR/llama-3.1-8b-awq" \
        "$OE_MODELS_DIR/llama-3.1-8b-instruct-awq" \
        "Llama 3.1 8B AWQ"

    # ── Phi-3.5 Mini AWQ ─────────────────────────────────────────────────
    download_hf "thesven/Phi-3.5-mini-instruct-awq" \
        "$OE_MODELS_DIR/phi-3.5-mini-instruct-awq" \
        "Phi-3.5 Mini AWQ (~2.2 GB)" \
    || record_error "Download Phi-3.5 Mini AWQ"

    build_generic_engine "Phi3ForCausalLM" \
        "$OE_MODELS_DIR/trt_ckpt/phi-3.5-mini-awq" \
        "$OE_ENGINES_DIR/phi-3.5-mini-awq" \
        "$OE_MODELS_DIR/phi-3.5-mini-instruct-awq" \
        "Phi-3.5 Mini AWQ"

    echo ""
fi

# =============================================================================
#  STT VARIANTS
# =============================================================================
if [[ "$INSTALL_STT" -eq 1 ]]; then
    echo -e "${BOLD}=== STT Variants ===${NC}"
    echo ""

    download_hf "openai/whisper-medium" \
        "$OE_MODELS_DIR/whisper-medium" \
        "Whisper Medium 769M (~1.5 GB)" \
    || record_error "Download Whisper Medium"

    download_hf "openai/whisper-small" \
        "$OE_MODELS_DIR/whisper-small" \
        "Whisper Small 244M (~500 MB)" \
    || record_error "Download Whisper Small"

    download_hf "distil-whisper/distil-large-v3" \
        "$OE_MODELS_DIR/distil-whisper-large-v3" \
        "Distil-Whisper Large V3 (~1.5 GB)" \
    || record_error "Download Distil-Whisper Large V3"

    # Canary 1B requires NeMo toolkit
    CANARY_DIR="$OE_MODELS_DIR/canary-1b"
    if dir_exists "$CANARY_DIR"; then
        echo -e "  ${GREEN}EXISTS${NC}  Canary 1B → $CANARY_DIR"
    else
        echo -e "  ${CYAN}DOWNLOAD${NC}  NVIDIA Canary 1B (requires nemo_toolkit)"
        mkdir -p "$CANARY_DIR"
        python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/canary-1b')
model.save_to('$CANARY_DIR/canary-1b.nemo')
print('Canary 1B saved')
" 2>/dev/null \
        && echo -e "  ${GREEN}OK${NC}      Canary 1B" \
        || echo -e "  ${YELLOW}SKIP${NC}    Canary 1B — install nemo_toolkit: pip install nemo_toolkit[asr]"
    fi

    echo ""
fi

# =============================================================================
#  TTS VARIANTS
# =============================================================================
if [[ "$INSTALL_TTS" -eq 1 ]]; then
    echo -e "${BOLD}=== TTS Variants ===${NC}"
    echo ""

    # ── Piper TTS ─────────────────────────────────────────────────────────
    PIPER_DIR="$OE_MODELS_DIR/piper"
    PIPER_ONNX="$PIPER_DIR/en_US-lessac-medium.onnx"
    if [[ -f "$PIPER_ONNX" ]]; then
        echo -e "  ${GREEN}EXISTS${NC}  Piper TTS → $PIPER_ONNX"
    else
        echo -e "  ${CYAN}DOWNLOAD${NC}  Piper TTS en_US-lessac-medium (~80 MB)"
        mkdir -p "$PIPER_DIR"
        curl -fsSL -L -o "$PIPER_ONNX" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        && curl -fsSL -L -o "${PIPER_ONNX}.json" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        && echo -e "  ${GREEN}OK${NC}      Piper TTS" \
        || record_error "Piper TTS download"
    fi

    # ── VITS2 ─────────────────────────────────────────────────────────────
    VITS2_DIR="$OE_MODELS_DIR/vits2"
    VITS2_ONNX="$VITS2_DIR/model.onnx"
    if [[ -f "$VITS2_ONNX" ]]; then
        echo -e "  ${GREEN}EXISTS${NC}  VITS2 ONNX → $VITS2_ONNX"
    else
        echo -e "  ${YELLOW}SKIP${NC}    VITS2 — no standard pre-exported ONNX available"
        echo -e "         To add: export a VITS2 checkpoint to ONNX and place at $VITS2_ONNX"
    fi

    echo ""
fi

# =============================================================================
#  CV VARIANTS
# =============================================================================
if [[ "$INSTALL_CV" -eq 1 ]]; then
    echo -e "${BOLD}=== CV Variants ===${NC}"
    echo ""

    YOLO_S_ENGINE="$OE_ENGINES_DIR/yolov8s-seg.engine"
    if [[ -f "$YOLO_S_ENGINE" ]]; then
        echo -e "  ${GREEN}EXISTS${NC}  YOLOv8s-seg TRT → $YOLO_S_ENGINE"
    else
        echo -e "  ${CYAN}BUILD${NC}   YOLOv8s-seg (export ONNX + build TRT engine)"
        YOLO_TMP="/tmp/yolov8s-seg-export"
        [[ -n "$YOLO_TMP" && "$YOLO_TMP" != "/" ]] && rm -rf "$YOLO_TMP"
        mkdir -p "$YOLO_TMP"

        # Export ONNX — ultralytics puts output next to the .pt file
        python3 -c "
import os, shutil
os.chdir('$YOLO_TMP')
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
path = model.export(format='onnx', imgsz=640)
print(f'ONNX exported to: {path}')
" 2>&1 | tail -5

        # Find the exported ONNX (could be in $YOLO_TMP or next to the downloaded .pt)
        YOLO_ONNX=""
        for candidate in "$YOLO_TMP/yolov8s-seg.onnx" "$YOLO_TMP/runs/"*"/yolov8s-seg.onnx"; do
            [[ -f "$candidate" ]] && YOLO_ONNX="$candidate" && break
        done

        if [[ -n "$YOLO_ONNX" ]] && [[ -f "$YOLO_ONNX" ]]; then
            echo -e "  ${CYAN}BUILD${NC}   trtexec FP16 engine from $YOLO_ONNX"
            trtexec --onnx="$YOLO_ONNX" --fp16 \
                --saveEngine="$YOLO_S_ENGINE" 2>&1 | tail -5 \
            && echo -e "  ${GREEN}OK${NC}      YOLOv8s-seg TRT → $YOLO_S_ENGINE" \
            || record_error "YOLOv8s-seg TRT engine build"
        else
            echo -e "  ${YELLOW}SKIP${NC}    YOLOv8s-seg — ONNX export failed (pip install --break-system-packages ultralytics)"
        fi

        rm -rf "$YOLO_TMP"
    fi

    # ── SAM2 Hiera Tiny ONNX (interactive segmentation) ──────────────────
    SAM2_DIR="$OE_MODELS_DIR/sam2"
    SAM2_ENCODER="$SAM2_DIR/sam2_hiera_tiny_encoder.onnx"
    SAM2_DECODER="$SAM2_DIR/sam2_hiera_tiny_decoder.onnx"
    if [[ -f "$SAM2_ENCODER" ]] && [[ -f "$SAM2_DECODER" ]]; then
        echo -e "  ${GREEN}EXISTS${NC}  SAM2 Hiera Tiny ONNX → $SAM2_DIR"
    else
        echo -e "  ${CYAN}DOWNLOAD${NC}  SAM2 Hiera Tiny ONNX (~700 MB)"
        mkdir -p "$SAM2_DIR"
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download facebook/sam2-hiera-tiny \
                --include "*.onnx" \
                --local-dir "$SAM2_DIR" \
            && echo -e "  ${GREEN}OK${NC}      SAM2 Hiera Tiny ONNX" \
            || record_error "SAM2 Hiera Tiny download"
        else
            curl -fsSL -L "https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny_encoder.onnx" \
                -o "$SAM2_ENCODER" \
            && curl -fsSL -L "https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny_decoder.onnx" \
                -o "$SAM2_DECODER" \
            && echo -e "  ${GREEN}OK${NC}      SAM2 Hiera Tiny ONNX" \
            || record_error "SAM2 Hiera Tiny download"
        fi
    fi

    echo ""
fi

# =============================================================================
#  CONVERSATION VARIANTS (VLMs)
# =============================================================================
if [[ "$INSTALL_CONVERSATION" -eq 1 ]]; then
    echo -e "${BOLD}=== Conversation Variants ===${NC}"
    echo ""

    # ── Llama 3.2 11B Vision AWQ ─────────────────────────────────────────
    # HF weights (tokenizer + AWQ checkpoint)
    download_hf "hugging-quants/Meta-Llama-3.2-11B-Vision-Instruct-AWQ-INT4" \
        "$OE_MODELS_DIR/llama-3.2-11b-vision-instruct-awq" \
        "Llama 3.2 11B Vision AWQ (~6.5 GB)" \
    || record_error "Download Llama 3.2 11B Vision AWQ"

    # TRT-LLM engine build (MllamaForConditionalGeneration)
    build_generic_engine "MllamaForConditionalGeneration" \
        "$OE_MODELS_DIR/trt_ckpt/llama-3.2-11b-vision-awq" \
        "$OE_ENGINES_DIR/llama-3.2-11b-vision-awq" \
        "$OE_MODELS_DIR/llama-3.2-11b-vision-instruct-awq" \
        "Llama 3.2 11B Vision AWQ"

    echo ""
fi

# =============================================================================
#  SUMMARY
# =============================================================================
echo -e "${BOLD}=== Installed Model Variants ===${NC}"
echo ""

# Description: Print a check/cross status line for a model variant
# Args: $1 - human-readable name, $2 - path to check (file or dir)
check_variant() {
    local NAME="$1"
    local PATH_CHK="$2"
    if [[ -e "$PATH_CHK" ]]; then
        echo -e "  ${GREEN}✓${NC}  $NAME"
    else
        echo -e "  ${YELLOW}✗${NC}  $NAME  (not found: $PATH_CHK)"
    fi
}

echo -e "  ${BOLD}LLM:${NC}"
check_variant "Qwen 2.5 7B NVFP4"               "$OE_ENGINES_DIR/qwen2.5-7b-nvfp4"
check_variant "Qwen 2.5 7B AWQ"                  "$OE_ENGINES_DIR/qwen2.5-7b-awq"
check_variant "Qwen 2.5 7B FP16 (HF)"           "$OE_MODELS_DIR/qwen2.5-7b-instruct"
check_variant "Qwen 2.5 3B AWQ"                  "$OE_ENGINES_DIR/qwen2.5-3b-awq"
check_variant "Qwen 2.5 3B HF"                   "$OE_MODELS_DIR/qwen2.5-3b-instruct-awq"
check_variant "Llama 3.1 8B AWQ"                 "$OE_ENGINES_DIR/llama-3.1-8b-awq"
check_variant "Llama 3.1 8B HF"                  "$OE_MODELS_DIR/llama-3.1-8b-instruct-awq"
check_variant "Qwen 2.5 1.5B AWQ (extra)"       "$OE_ENGINES_DIR/qwen2.5-1.5b-awq"
check_variant "Phi-3.5 Mini AWQ (extra)"         "$OE_ENGINES_DIR/phi-3.5-mini-awq"

echo -e "  ${BOLD}Conversation:${NC}"
check_variant "Qwen2.5-Omni 7B NVFP4"           "$OE_ENGINES_DIR/qwen2.5-7b-nvfp4"
check_variant "Qwen2.5-Omni 3B AWQ"             "$OE_ENGINES_DIR/qwen2.5-3b-awq"
check_variant "Gemma 4 E4B (HF)"                 "$OE_MODELS_DIR/gemma-4-e4b"
check_variant "Llama 3.2 11B Vision AWQ"         "$OE_ENGINES_DIR/llama-3.2-11b-vision-awq"
check_variant "Llama 3.2 11B Vision HF"          "$OE_MODELS_DIR/llama-3.2-11b-vision-instruct-awq"

echo -e "  ${BOLD}TTS:${NC}"
check_variant "Kokoro INT8"                       "$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
check_variant "Kokoro FP32"                       "$OE_MODELS_DIR/kokoro-onnx/onnx/model.onnx"
check_variant "VITS2 ONNX"                        "$OE_MODELS_DIR/vits2/model.onnx"
check_variant "Piper TTS"                         "$OE_MODELS_DIR/piper/en_US-lessac-medium.onnx"

echo -e "  ${BOLD}VAD:${NC}"
check_variant "Silero VAD"                        "$OE_MODELS_DIR/silero_vad.onnx"

echo -e "  ${BOLD}STT (extra):${NC}"
check_variant "Whisper V3 Turbo INT8"            "$OE_ENGINES_DIR/whisper-turbo"
check_variant "Whisper Medium"                    "$OE_MODELS_DIR/whisper-medium"
check_variant "Whisper Small"                     "$OE_MODELS_DIR/whisper-small"
check_variant "Distil-Whisper Large V3"          "$OE_MODELS_DIR/distil-whisper-large-v3"
check_variant "Canary 1B"                         "$OE_MODELS_DIR/canary-1b"

echo -e "  ${BOLD}CV:${NC}"
check_variant "YOLOv8n-seg TRT"                  "$OE_ENGINES_DIR/yolov8n-seg.engine"
check_variant "YOLOv8s-seg TRT"                  "$OE_ENGINES_DIR/yolov8s-seg.engine"
check_variant "SAM2 Hiera Tiny Encoder"          "$OE_MODELS_DIR/sam2/sam2_hiera_tiny_encoder.onnx"
check_variant "SAM2 Hiera Tiny Decoder"          "$OE_MODELS_DIR/sam2/sam2_hiera_tiny_decoder.onnx"

echo ""
if [[ "$ERRORS" -gt 0 ]]; then
    log_error "$ERRORS error(s) during cross-model installation."
    echo ""
fi

echo -e "${BOLD}Next steps:${NC}"
echo "  # Run cross-model benchmark sweep"
echo "  python3 scripts/benchmark/benchmark_models.py --quant-sweep --vram-budget 12288"
echo ""
echo "  # Or via run_all.sh"
echo "  bash run_all.sh --model-test --quant-sweep --vram-budget 12288"
echo ""
exit $(( ERRORS > 0 ? 1 : 0 ))
