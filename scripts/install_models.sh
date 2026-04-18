#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Model Installer
#
# Downloads all required models to ~/omniedge_models/ (or $OE_MODELS_DIR).
# Idempotent — skips models that already exist.
#
# Usage:
#   bash scripts/install_models.sh            # interactive (choose conversation model)
#   bash scripts/install_models.sh --all      # download everything
#   bash scripts/install_models.sh --minimal  # ONNX models only (no LLM)
#   bash scripts/install_models.sh --status   # show which models are installed
# =============================================================================
set -euo pipefail

MODELS_DIR="${OE_MODELS_DIR:-$HOME/omniedge_models}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================="
echo "  OmniEdge_AI — Model Installer"
echo "  Target: $MODELS_DIR"
echo "============================================="
echo ""

mkdir -p "$MODELS_DIR"

# ── Check dependencies ──────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}ERROR: python3 not found. Install Python 3.10+.${NC}"
    exit 1
fi

if ! python3 -c "import huggingface_hub" &>/dev/null; then
    echo "Installing huggingface-hub..."
    pip3 install --user huggingface-hub
fi

if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
    echo -e "${RED}ERROR: wget or curl required.${NC}"
    exit 1
fi

# ── Helper: download from HuggingFace ───────────────────────────────────────
download_hf() {
    local repo="$1"
    local local_dir="$2"
    local dest="$MODELS_DIR/$local_dir"

    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo -e "  ${GREEN}SKIP${NC}  $local_dir (already exists)"
        return 0
    fi

    echo -e "  ${YELLOW}DOWNLOADING${NC}  $repo → $local_dir"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dest')
" && echo -e "  ${GREEN}OK${NC}    $local_dir" \
  || echo -e "  ${RED}FAIL${NC}  $local_dir"
}

# ── Helper: download single file ────────────────────────────────────────────
download_file() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo -e "  ${GREEN}SKIP${NC}  $(basename "$dest") (already exists)"
        return 0
    fi

    echo -e "  ${YELLOW}DOWNLOADING${NC}  $(basename "$dest")"
    mkdir -p "$(dirname "$dest")"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -fSL -o "$dest" "$url"
    fi
    echo -e "  ${GREEN}OK${NC}    $(basename "$dest")"
}

# ── Parse mode argument early (before any downloads) ──────────────────────
MODE="${1:-}"

if [ "$MODE" = "--status" ]; then
    echo "── Model Status ──────────────────────────────────────────"
    check() { [ -e "$1" ] && echo -e "  ${GREEN}OK${NC}    $2" || echo -e "  ${RED}MISSING${NC}  $2"; }
    check "$MODELS_DIR/silero_vad.onnx" "Silero VAD (2 MB)"
    check "$MODELS_DIR/kokoro-onnx" "Kokoro TTS ONNX (1.5 GB)"
    check "$MODELS_DIR/kokoro" "Kokoro voices (544 MB)"
    check "$MODELS_DIR/dtln/dtln_1.onnx" "DTLN audio denoise (4 MB)"
    check "$MODELS_DIR/yolo_work/yolov8n-seg.onnx" "YOLOv8n-seg (14 MB)"
    check "$MODELS_DIR/whisper-large-v3-turbo" "Whisper Large v3 Turbo (3.1 GB)"
    check "$MODELS_DIR/qwen2.5-7b-instruct-awq" "Qwen2.5-7B-Instruct-AWQ (5.2 GB)"
    check "$MODELS_DIR/qwen2.5-3b-instruct-awq" "Qwen2.5-3B-Instruct-AWQ (2.6 GB)"
    echo ""
    echo "── Conversation Models ───────────────────────────────────"
    check "$MODELS_DIR/qwen2.5-omni-3b" "Qwen2.5-Omni-3B (6 GB)"
    check "$MODELS_DIR/qwen2.5-omni-7b" "Qwen2.5-Omni-7B (14 GB)"
    check "$MODELS_DIR/gemma-4-e4b" "Gemma 4 E4B (15 GB)"
    echo ""
    TOTAL_SIZE=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
    echo "  Location: $MODELS_DIR"
    echo "  Total:    $TOTAL_SIZE"
    exit 0
fi

# =============================================================================
# 1. ONNX Models (always needed, portable, small)
# =============================================================================
echo "── ONNX Models (always needed) ──────────────────────────"

# Silero VAD v5 (2.3 MB) — voice activity detection
download_file \
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
    "$MODELS_DIR/silero_vad.onnx"

# Kokoro TTS ONNX (1.5 GB) — text-to-speech engine
download_hf "hexgrad/Kokoro-82M-v1.1-ONNX" "kokoro-onnx"

# Kokoro voice packs (544 MB) — TTS voices
download_hf "hexgrad/Kokoro-82M" "kokoro"

# BasicVSR++ (88 MB) — video super-resolution / denoise
if [ ! -f "$MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx" ]; then
    echo -e "  ${YELLOW}DOWNLOADING${NC}  basicvsrpp"
    mkdir -p "$MODELS_DIR/basicvsrpp"
    # BasicVSR++ is exported manually from MMEditing — check if pre-exported exists
    if [ -f "$MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx" ]; then
        echo -e "  ${GREEN}SKIP${NC}  basicvsrpp (already exists)"
    else
        echo -e "  ${YELLOW}NOTE${NC}  BasicVSR++ ONNX must be pre-exported. Place basicvsrpp_denoise.onnx in $MODELS_DIR/basicvsrpp/"
    fi
else
    echo -e "  ${GREEN}SKIP${NC}  basicvsrpp (already exists)"
fi

# DTLN audio denoise (3.8 MB) — two-stage LSTM noise reduction
if [ ! -f "$MODELS_DIR/dtln/dtln_1.onnx" ]; then
    echo -e "  ${YELLOW}DOWNLOADING${NC}  DTLN audio denoise"
    mkdir -p "$MODELS_DIR/dtln"
    download_file \
        "https://github.com/breizhn/DTLN/raw/master/pretrained_model/dtln_1.onnx" \
        "$MODELS_DIR/dtln/dtln_1.onnx"
    download_file \
        "https://github.com/breizhn/DTLN/raw/master/pretrained_model/dtln_2.onnx" \
        "$MODELS_DIR/dtln/dtln_2.onnx"
else
    echo -e "  ${GREEN}SKIP${NC}  dtln (already exists)"
fi

# Face models: SCRFD detector + AuraFace v1 recognizer (Apache 2.0, glintr100)
if [ ! -d "$MODELS_DIR/face_models/scrfd_auraface" ] || \
   [ ! -f "$MODELS_DIR/face_models/scrfd_auraface/detector.onnx" ]; then
    echo -e "  ${YELLOW}NOTE${NC}  SCRFD detector.onnx must be in $MODELS_DIR/face_models/scrfd_auraface/"
    echo "         AuraFace recognizer.onnx is auto-downloaded from fal/AuraFace-v1 on first run."
else
    echo -e "  ${GREEN}SKIP${NC}  face_models (already exists)"
fi

# FaceMesh V2 landmarks (3.6 MB)
if [ ! -f "$MODELS_DIR/face_mesh/face_landmark.tflite" ]; then
    echo -e "  ${YELLOW}DOWNLOADING${NC}  FaceMesh V2"
    mkdir -p "$MODELS_DIR/face_mesh"
    download_file \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
        "$MODELS_DIR/face_mesh/face_landmark.tflite"
else
    echo -e "  ${GREEN}SKIP${NC}  face_mesh (already exists)"
fi

# SAM2 — Segment Anything Model 2 (encoder + decoder ONNX)
if [ ! -f "$MODELS_DIR/sam2/sam2_hiera_tiny_encoder.onnx" ]; then
    echo -e "  ${YELLOW}NOTE${NC}  SAM2 ONNX models must be pre-exported."
    echo "         Place sam2_hiera_tiny_encoder.onnx + sam2_hiera_tiny_decoder.onnx in $MODELS_DIR/sam2/"
else
    echo -e "  ${GREEN}SKIP${NC}  sam2 (already exists)"
fi

# YOLOv8n-seg (14 MB) — background blur (bg_blur stub pending Phase 10b swap to MediaPipe Selfie Seg).
if [ ! -f "$MODELS_DIR/yolo_work/yolov8n-seg.onnx" ]; then
    echo -e "  ${YELLOW}DOWNLOADING${NC}  YOLOv8n-seg"
    mkdir -p "$MODELS_DIR/yolo_work"
    python3 -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8n-seg.pt')
model.export(format='onnx', imgsz=640, half=True, simplify=True)
shutil.move('yolov8n-seg.onnx', '$MODELS_DIR/yolo_work/yolov8n-seg.onnx')
# Clean up downloaded .pt
os.remove('yolov8n-seg.pt') if os.path.exists('yolov8n-seg.pt') else None
" && echo -e "  ${GREEN}OK${NC}    yolov8n-seg.onnx" \
  || echo -e "  ${RED}FAIL${NC}  YOLOv8n-seg export (install ultralytics: pip install ultralytics)"
else
    echo -e "  ${GREEN}SKIP${NC}  yolo_work (already exists)"
fi

# YOLOX-Nano (3.9 MB) — security camera detection (Apache 2.0).
# Auto-fetched at runtime from Megvii-BaseDetection/YOLOX via fetchHfModel,
# but pre-downloading keeps first-boot latency low.
if [ ! -f "$MODELS_DIR/security/yolox_nano.onnx" ]; then
    echo -e "  ${YELLOW}DOWNLOADING${NC}  YOLOX-Nano"
    mkdir -p "$MODELS_DIR/security"
    wget -q --show-progress \
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx" \
        -O "$MODELS_DIR/security/yolox_nano.onnx" \
        && echo -e "  ${GREEN}OK${NC}    yolox_nano.onnx" \
        || echo -e "  ${RED}FAIL${NC}  YOLOX-Nano download"
else
    echo -e "  ${GREEN}SKIP${NC}  yolox_nano (already exists)"
fi

echo ""

# =============================================================================
# 2. TRT-LLM Source Weights (needed for first-run engine builds)
# =============================================================================
echo "── TRT-LLM Source Weights ──────────────────────────────"

# Qwen2.5-7B-Instruct-AWQ (5.2 GB) — tokenizer + AWQ weights for TRT engine
download_hf "Qwen/Qwen2.5-7B-Instruct-AWQ" "qwen2.5-7b-instruct-awq"

# Qwen2.5-3B-Instruct-AWQ (2.6 GB) — lighter variant
download_hf "Qwen/Qwen2.5-3B-Instruct-AWQ" "qwen2.5-3b-instruct-awq"

# Whisper Large v3 Turbo (3.1 GB) — STT engine build
download_hf "openai/whisper-large-v3-turbo" "whisper-large-v3-turbo"

echo ""

# =============================================================================
# 3. Conversation Models (user selects)
# =============================================================================

if [ "$MODE" = "--minimal" ]; then
    echo -e "${YELLOW}Minimal mode: skipping conversation models.${NC}"
    echo "  You can run this script again later to add conversation models."
elif [ "$MODE" = "--all" ]; then
    echo "── Conversation Models (all) ──────────────────────────────"
    download_hf "Qwen/Qwen2.5-Omni-3B" "qwen2.5-omni-3b"
    download_hf "Qwen/Qwen2.5-Omni-7B" "qwen2.5-omni-7b"
    download_hf "google/gemma-4-e4b" "gemma-4-e4b"
else
    echo "── Conversation Models ────────────────────────────────────"
    echo ""
    echo "  Choose which conversation model(s) to download:"
    echo "    1. Qwen2.5-Omni-3B   (~6 GB)  — multimodal STT+LLM+TTS, fastest"
    echo "    2. Qwen2.5-Omni-7B   (~14 GB) — multimodal STT+LLM+TTS, best quality"
    echo "    3. Gemma 4 E4B        (~15 GB) — MoE multimodal, needs TTS sidecar"
    echo "    4. All of the above"
    echo "    5. Skip (text-only mode with Qwen2.5-7B-AWQ TRT engine)"
    echo ""
    read -rp "  Select [1-5, default=1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1)
            download_hf "Qwen/Qwen2.5-Omni-3B" "qwen2.5-omni-3b"
            ;;
        2)
            download_hf "Qwen/Qwen2.5-Omni-7B" "qwen2.5-omni-7b"
            ;;
        3)
            download_hf "google/gemma-4-e4b" "gemma-4-e4b"
            ;;
        4)
            download_hf "Qwen/Qwen2.5-Omni-3B" "qwen2.5-omni-3b"
            download_hf "Qwen/Qwen2.5-Omni-7B" "qwen2.5-omni-7b"
            download_hf "google/gemma-4-e4b" "gemma-4-e4b"
            ;;
        5)
            echo -e "  ${YELLOW}Skipping conversation models.${NC}"
            echo "  Text-only mode will use Qwen2.5-7B-AWQ with TRT-LLM engine."
            ;;
        *)
            echo -e "  ${YELLOW}Invalid choice. Defaulting to Qwen2.5-Omni-3B.${NC}"
            download_hf "Qwen/Qwen2.5-Omni-3B" "qwen2.5-omni-3b"
            ;;
    esac
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================="
TOTAL_SIZE=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
echo -e "  ${GREEN}Model installation complete.${NC}"
echo "  Location:   $MODELS_DIR"
echo "  Total size: $TOTAL_SIZE"
echo ""
echo "  Next steps:"
echo "    1. Enter dev container:  bash scripts/docker/dev-shell.sh"
echo "    2. Run the system:      bash run_all.sh"
echo "    3. Open browser:        http://localhost:9001"
echo "============================================="
