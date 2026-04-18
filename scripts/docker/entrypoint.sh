#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Docker Container Entrypoint
#
# Validates GPU, model mount, and disk space, then delegates to run_all.sh.
# =============================================================================
set -uo pipefail

ENGINES_DIR="${OE_ENGINES_DIR:-/opt/omniedge/engines}"
MODELS_DIR="${OE_MODELS_DIR:-/opt/omniedge/models}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================="
echo "  OmniEdge_AI — Container Starting"
echo "============================================="

# ── GPU Check ─────────────────────────────────────────────────────────────────
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}ERROR: No NVIDIA GPU detected.${NC}"
    echo "  Install nvidia-container-toolkit and verify:"
    echo "    docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.12-py3 nvidia-smi"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo -e "${GREEN}GPU:${NC} ${GPU_NAME} (${VRAM_TOTAL} MiB VRAM)"

if [ "${VRAM_TOTAL}" -lt 11000 ]; then
    echo -e "${YELLOW}WARNING: < 12 GB VRAM detected. Some models may not fit.${NC}"
fi

# ── Validate Model Mount ────────────────────────────────────────────────────
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    echo -e "${RED}ERROR: No models found at $MODELS_DIR${NC}"
    echo "  Models must be mounted from the host:"
    echo "    1. Run: bash scripts/install_models.sh"
    echo "    2. Ensure OE_MODELS_DIR is set in .env or docker-compose.yaml"
    exit 1
fi

# Check available disk
AVAIL_GB=$(df -BG "$MODELS_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "${AVAIL_GB:-0}" -lt 5 ]; then
    echo -e "${YELLOW}WARNING: Only ${AVAIL_GB}GB free in model directory.${NC}"
fi

# ── Clean Up Orphans from Previous Run ────────────────────────────────────────
echo "Cleaning up stale processes..."
pkill -9 -f "qwen_generate.py" 2>/dev/null || true
pkill -9 -f "model_generate.py" 2>/dev/null || true
pkill -9 -f "omniedge_" 2>/dev/null || true
sleep 1

# Clean stale build artifacts
rm -rf /tmp/trtllm* /tmp/pip-* /root/.cache/pip 2>/dev/null || true

# ── Bridge Engine Paths ──────────────────────────────────────────────────────
# Config resolves engine paths relative to models_root (e.g. trt_engines/qwen2.5-7b-nvfp4),
# but Docker persists engines in $ENGINES_DIR via a named volume.
# Symlink so both paths resolve to the same directory.
mkdir -p "$ENGINES_DIR"
if ! ln -sfn "$ENGINES_DIR" "$MODELS_DIR/trt_engines" 2>/dev/null; then
    echo -e "  ${YELLOW}NOTE${NC}  Models mount is read-only — using host trt_engines"
fi

# Clean stale checkpoint from previous failed builds
rm -rf "$ENGINES_DIR/../trt_ckpt" 2>/dev/null || true

# ── Delegate to run_all.sh ───────────────────────────────────────────────────
# All installation, building, and launching is handled by run_all.sh.
export OE_MODELS_DIR="$MODELS_DIR"
export OE_ENGINES_DIR="$ENGINES_DIR"

cd /opt/omniedge/src
exec bash run_all.sh --install
