#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Docker Entrypoint (Mode-Aware)
# =============================================================================
set -euo pipefail

ENGINES_DIR="${OE_ENGINES_DIR:-/opt/omniedge/engines}"
MODELS_DIR="${OE_MODELS_DIR:-/opt/omniedge/models}"
SERVICE_MODE="${OE_SERVICE_MODE:-conversation}"

if [[ "$SERVICE_MODE" != "conversation" && "$SERVICE_MODE" != "security" && "$SERVICE_MODE" != "beauty" ]]; then
    SERVICE_MODE="conversation"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================="
echo "  OmniEdge_AI — Container Starting"
echo "  Service mode: ${SERVICE_MODE}"
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

if [[ "${VRAM_TOTAL}" -lt 11000 ]]; then
    echo -e "${YELLOW}WARNING: < 12 GB VRAM detected. Some models may not fit.${NC}"
fi

# ── Validate Model Mount ────────────────────────────────────────────────────
if [[ ! -d "$MODELS_DIR" ]] || [[ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]]; then
    echo -e "${RED}ERROR: No models found at $MODELS_DIR${NC}"
    echo "  Models must be mounted from the host."
    exit 1
fi

# Stage repo-bundled VAD model when missing.
if [[ -f /opt/omniedge/models/silero_vad.onnx ]] && [[ ! -f "$MODELS_DIR/silero_vad.onnx" ]]; then
    cp /opt/omniedge/models/silero_vad.onnx "$MODELS_DIR/silero_vad.onnx" 2>/dev/null || true
fi

# Bridge engine paths: configs resolve models_root/trt_engines.
mkdir -p "$ENGINES_DIR"
if ! ln -sfn "$ENGINES_DIR" "$MODELS_DIR/trt_engines" 2>/dev/null; then
    echo -e "  ${YELLOW}NOTE${NC} Models mount is read-only — using existing trt_engines path"
fi

export OE_MODELS_DIR="$MODELS_DIR"
export OE_ENGINES_DIR="$ENGINES_DIR"

# ── Generate mode INI ───────────────────────────────────────────────────────
MODE_INI="/tmp/omniedge.${SERVICE_MODE}.ini"
FRONTEND_DIR="/opt/omniedge/share/frontend/${SERVICE_MODE}"
if [[ ! -d "$FRONTEND_DIR" ]]; then
    FRONTEND_DIR="/opt/omniedge/share/frontend"
fi

bash /opt/omniedge/scripts/generate_mode_ini.sh \
    "$SERVICE_MODE" "$MODE_INI" "$FRONTEND_DIR" >/dev/null

echo -e "${GREEN}Config:${NC} mode INI generated at $MODE_INI"

# ── Launch binaries (canonical names, mode-specific image pruning) ─────────
WS_BIN="/opt/omniedge/bin/omniedge_ws_bridge"
DAEMON_BIN="/opt/omniedge/bin/omniedge_daemon"

REQUIRED_BINS=("$WS_BIN" "$DAEMON_BIN")
case "$SERVICE_MODE" in
    conversation)
        REQUIRED_BINS+=(
            "/opt/omniedge/bin/omniedge_video_ingest"
            "/opt/omniedge/bin/omniedge_audio_ingest"
            "/opt/omniedge/bin/omniedge_screen_ingest"
            "/opt/omniedge/bin/omniedge_bg_blur"
            "/opt/omniedge/bin/omniedge_conversation"
            "/opt/omniedge/bin/omniedge_tts"
            "/opt/omniedge/bin/omniedge_audio_denoise"
        )
        ;;
    security)
        REQUIRED_BINS+=(
            "/opt/omniedge/bin/omniedge_video_ingest"
            "/opt/omniedge/bin/omniedge_audio_ingest"
            "/opt/omniedge/bin/omniedge_security_camera"
            "/opt/omniedge/bin/omniedge_security_vlm"
        )
        ;;
    beauty)
        REQUIRED_BINS+=(
            "/opt/omniedge/bin/omniedge_video_ingest"
            "/opt/omniedge/bin/omniedge_audio_ingest"
            "/opt/omniedge/bin/omniedge_beauty"
        )
        ;;
esac

MISSING=()
for bin in "${REQUIRED_BINS[@]}"; do
    if [[ ! -x "$bin" ]]; then
        MISSING+=("$bin")
    fi
done
if [[ "${#MISSING[@]}" -gt 0 ]]; then
    echo -e "${RED}ERROR: required binaries missing for mode '${SERVICE_MODE}':${NC}"
    for bin in "${MISSING[@]}"; do
        echo "  - $bin"
    done
    exit 1
fi

WS_LOG="/var/log/omniedge_ws_bridge.log"
DAEMON_LOG="/var/log/omniedge_daemon.log"
mkdir -p /var/log

echo "Starting ws_bridge..."
"$WS_BIN" --config "$MODE_INI" >"$WS_LOG" 2>&1 &
WS_PID=$!
sleep 1
if ! kill -0 "$WS_PID" 2>/dev/null; then
    echo -e "${RED}ERROR: ws_bridge failed to start.${NC}"
    tail -n 40 "$WS_LOG" || true
    exit 1
fi

cleanup() {
    kill "$WS_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting daemon..."
exec "$DAEMON_BIN" --config /opt/omniedge/etc/omniedge_config.yaml --ini "$MODE_INI" 2>&1 | tee "$DAEMON_LOG"
