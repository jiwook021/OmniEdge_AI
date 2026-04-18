#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Enhancement model installation (BasicVSR++ + DTLN + SAM2)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_ENHANCEMENT_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_ENHANCEMENT_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Enhancement Models (BasicVSR++ + DTLN)
# ═══════════════════════════════════════════════════════════════════════════════
install_enhancement() {
    section "Enhancement Models (BasicVSR++ + DTLN)"

    # ── BasicVSR++ ONNX model ──
    local VSRPP_DIR="$OE_MODELS_DIR/basicvsrpp"
    local VSRPP_ONNX="$VSRPP_DIR/basicvsrpp_denoise.onnx"
    if [ -f "$VSRPP_ONNX" ]; then
        skip "BasicVSR++ ONNX already present: $VSRPP_ONNX"
    else
        mkdir -p "$VSRPP_DIR"
        echo "  Exporting BasicVSR++ to ONNX via mmagic..."
        # BasicVSR++ has no pre-exported ONNX on HuggingFace.
        # Export from mmagic (OpenMMLab) if available, otherwise skip gracefully.
        # Video super-resolution is optional — the system runs fine without it.

        # Check if PyTorch is an NGC custom build — mmcv has no prebuilt wheels
        # for NGC's custom PyTorch (version contains 'nv' or 'a0'), so skip.
        local TORCH_VER
        TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
        if [[ "$TORCH_VER" == *"nv"* ]] || [[ "$TORCH_VER" == *"a0"* ]]; then
            warn "BasicVSR++ skipped — mmcv has no prebuilt wheel for NGC PyTorch ($TORCH_VER)"
            warn "  Video super-resolution will be disabled (optional feature)"
        else
            pip install --break-system-packages mmagic mmengine mmcv 2>/dev/null || true
            python3 -c "
import torch, os
from mmagic.apis import MMagicInferencer
editor = MMagicInferencer('basicvsr_plusplus', seed=0)
model = editor.inferencer.model.eval()
# BasicVSR++ expects (B, T, C, H, W) — use 2 frames at 64x64 for tracing
dummy = torch.randn(1, 2, 3, 64, 64).cuda()
torch.onnx.export(model, dummy, '$VSRPP_ONNX',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {1: 'frames', 3: 'height', 4: 'width'},
                                'output': {1: 'frames', 3: 'height', 4: 'width'}},
                  opset_version=17)
print('BasicVSR++ ONNX exported successfully')
" 2>/dev/null
            if [ -f "$VSRPP_ONNX" ]; then
                ok "BasicVSR++ ONNX exported: $VSRPP_ONNX"
            else
                warn "BasicVSR++ export failed — video super-resolution will be disabled"
                warn "  Install mmagic manually: pip install mmagic mmengine mmcv"
            fi
        fi
    fi

    # ── DTLN ONNX models (two stages) ──
    local DTLN_DIR="$OE_MODELS_DIR/dtln"
    local DTLN_1="$DTLN_DIR/dtln_1.onnx"
    local DTLN_2="$DTLN_DIR/dtln_2.onnx"
    if [ -f "$DTLN_1" ] && [ -f "$DTLN_2" ]; then
        skip "DTLN ONNX models already present: $DTLN_DIR"
    else
        mkdir -p "$DTLN_DIR"
        echo "  Downloading DTLN ONNX models..."
        # Download from GitHub (breizhn/DTLN) — upstream names: model_1.onnx, model_2.onnx
        local DTLN_BASE_URL="https://github.com/breizhn/DTLN/raw/master/pretrained_model"
        curl -fsSL -L "$DTLN_BASE_URL/model_1.onnx" -o "$DTLN_1" \
            || record_error "DTLN stage-1 download failed"
        curl -fsSL -L "$DTLN_BASE_URL/model_2.onnx" -o "$DTLN_2" \
            || record_error "DTLN stage-2 download failed"
        if [ -f "$DTLN_1" ] && [ -f "$DTLN_2" ]; then
            ok "DTLN ONNX models downloaded: $DTLN_DIR"
        else
            warn "DTLN models not found after download — module will be unavailable"
        fi
    fi

    # ── SAM2 ONNX models (Segment Anything Model 2 — interactive segmentation) ──
    local SAM2_DIR="$OE_MODELS_DIR/sam2"
    local SAM2_ENCODER="$SAM2_DIR/sam2_hiera_tiny_encoder.onnx"
    local SAM2_DECODER="$SAM2_DIR/sam2_hiera_tiny_decoder.onnx"
    if [ -f "$SAM2_ENCODER" ] && [ -f "$SAM2_DECODER" ]; then
        skip "SAM2 Hiera Tiny ONNX models already present: $SAM2_DIR"
    else
        # facebook/sam2-hiera-tiny distributes only .pt/.safetensors — no ONNX.
        # Converting SAM2 to ONNX requires splitting encoder/decoder and is
        # non-trivial. Mark as optional/manual for now.
        mkdir -p "$SAM2_DIR"
        warn "SAM2 ONNX models require manual export from PyTorch (not distributed as ONNX)"
        warn "  SAM2 segmentation will be disabled (optional feature)"
    fi
}
