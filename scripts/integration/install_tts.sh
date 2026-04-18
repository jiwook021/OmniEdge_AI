#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — TTS installation (Kokoro ONNX + INT8, Piper, VITS2)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_TTS_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_TTS_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  4. TTS — Kokoro ONNX + INT8
# ═══════════════════════════════════════════════════════════════════════════════
install_tts() {
    section "TTS — Kokoro ONNX + INT8"

    # ONNX Runtime with CUDA EP
    pip install --break-system-packages onnxruntime-gpu 2>/dev/null || true

    local KOKORO_ONNX_DIR="$OE_MODELS_DIR/kokoro-onnx/onnx"

    # Download voices (into kokoro/ which is the voices-only download)
    if [ -d "$OE_MODELS_DIR/kokoro/voices" ] && [ -n "$(ls -A "$OE_MODELS_DIR/kokoro/voices" 2>/dev/null)" ]; then
        skip "Kokoro voices"
    else
        echo "  Downloading Kokoro voices..."
        mkdir -p "$OE_MODELS_DIR/kokoro"
        hf download hexgrad/Kokoro-82M --include "voices/*" --local-dir "$OE_MODELS_DIR/kokoro"
        ok "Kokoro voices downloaded"
    fi

    # Download ONNX model (into kokoro-onnx/ — the ONNX community variant)
    if [ -f "$KOKORO_ONNX_DIR/model.onnx" ]; then
        skip "Kokoro ONNX model"
    else
        echo "  Downloading Kokoro ONNX model (~200 MB)..."
        mkdir -p "$(dirname "$KOKORO_ONNX_DIR")"
        hf download onnx-community/Kokoro-82M-v1.0-ONNX --local-dir "$OE_MODELS_DIR/kokoro-onnx"
        ok "Kokoro ONNX model downloaded"
    fi

    # Quantize to INT8 (stays in kokoro-onnx/onnx/model_int8.onnx)
    if [ -f "$KOKORO_ONNX_DIR/model_int8.onnx" ]; then
        skip "Kokoro INT8 quantized model"
    else
        echo "  Quantizing Kokoro to INT8..."
        python3 "$PROJECT_ROOT/scripts/quantize/quantize_kokoro_onnx.py" || record_error "Kokoro INT8 quantization failed"
    fi

    # Verify load
    python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('$KOKORO_ONNX_DIR/model_int8.onnx',
    providers=['CUDAExecutionProvider','CPUExecutionProvider'])
print('  Kokoro INT8 loads OK. Inputs:', [i.name for i in sess.get_inputs()])
" 2>/dev/null || warn "Kokoro INT8 model failed to load in ONNX Runtime"

    # ── Piper TTS ONNX (lightweight CPU-friendly alternative) ──
    local PIPER_DIR="$OE_MODELS_DIR/piper"
    local PIPER_ONNX="$PIPER_DIR/en_US-lessac-medium.onnx"
    if [ -f "$PIPER_ONNX" ]; then
        skip "Piper TTS ONNX model"
    else
        echo "  Downloading Piper TTS en_US-lessac-medium (~80 MB)..."
        mkdir -p "$PIPER_DIR"
        curl -fsSL -L -o "$PIPER_ONNX" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        && curl -fsSL -L -o "${PIPER_ONNX}.json" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        && ok "Piper TTS ONNX model" \
        || record_error "Piper TTS download failed"
    fi

    # ── VITS2 ONNX (alternative TTS architecture) ──
    local VITS2_DIR="$OE_MODELS_DIR/vits2"
    local VITS2_ONNX="$VITS2_DIR/model.onnx"
    if [ -f "$VITS2_ONNX" ]; then
        skip "VITS2 ONNX model"
    else
        echo "  Exporting VITS2 to ONNX..."
        mkdir -p "$VITS2_DIR"
        # VITS2 has no standard pre-exported ONNX on HuggingFace.
        # Attempt export from the vits2 Python package if installed.
        python3 -c "
import torch, os

try:
    from vits2 import VITS2
    model = VITS2.from_pretrained('vits2-ljs')
    model.eval()
    dummy_text = torch.randint(0, 100, (1, 50))
    dummy_len = torch.tensor([50])
    torch.onnx.export(model, (dummy_text, dummy_len),
                      '$VITS2_ONNX',
                      input_names=['text', 'text_lengths'],
                      output_names=['audio'],
                      dynamic_axes={'text': {1: 'seq_len'}, 'audio': {2: 'audio_len'}},
                      opset_version=17)
    print('VITS2 ONNX exported successfully')
except ImportError:
    print('VITS2 package not installed — skipping (pip install vits2)')
    raise SystemExit(0)
except Exception as e:
    print(f'VITS2 export failed: {e}')
    raise SystemExit(1)
" 2>/dev/null
        if [ -f "$VITS2_ONNX" ]; then
            ok "VITS2 ONNX model exported"
        else
            warn "VITS2 ONNX not available — export manually and place at $VITS2_ONNX"
        fi
    fi
}
