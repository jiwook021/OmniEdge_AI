#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — STT installation (Whisper Large V3 Turbo)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_STT_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_STT_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  3. STT — Whisper Large V3 Turbo
# ═══════════════════════════════════════════════════════════════════════════════
install_stt() {
    section "STT — Whisper Large V3 Turbo"

    local TRTLLM_ROOT="${TRTLLM_ROOT:-$HOME/TensorRT-LLM}"

    # Download model
    local WHISPER_DIR="$OE_MODELS_DIR/whisper-large-v3-turbo"
    if [ -d "$WHISPER_DIR" ] && [ -n "$(ls -A "$WHISPER_DIR" 2>/dev/null)" ]; then
        skip "Whisper model"
    else
        echo "  Downloading Whisper Large V3 Turbo (~3 GB)..."
        hf download openai/whisper-large-v3-turbo --local-dir "$WHISPER_DIR"
        ok "Whisper model downloaded"
    fi

    # Convert checkpoint
    local CKPT_DIR="$OE_MODELS_DIR/trt_ckpt/whisper-turbo-int8"
    if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
        skip "Whisper INT8 checkpoint"
    else
        if [ ! -d "$TRTLLM_ROOT" ]; then
            warn "TensorRT-LLM repo not found at $TRTLLM_ROOT — skipping checkpoint conversion"
        else
            # Whisper checkpoint conversion needs ~12-16 GB
            ensure_swap_for_step 16
            if ! require_memory_gb 10 "Whisper INT8 checkpoint conversion"; then
                true
            else
                echo "  Converting Whisper checkpoint to INT8..."
                export PYTHONPATH="${PYTHONPATH:-}:$TRTLLM_ROOT"
                mkdir -p "$CKPT_DIR"
                python3 "$TRTLLM_ROOT/examples/models/core/whisper/convert_checkpoint.py" \
                    --model_dir "$WHISPER_DIR" \
                    --output_dir "$CKPT_DIR" \
                    --dtype float16 \
                    --use_weight_only \
                    --weight_only_precision int8_float16 || {
                    record_error "Whisper checkpoint conversion failed"
                    [ -n "$CKPT_DIR" ] && rm -rf "$CKPT_DIR"
                }
            fi
        fi
    fi

    # Build encoder/decoder engines
    if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
        local ENC_DIR="$OE_ENGINES_DIR/whisper-turbo/encoder"
        local DEC_DIR="$OE_ENGINES_DIR/whisper-turbo/decoder"

        if [ -d "$ENC_DIR" ] && [ -n "$(ls -A "$ENC_DIR" 2>/dev/null)" ]; then
            skip "Whisper encoder engine"
        else
            ensure_swap_for_step 12
            if ! require_memory_gb 8 "Whisper encoder engine build"; then
                true
            else
                echo "  Building Whisper encoder engine..."
                mkdir -p "$ENC_DIR"
                python3 -m tensorrt_llm.commands.build \
                    --checkpoint_dir "$CKPT_DIR/encoder" \
                    --output_dir "$ENC_DIR" \
                    --max_batch_size 1 \
                    --max_input_len 3000 \
                    --gemm_plugin disable 2>&1 || record_error "Whisper encoder engine build failed"
            fi
        fi

        if [ -d "$DEC_DIR" ] && [ -n "$(ls -A "$DEC_DIR" 2>/dev/null)" ]; then
            skip "Whisper decoder engine"
        else
            ensure_swap_for_step 12
            if ! require_memory_gb 8 "Whisper decoder engine build"; then
                true
            else
                echo "  Building Whisper decoder engine..."
                mkdir -p "$DEC_DIR"
                python3 -m tensorrt_llm.commands.build \
                    --checkpoint_dir "$CKPT_DIR/decoder" \
                    --output_dir "$DEC_DIR" \
                    --max_batch_size 1 \
                    --max_seq_len 448 \
                    --max_encoder_input_len 3000 \
                    --gemm_plugin disable 2>&1 || record_error "Whisper decoder engine build failed"
            fi
        fi
    fi
}
