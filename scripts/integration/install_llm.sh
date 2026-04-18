#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — LLM installation (TensorRT-LLM + Qwen + Gemma + Llama)
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_LLM_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_LLM_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_memory.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  2. LLM — TensorRT-LLM + Qwen 2.5 7B
# ═══════════════════════════════════════════════════════════════════════════════
install_llm() {
    section "LLM — TensorRT-LLM + Conversation Models"

    # Fix Open MPI initialization in Docker containers without --ipc=host.
    # TensorRT-LLM internally initializes MPI (via NCCL) even for single-GPU
    # quantization (tp_size=1). Without these, opal_shmem_base_open fails
    # because the POSIX shmem component can't initialize in an isolated
    # Docker IPC namespace.
    # --- OpenMPI workarounds for Docker containers (single-GPU, tp_size=1) ---
    # TRT-LLM internally initializes MPI even for single-GPU quantization.
    # The NGC container's OpenMPI often can't find shmem plugins or access
    # /dev/shm properly. These vars fix all known failure modes:
    export OMPI_MCA_btl=self,tcp                      # No shared-memory btl
    export OMPI_MCA_btl_tcp_if_include=lo             # Loopback only (container-safe)
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    export OMPI_MCA_shmem=posix                       # Prefer posix shmem
    export OMPI_MCA_orte_tmpdir_base=/tmp             # Writable tmpdir for orte
    export PMIX_MCA_gds=hash
    export OMPI_MCA_opal_warn_on_missing_libcuda=0
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    export TRTLLM_USE_MPI_KVCACHE=0
    export WORLD_SIZE=1
    export RANK=0
    export PYTHONUNBUFFERED=1

    # Ensure /dev/shm is writable for POSIX shmem (mount may be read-only in some containers)
    if [ -d /dev/shm ] && ! touch /dev/shm/.ompi_test 2>/dev/null; then
        # /dev/shm not writable — mount a tmpfs over it
        mount -t tmpfs -o size=64m tmpfs /dev/shm 2>/dev/null || true
    else
        rm -f /dev/shm/.ompi_test 2>/dev/null
    fi

    # Ensure huggingface-cli is available for model downloads
    if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
        echo "  Installing huggingface-hub CLI..."
        pip install --break-system-packages huggingface-hub[cli] || {
            record_error "huggingface-hub CLI not found — install with: pip install huggingface-hub[cli]"
            return
        }
    fi

    # TensorRT-LLM
    if python3 -c "import tensorrt_llm" &>/dev/null; then
        local _trtllm_ver
        _trtllm_ver="$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null | tail -1)"
        skip "tensorrt_llm ${_trtllm_ver}"
    else
        echo "  Installing TensorRT-LLM..."
        pip install --break-system-packages tensorrt-llm --extra-index-url https://pypi.nvidia.com
        ok "tensorrt_llm installed"
    fi

    # ── Clone TRT-LLM repo for quantize.py if not present ──
    local TRTLLM_REPO="$HOME/TensorRT-LLM"
    if [ ! -d "$TRTLLM_REPO" ]; then
        log_info "Cloning TensorRT-LLM repo for quantization scripts..."
        git clone --depth 1 --branch v1.2.0 \
            https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_REPO" 2>&1 || \
            warn "TensorRT-LLM clone failed — quantization will be skipped"
    fi

    # Patch TRT-LLM for TensorRT 10.14+ API changes:
    # 1. parameter.py: set_weights_name now requires trt.Weights, not numpy
    # 2. builder.py: IConstantLayer.weights returns numpy copy, not trt.Weights ref
    local TRTLLM_SITE
    TRTLLM_SITE="$(python3 -c 'import tensorrt_llm; import os; print(os.path.dirname(tensorrt_llm.__file__))' 2>/dev/null | tail -1 || true)"
    if [[ -n "$TRTLLM_SITE" ]]; then
        local NEED_PATCH=0

        # Patch parameter.py: wrap numpy weights in trt.Weights
        if [[ -f "$TRTLLM_SITE/parameter.py" ]] && ! grep -q "isinstance(weights, trt.Weights)" "$TRTLLM_SITE/parameter.py" 2>/dev/null; then
            NEED_PATCH=1
            python3 -c "
import pathlib
p = pathlib.Path('$TRTLLM_SITE/parameter.py')
src = p.read_text()
old = '''            return network.trt_network.set_weights_name(
                self._get_weights(network), name)'''
new = '''            weights = self._get_weights(network)
            # TensorRT 10.14+ requires trt.Weights, not numpy arrays
            if not isinstance(weights, trt.Weights) and weights is not None:
                import numpy as np
                if isinstance(weights, np.ndarray):
                    weights = trt.Weights(np.ascontiguousarray(weights))
                else:
                    weights = trt.Weights(weights)
            return network.trt_network.set_weights_name(weights, name)'''
p.write_text(src.replace(old, new))
"
        fi

        # Patch builder.py: skip weight naming failures (non-critical for inference)
        if [[ -f "$TRTLLM_SITE/builder.py" ]] && grep -q "raise RuntimeError.*Failed to set weight" "$TRTLLM_SITE/builder.py" 2>/dev/null; then
            NEED_PATCH=1
            python3 -c "
import pathlib
p = pathlib.Path('$TRTLLM_SITE/builder.py')
src = p.read_text()
old = '''                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')
                # This mark_weights_refittable has no side effect when refit_individual is not enabled.
                network.trt_network.mark_weights_refittable(name)'''
new = '''                if not param.set_name(name, network):
                    # TensorRT 10.14+ changed IConstantLayer.weights to
                    # return a numpy copy instead of a trt.Weights reference.
                    # Weight naming for refit is non-critical - log and skip.
                    logger.debug(f'Weight naming skipped (TRT 10.14+ compat): {name}')
                else:
                    # This mark_weights_refittable has no side effect when refit_individual is not enabled.
                    network.trt_network.mark_weights_refittable(name)'''
p.write_text(src.replace(old, new))
"
        fi

        if [[ "$NEED_PATCH" -eq 1 ]]; then
            ok "TRT-LLM patched for TensorRT 10.14+"
        fi
    fi

    # Detect Blackwell by GPU name (compute_cap query is not universally available)
    local GPU_NAME
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

    local IS_BLACKWELL=0
    if echo "${GPU_NAME:-}" | grep -qi "blackwell"; then
        IS_BLACKWELL=1
        echo -e "  ${CYAN}Blackwell GPU detected (${GPU_NAME}) — using nvfp4 quantization${NC}"
    fi

    local RAM_GB
    RAM_GB=$(total_ram_gb)

    # Download Qwen 2.5 7B Instruct (FP16 base)
    local QWEN_DIR="$OE_MODELS_DIR/qwen2.5-7b-instruct"
    if [ -d "$QWEN_DIR" ] && [ -n "$(ls -A "$QWEN_DIR" 2>/dev/null)" ]; then
        skip "Qwen FP16 base model"
    else
        echo "  Downloading Qwen 2.5 7B Instruct (~14 GB)..."
        hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$QWEN_DIR"
        ok "Qwen FP16 model downloaded"
    fi

    if [ "$IS_BLACKWELL" -eq 1 ]; then
        # ── Blackwell: nvfp4 ──
        local CKPT_DIR="$OE_MODELS_DIR/trt_ckpt/qwen2.5-7b-nvfp4"
        local ENGINE_DIR="$OE_ENGINES_DIR/qwen2.5-7b-nvfp4"

        if [ ! -f "$CKPT_DIR/config.json" ]; then
            # nvfp4 quantization needs ~30 GB RAM (loads full FP16 model + calibration)
            ensure_swap_for_step 30
            if ! require_memory_gb 24 "Qwen nvfp4 quantization"; then
                echo -e "    ${YELLOW}Workaround: quantize on a machine with 32+ GB RAM, then copy${NC}"
                echo -e "    ${YELLOW}  $CKPT_DIR to this machine.${NC}"
            else
                echo "  Quantizing Qwen to nvfp4 (~30 GB RAM, may take 15-30 min)..."
                local TRTLLM_QUANTIZE
                TRTLLM_QUANTIZE="$(find "$HOME" -path "*/TensorRT-LLM/examples/quantization/quantize.py" 2>/dev/null | head -1)"
                if [ -z "$TRTLLM_QUANTIZE" ]; then
                    record_error "TRT-LLM quantize.py not found — clone TensorRT-LLM repo"
                else
                    mkdir -p "$CKPT_DIR"
                    # Qwen 7B FP16 (~14 GB) exceeds 12 GB VRAM — quantize on CPU
                    # to avoid modelopt mixed-device (cuda:0 + cpu) tensor error
                    python3 "$TRTLLM_QUANTIZE" \
                        --model_dir "$QWEN_DIR" \
                        --output_dir "$CKPT_DIR" \
                        --dtype float16 \
                        --qformat nvfp4 \
                        --calib_size 512 \
                        --device cpu \
                        --device_map cpu \
                        --tp_size 1 2>&1 || {
                        record_error "nvfp4 quantization failed"
                        [ -n "$CKPT_DIR" ] && rm -rf "$CKPT_DIR"
                    }
                fi
            fi
        else
            skip "nvfp4 checkpoint"
        fi

        if [ -f "$CKPT_DIR/config.json" ]; then
            if [ -d "$ENGINE_DIR" ] && [ -n "$(ls -A "$ENGINE_DIR" 2>/dev/null)" ]; then
                skip "nvfp4 TRT engine"
            else
                # TRT engine build needs ~16-20 GB
                ensure_swap_for_step 20
                if ! require_memory_gb 14 "Qwen TRT engine build"; then
                    echo -e "    ${YELLOW}Workaround: build engine on a machine with 24+ GB RAM${NC}"
                else
                    echo "  Building TRT engine from nvfp4 checkpoint (~16 GB RAM, ~10 min)..."
                    mkdir -p "$ENGINE_DIR"
                    # Remove stale model.cache directory (TRT expects a file, not a dir)
                    [ -d model.cache ] && rm -rf model.cache
                    python3 -m tensorrt_llm.commands.build \
                        --checkpoint_dir "$CKPT_DIR" \
                        --output_dir "$ENGINE_DIR" \
                        --gemm_plugin auto \
                        --gpt_attention_plugin auto \
                        --max_batch_size 1 \
                        --max_input_len 2048 \
                        --max_seq_len 4096 \
                        --max_num_tokens 4096 2>&1 || record_error "nvfp4 TRT engine build failed"
                fi
            fi
        fi
    else
        # ── Non-Blackwell: INT4-AWQ ──
        if [ "$RAM_GB" -ge 28 ]; then
            pip install --break-system-packages autoawq 2>/dev/null \
                || warn "autoawq install failed (version conflict)"
        else
            warn "Only ${RAM_GB} GB RAM — AWQ quantization needs ~32 GB"
        fi

        local QWEN_AWQ="$OE_MODELS_DIR/qwen2.5-7b-instruct-awq"
        if [ -d "$QWEN_AWQ" ] && [ -n "$(ls -A "$QWEN_AWQ" 2>/dev/null)" ]; then
            skip "Qwen AWQ model"
        else
            if python3 -c "import awq" 2>/dev/null && [ "$RAM_GB" -ge 28 ]; then
                echo "  Quantizing Qwen to INT4-AWQ (~20-40 min)..."
                python3 "$SCRIPT_DIR/quantize/quantize_qwen_awq.py"
            else
                echo "  Downloading pre-quantized Qwen AWQ (~4 GB)..."
                hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir "$QWEN_AWQ"
            fi
            ok "Qwen AWQ model ready"
        fi

        local CKPT_DIR="$OE_MODELS_DIR/trt_ckpt/qwen2.5-7b-awq"
        if [ -f "$CKPT_DIR/config.json" ]; then
            skip "AWQ TRT-LLM checkpoint"
        else
            ensure_swap_for_step 16
            if ! require_memory_gb 12 "Qwen AWQ checkpoint conversion"; then
                true  # skip silently — error already recorded
            else
                echo "  Converting to TRT-LLM checkpoint..."
                mkdir -p "$CKPT_DIR"
                python3 -c "
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
qc = QuantConfig(quant_algo='W4A16_AWQ')
model = QWenForCausalLM.from_hugging_face(
    '$QWEN_AWQ', dtype='float16', quant_config=qc, use_autoawq=True)
model.save_checkpoint('$CKPT_DIR')
print('Checkpoint saved: $CKPT_DIR')
" 2>&1 || {
                    record_error "TRT-LLM checkpoint conversion failed"
                    [ -n "$CKPT_DIR" ] && rm -rf "$CKPT_DIR"
                }
            fi
        fi

        local ENGINE_DIR="$OE_ENGINES_DIR/qwen2.5-7b-awq"
        if [ -f "$CKPT_DIR/config.json" ]; then
            if [ -d "$ENGINE_DIR" ] && [ -n "$(ls -A "$ENGINE_DIR" 2>/dev/null)" ]; then
                skip "AWQ TRT engine"
            else
                ensure_swap_for_step 20
                if ! require_memory_gb 14 "Qwen AWQ TRT engine build"; then
                    true
                else
                    echo "  Building TRT engine (~10 min)..."
                    mkdir -p "$ENGINE_DIR"
                    [ -d model.cache ] && rm -rf model.cache
                    python3 -m tensorrt_llm.commands.build \
                        --checkpoint_dir "$CKPT_DIR" \
                        --output_dir "$ENGINE_DIR" \
                        --gemm_plugin float16 \
                        --gpt_attention_plugin float16 \
                        --max_batch_size 1 \
                        --max_input_len 2048 \
                        --max_seq_len 4096 \
                        --max_num_tokens 4096 2>&1 || record_error "AWQ TRT engine build failed"
                fi
            fi
        fi
    fi

    # ── Qwen 2.5 7B AWQ (ensure AWQ engine exists for benchmarks) ────────
    # On Blackwell, symlink the nvfp4 engine (INT4 AWQ plugin crashes on sm_120).
    # On non-Blackwell, the AWQ engine was already built above — this block
    # is idempotent.
    local QWEN7B_AWQ_ENGINE="$OE_ENGINES_DIR/qwen2.5-7b-awq"

    if [ -d "$QWEN7B_AWQ_ENGINE" ] && [ -n "$(ls -A "$QWEN7B_AWQ_ENGINE" 2>/dev/null)" ]; then
        skip "Qwen 7B AWQ TRT engine (benchmark)"
    else
        if [ "$IS_BLACKWELL" -eq 1 ]; then
            # Symlink the nvfp4 engine into the AWQ path
            local NVFP4_ENGINE="$OE_ENGINES_DIR/qwen2.5-7b-nvfp4"
            if [ -f "$NVFP4_ENGINE/rank0.engine" ]; then
                mkdir -p "$QWEN7B_AWQ_ENGINE"
                cp "$NVFP4_ENGINE/config.json" "$QWEN7B_AWQ_ENGINE/config.json"
                ln -sf "$NVFP4_ENGINE/rank0.engine" "$QWEN7B_AWQ_ENGINE/rank0.engine"
                ok "Qwen 7B AWQ engine (symlinked from nvfp4)"
            else
                record_error "Qwen 7B NVFP4 engine not found — build nvfp4 first"
            fi
        else
            local QWEN7B_AWQ="$OE_MODELS_DIR/qwen2.5-7b-instruct-awq"
            local QWEN7B_AWQ_CKPT="$OE_MODELS_DIR/trt_ckpt/qwen2.5-7b-awq"

            # Download pre-quantized AWQ weights if not present
            if [ ! -d "$QWEN7B_AWQ" ] || [ -z "$(ls -A "$QWEN7B_AWQ" 2>/dev/null)" ]; then
                echo "  Downloading Qwen 2.5 7B Instruct AWQ (~4 GB)..."
                hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir "$QWEN7B_AWQ" \
                    || record_error "Qwen 7B AWQ download failed"
            fi

            # Convert checkpoint if needed
            if [ -d "$QWEN7B_AWQ" ] && [ -n "$(ls -A "$QWEN7B_AWQ" 2>/dev/null)" ]; then
                if [ ! -f "$QWEN7B_AWQ_CKPT/config.json" ]; then
                    ensure_swap_for_step 16
                    if require_memory_gb 12 "Qwen 7B AWQ checkpoint conversion"; then
                        echo "  Converting Qwen 7B AWQ to TRT-LLM checkpoint..."
                        mkdir -p "$QWEN7B_AWQ_CKPT"
                        python3 -c "
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
qc = QuantConfig(quant_algo='W4A16_AWQ')
model = QWenForCausalLM.from_hugging_face(
    '$QWEN7B_AWQ', dtype='float16', quant_config=qc, use_autoawq=True)
model.save_checkpoint('$QWEN7B_AWQ_CKPT')
print('Checkpoint saved: $QWEN7B_AWQ_CKPT')
" 2>&1 || {
                            record_error "Qwen 7B AWQ checkpoint conversion failed"
                            [ -n "$QWEN7B_AWQ_CKPT" ] && rm -rf "$QWEN7B_AWQ_CKPT"
                        }
                    fi
                fi

                # Build engine
                if [ -f "$QWEN7B_AWQ_CKPT/config.json" ]; then
                    ensure_swap_for_step 20
                    if require_memory_gb 14 "Qwen 7B AWQ TRT engine build"; then
                        echo "  Building Qwen 7B AWQ TRT engine..."
                        mkdir -p "$QWEN7B_AWQ_ENGINE"
                        [ -d model.cache ] && rm -rf model.cache
                        python3 -m tensorrt_llm.commands.build \
                            --checkpoint_dir "$QWEN7B_AWQ_CKPT" \
                            --output_dir "$QWEN7B_AWQ_ENGINE" \
                            --gemm_plugin float16 \
                            --gpt_attention_plugin float16 \
                            --max_batch_size 1 \
                            --max_input_len 2048 \
                            --max_seq_len 4096 \
                            --max_num_tokens 4096 2>&1 || record_error "Qwen 7B AWQ TRT engine build failed"
                        ok "Qwen 7B AWQ TRT engine built"
                    fi
                fi
            fi
        fi
    fi

    # ── Qwen 2.5 3B (smaller variant for constrained platforms) ──────────
    local QWEN3B_AWQ="$OE_MODELS_DIR/qwen2.5-3b-instruct-awq"
    local QWEN3B_ENGINE="$OE_ENGINES_DIR/qwen2.5-3b-awq"

    if [ -d "$QWEN3B_ENGINE" ] && [ -n "$(ls -A "$QWEN3B_ENGINE" 2>/dev/null)" ]; then
        skip "Qwen 3B TRT engine"
    else
        if [ "$IS_BLACKWELL" -eq 1 ]; then
            # ── Blackwell: NVFP4 (INT4 AWQ plugin crashes on sm_120) ──
            local QWEN3B_FP16="$OE_MODELS_DIR/qwen2.5-3b-instruct"
            local QWEN3B_CKPT="$OE_MODELS_DIR/trt_ckpt/qwen2.5-3b-nvfp4"

            # Download FP16 base model (~6 GB)
            if [ -d "$QWEN3B_FP16" ] && [ -n "$(ls -A "$QWEN3B_FP16" 2>/dev/null)" ]; then
                skip "Qwen 3B FP16 base model"
            else
                echo "  Downloading Qwen 2.5 3B Instruct FP16 (~6 GB)..."
                hf download Qwen/Qwen2.5-3B-Instruct --local-dir "$QWEN3B_FP16" \
                    || record_error "Qwen 3B FP16 download failed"
            fi

            # NVFP4 quantization checkpoint
            if [ ! -f "$QWEN3B_CKPT/config.json" ]; then
                local TRTLLM_QUANTIZE
                TRTLLM_QUANTIZE="$(find "$HOME" -path "*/TensorRT-LLM/examples/quantization/quantize.py" 2>/dev/null | head -1)"
                if [ -z "$TRTLLM_QUANTIZE" ]; then
                    record_error "TRT-LLM quantize.py not found — clone TensorRT-LLM repo"
                else
                    mkdir -p "$QWEN3B_CKPT"
                    python3 "$TRTLLM_QUANTIZE" \
                        --model_dir "$QWEN3B_FP16" \
                        --output_dir "$QWEN3B_CKPT" \
                        --dtype float16 \
                        --qformat nvfp4 \
                        --calib_size 512 \
                        --tp_size 1 2>&1 || {
                        record_error "Qwen 3B nvfp4 quantization failed"
                        [ -n "$QWEN3B_CKPT" ] && rm -rf "$QWEN3B_CKPT"
                    }
                fi
            fi

            # Build engine
            if [ -f "$QWEN3B_CKPT/config.json" ]; then
                ensure_swap_for_step 14
                if require_memory_gb 10 "Qwen 3B TRT engine build"; then
                    echo "  Building Qwen 3B nvfp4 TRT engine (~5 min)..."
                    mkdir -p "$QWEN3B_ENGINE"
                    [ -d model.cache ] && rm -rf model.cache
                    python3 -m tensorrt_llm.commands.build \
                        --checkpoint_dir "$QWEN3B_CKPT" \
                        --output_dir "$QWEN3B_ENGINE" \
                        --gemm_plugin auto \
                        --gpt_attention_plugin auto \
                        --max_batch_size 1 \
                        --max_input_len 2048 \
                        --max_seq_len 4096 \
                        --max_num_tokens 4096 2>&1 || record_error "Qwen 3B TRT engine build failed"
                    ok "Qwen 3B TRT engine built"
                fi
            fi
        else
            # ── Non-Blackwell: INT4-AWQ ──
            local QWEN3B_CKPT="$OE_MODELS_DIR/trt_ckpt/qwen2.5-3b-awq"

            # Download pre-quantized AWQ weights (~2 GB)
            if [ -d "$QWEN3B_AWQ" ] && [ -n "$(ls -A "$QWEN3B_AWQ" 2>/dev/null)" ]; then
                skip "Qwen 3B AWQ model"
            else
                echo "  Downloading Qwen 2.5 3B Instruct AWQ (~2 GB)..."
                hf download Qwen/Qwen2.5-3B-Instruct-AWQ --local-dir "$QWEN3B_AWQ" \
                    || record_error "Qwen 3B AWQ download failed"
            fi

            # Convert to TRT-LLM checkpoint
            if [ -d "$QWEN3B_AWQ" ] && [ -n "$(ls -A "$QWEN3B_AWQ" 2>/dev/null)" ]; then
                if [ ! -f "$QWEN3B_CKPT/config.json" ]; then
                    ensure_swap_for_step 12
                    if require_memory_gb 10 "Qwen 3B checkpoint conversion"; then
                        echo "  Converting Qwen 3B to TRT-LLM checkpoint..."
                        mkdir -p "$QWEN3B_CKPT"
                        python3 -c "
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
qc = QuantConfig(quant_algo='W4A16_AWQ')
model = QWenForCausalLM.from_hugging_face(
    '$QWEN3B_AWQ', dtype='float16', quant_config=qc, use_autoawq=True)
model.save_checkpoint('$QWEN3B_CKPT')
print('Checkpoint saved: $QWEN3B_CKPT')
" 2>&1 || {
                            record_error "Qwen 3B checkpoint conversion failed"
                            [ -n "$QWEN3B_CKPT" ] && rm -rf "$QWEN3B_CKPT"
                        }
                    fi
                else
                    skip "Qwen 3B TRT-LLM checkpoint"
                fi

                # Build TRT engine
                if [ -f "$QWEN3B_CKPT/config.json" ]; then
                    ensure_swap_for_step 14
                    if require_memory_gb 10 "Qwen 3B TRT engine build"; then
                        echo "  Building Qwen 3B TRT engine (~5 min)..."
                        mkdir -p "$QWEN3B_ENGINE"
                        [ -d model.cache ] && rm -rf model.cache
                        python3 -m tensorrt_llm.commands.build \
                            --checkpoint_dir "$QWEN3B_CKPT" \
                            --output_dir "$QWEN3B_ENGINE" \
                            --gemm_plugin float16 \
                            --gpt_attention_plugin float16 \
                            --max_batch_size 1 \
                            --max_input_len 2048 \
                            --max_seq_len 4096 \
                            --max_num_tokens 4096 2>&1 || record_error "Qwen 3B TRT engine build failed"
                        ok "Qwen 3B TRT engine built"
                    fi
                fi
            fi
        fi
    fi

    # ── Gemma 4 E4B (multimodal: audio + vision input) ────────────────────
    # NOTE: TRT-LLM 1.2.0 does not support the Gemma 4 architecture
    # (mixed sliding/full attention, KV sharing, per-layer-type rope).
    # Gemma 4 runs via HuggingFace transformers (is_engine=False in registry).
    local GEMMA_DIR="$OE_MODELS_DIR/gemma-4-e4b"

    if [ -d "$GEMMA_DIR" ] && [ -n "$(ls -A "$GEMMA_DIR" 2>/dev/null)" ]; then
        skip "Gemma 4 E4B model"
    else
        echo "  Downloading Gemma 4 E4B (~6 GB)..."
        hf download google/gemma-4-e4b --local-dir "$GEMMA_DIR" \
            || record_error "Gemma 4 E4B download failed"
    fi

    # ── Llama 3.1 8B (alternative architecture for benchmark) ───────────
    local LLAMA_AWQ="$OE_MODELS_DIR/llama-3.1-8b-instruct-awq"
    local LLAMA_ENGINE="$OE_ENGINES_DIR/llama-3.1-8b-awq"

    if [ -d "$LLAMA_ENGINE" ] && [ -n "$(ls -A "$LLAMA_ENGINE" 2>/dev/null)" ]; then
        skip "Llama 3.1 8B TRT engine"
    else
        if [ "$IS_BLACKWELL" -eq 1 ]; then
            # ── Blackwell: NVFP4 from FP16 base (AWQ plugin crashes on sm_120) ──
            local LLAMA_FP16="$OE_MODELS_DIR/llama-3.1-8b-instruct"
            local LLAMA_CKPT="$OE_MODELS_DIR/trt_ckpt/llama-3.1-8b-nvfp4"

            # Download FP16 base (~16 GB) — try ungated mirror first
            if [ -d "$LLAMA_FP16" ] && [ -n "$(ls -A "$LLAMA_FP16" 2>/dev/null)" ]; then
                skip "Llama 3.1 8B FP16 base model"
            else
                echo "  Downloading Llama 3.1 8B FP16 (~16 GB)..."
                hf download unsloth/Meta-Llama-3.1-8B-Instruct --local-dir "$LLAMA_FP16" \
                    || hf download meta-llama/Llama-3.1-8B-Instruct --local-dir "$LLAMA_FP16" \
                    || record_error "Llama 3.1 8B FP16 download failed (may need HF login for gated model)"
            fi

            # NVFP4 quantization checkpoint
            if [ -d "$LLAMA_FP16" ] && [ -n "$(ls -A "$LLAMA_FP16" 2>/dev/null)" ]; then
                if [ ! -f "$LLAMA_CKPT/config.json" ]; then
                    local TRTLLM_QUANTIZE
                    TRTLLM_QUANTIZE="$(find "$HOME" -path "*/TensorRT-LLM/examples/quantization/quantize.py" 2>/dev/null | head -1)"
                    if [ -z "$TRTLLM_QUANTIZE" ]; then
                        record_error "TRT-LLM quantize.py not found — clone TensorRT-LLM repo"
                    else
                        mkdir -p "$LLAMA_CKPT"
                        # Llama 8B FP16 (~16 GB) exceeds 12 GB VRAM — quantize on CPU
                        # to avoid modelopt mixed-device (cuda:0 + cpu) tensor error
                        python3 "$TRTLLM_QUANTIZE" \
                            --model_dir "$LLAMA_FP16" \
                            --output_dir "$LLAMA_CKPT" \
                            --dtype float16 \
                            --qformat nvfp4 \
                            --calib_size 512 \
                            --device cpu \
                            --device_map cpu \
                            --tp_size 1 2>&1 || {
                            record_error "Llama 3.1 8B nvfp4 quantization failed"
                            [ -n "$LLAMA_CKPT" ] && rm -rf "$LLAMA_CKPT"
                        }
                    fi
                fi
            fi

            # Build engine
            if [ -f "$LLAMA_CKPT/config.json" ]; then
                ensure_swap_for_step 18
                if require_memory_gb 12 "Llama 3.1 8B TRT engine build"; then
                    echo "  Building Llama 3.1 8B nvfp4 TRT engine (~10 min)..."
                    mkdir -p "$LLAMA_ENGINE"
                    [ -d model.cache ] && rm -rf model.cache
                    python3 -m tensorrt_llm.commands.build \
                        --checkpoint_dir "$LLAMA_CKPT" \
                        --output_dir "$LLAMA_ENGINE" \
                        --gemm_plugin auto \
                        --gpt_attention_plugin auto \
                        --max_batch_size 1 \
                        --max_input_len 2048 \
                        --max_seq_len 4096 \
                        --max_num_tokens 4096 2>&1 || record_error "Llama 3.1 8B TRT engine build failed"
                    ok "Llama 3.1 8B TRT engine built"
                fi
            fi
        else
            # ── Non-Blackwell: INT4-AWQ ──
            local LLAMA_CKPT="$OE_MODELS_DIR/trt_ckpt/llama-3.1-8b-awq"

            # Download pre-quantized AWQ weights (~4.5 GB)
            if [ -d "$LLAMA_AWQ" ] && [ -n "$(ls -A "$LLAMA_AWQ" 2>/dev/null)" ]; then
                skip "Llama 3.1 8B AWQ model"
            else
                echo "  Downloading Llama 3.1 8B Instruct AWQ (~4.5 GB)..."
                hf download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --local-dir "$LLAMA_AWQ" \
                    || record_error "Llama 3.1 8B AWQ download failed"
            fi

            # Convert to TRT-LLM checkpoint
            if [ -d "$LLAMA_AWQ" ] && [ -n "$(ls -A "$LLAMA_AWQ" 2>/dev/null)" ]; then
                if [ ! -f "$LLAMA_CKPT/config.json" ]; then
                    ensure_swap_for_step 16
                    if require_memory_gb 12 "Llama 3.1 8B checkpoint conversion"; then
                        echo "  Converting Llama 3.1 8B to TRT-LLM checkpoint..."
                        mkdir -p "$LLAMA_CKPT"
                        python3 -c "
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
qc = QuantConfig(quant_algo='W4A16_AWQ')
model = LLaMAForCausalLM.from_hugging_face(
    '$LLAMA_AWQ', dtype='float16', quant_config=qc, use_autoawq=True)
model.save_checkpoint('$LLAMA_CKPT')
print('Checkpoint saved: $LLAMA_CKPT')
" 2>&1 || {
                            record_error "Llama 3.1 8B checkpoint conversion failed"
                            [ -n "$LLAMA_CKPT" ] && rm -rf "$LLAMA_CKPT"
                        }
                    fi
                else
                    skip "Llama 3.1 8B TRT-LLM checkpoint"
                fi

                # Build TRT engine
                if [ -f "$LLAMA_CKPT/config.json" ]; then
                    ensure_swap_for_step 18
                    if require_memory_gb 12 "Llama 3.1 8B TRT engine build"; then
                        echo "  Building Llama 3.1 8B TRT engine (~10 min)..."
                        mkdir -p "$LLAMA_ENGINE"
                        [ -d model.cache ] && rm -rf model.cache
                        python3 -m tensorrt_llm.commands.build \
                            --checkpoint_dir "$LLAMA_CKPT" \
                            --output_dir "$LLAMA_ENGINE" \
                            --gemm_plugin float16 \
                            --gpt_attention_plugin float16 \
                            --max_batch_size 1 \
                            --max_input_len 2048 \
                            --max_seq_len 4096 \
                            --max_num_tokens 4096 2>&1 || record_error "Llama 3.1 8B TRT engine build failed"
                        ok "Llama 3.1 8B TRT engine built"
                    fi
                fi
            fi
        fi
    fi
}
