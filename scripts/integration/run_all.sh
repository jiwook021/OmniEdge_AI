#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI -- Integration Test Runner
#
# Runs model-level integration tests for all modules:
#   ISP  -- GPU compositor pixel tests (via ctest -L gpu)
#   LLM  -- TRT-LLM engine inference (question -> answer)
#   TTS  -- Kokoro ONNX synthesis (text -> PCM validation)
#   VRAM -- Wait for GPU memory release between test suites
#   Py   -- Python integration tests (LLM, STT, TTS)
#
# Exit codes: 0 = all passed, 1 = at least one failure
# Tests return 77 when skipped (missing model/deps).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

# Fix MPI TCP hang in containers: force loopback-only communication
export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-lo}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Source model registry for discovery helpers
source "$SCRIPT_DIR/model_registry.sh"

# Runtime library paths — centralised in common.sh
export LD_LIBRARY_PATH="${OE_RUNTIME_LD_LIBRARY_PATH:-$(build_ld_library_path)}"

# =============================================================================
echo -e "${BOLD}OmniEdge_AI -- Integration Tests${NC}"
echo ""

# -- ISP GPU Pixel Tests ----------------------------------------------------
# Description: Run ctest GPU-labeled tests (compositor, ISP pipeline)
# Excludes model-loading tests (LLM/TTS) which are run separately below.
run_isp_tests() {
    echo -e "${BOLD}-- ISP GPU Pixel Tests --${NC}"

    if [[ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  Build directory not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    local gpu_test_count
    gpu_test_count=$(cd "$BUILD_DIR" && ctest -N -L gpu 2>/dev/null | grep -c "^  Test " || echo "0")

    if [[ "$gpu_test_count" -eq 0 ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  No GPU tests registered"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    log_info "Running $gpu_test_count GPU-labeled tests..."

    # Exclude model-loading inferencer tests (run in dedicated sections below)
    local output
    output=$(cd "$BUILD_DIR" && ctest -L gpu \
        --exclude-regex "PipelineModelTests|TrtLlmInferencer|OnnxKokoroInferencer" \
        --output-on-failure -j"$(nproc)" 2>&1) || true

    parse_ctest_summary "$output"

    if [[ "${_CTEST_FAILED}" -eq 0 ]] && [[ "${_CTEST_RAN}" -gt 0 ]]; then
        echo -e "  ${GREEN}PASS${NC}  All $_CTEST_RAN GPU tests passed"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    elif [[ "${_CTEST_RAN}" -eq 0 ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  No GPU tests were executed"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
    else
        echo -e "  ${RED}FAIL${NC}  $_CTEST_FAILED / $_CTEST_RAN GPU tests failed"
        echo "$output" | grep -E '^\s*\d+/\d+ Test .*(Failed|Not Run)' | sed 's/^/    /' | head -20 || true
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# -- LLM C++ GPU Tests ------------------------------------------------------
# Description: Run TRT-LLM inferencer C++ tests via ctest
# Checks engine-TRT version compatibility first. If incompatible,
# runs CPU-only tests (factory, API contracts) to avoid executor hangs.
run_llm_tests() {
    echo ""
    echo -e "${BOLD}-- LLM C++ GPU Tests --${NC}"

    if [[ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  Build directory not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    # Discover engine and tokenizer using model registry helpers
    local engine_dir tokenizer_dir
    engine_dir=$(get_llm_engine_dir) || true
    tokenizer_dir=$(get_llm_tokenizer_dir) || true

    if [[ -z "$engine_dir" ]] || [[ -z "$tokenizer_dir" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  LLM engine or tokenizer not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    log_info "Engine:    $engine_dir"
    log_info "Tokenizer: $tokenizer_dir"

    # Build test binary if not built (EXCLUDE_FROM_ALL target)
    if ! check_binary "$BUILD_DIR/tests/integration/inferencers/test_trtllm_inferencer"; then
        log_info "Building test_trtllm_inferencer (EXCLUDE_FROM_ALL)..."
        cmake --build "$BUILD_DIR" --target test_trtllm_inferencer -j"$(nproc)" 2>/dev/null || true
    fi

    # Check engine-TRT version compatibility before attempting to load.
    # Loading an incompatible engine hangs the TRT-LLM Executor.
    local engine_compat=1
    local compat_result
    if [[ -f "$PROJECT_ROOT/scripts/helpers/check_engine_compat.py" ]]; then
        compat_result=$(python3 "$PROJECT_ROOT/scripts/helpers/check_engine_compat.py" \
            --engine-dir "$engine_dir" 2>/dev/null) || true
        if [[ "$compat_result" == incompatible:* ]]; then
            engine_compat=0
            log_warn "Engine/TRT version mismatch: $compat_result"
        fi
    else
        # Fallback: inline version check
        local trt_ver
        trt_ver=$(python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || echo "unknown")
        if [[ -f "$engine_dir/config.json" ]] && [[ "$trt_ver" != "unknown" ]]; then
            local engine_trt_ver
            engine_trt_ver=$(python3 -c "
import json
cfg = json.load(open('$engine_dir/config.json'))
pp = cfg.get('pretrained_config', {})
print(pp.get('trt_version', ''))
" 2>/dev/null || echo "")
            if [[ -n "$engine_trt_ver" ]] && [[ "$engine_trt_ver" != "$trt_ver" ]]; then
                engine_compat=0
                log_warn "Engine built with TRT $engine_trt_ver, installed TRT $trt_ver"
            fi
        fi
    fi

    if [[ "$engine_compat" -eq 1 ]]; then
        # Compatible — run full GPU tests with engine loaded
        export OE_TEST_LLM_ENGINE_DIR="$engine_dir"
        export OE_TEST_LLM_TOKENIZER_DIR="$tokenizer_dir"
    else
        # Incompatible — run CPU-only tests to avoid hanging the executor
        export OE_TEST_LLM_TOKENIZER_DIR="$tokenizer_dir"
        log_info "Running CPU-only LLM tests (engine incompatible)"
    fi

    # Run LLM tests (300s timeout — 5+ GB engine takes ~60s to load)
    local output
    output=$(cd "$BUILD_DIR" && timeout 300 ctest -R TrtLlmInferencer \
        --output-on-failure --timeout 240 2>&1) || true

    parse_ctest_summary "$output"

    # Classify results
    if echo "$output" | grep -q "Unable to find executable"; then
        echo -e "  ${YELLOW}SKIP${NC}  LLM C++ test binary not available after build"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
    elif [[ "${_CTEST_RAN}" -eq 0 ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  No LLM C++ tests matched (binary not built)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
    elif [[ "${_CTEST_FAILED}" -eq 0 ]]; then
        if [[ "$engine_compat" -eq 1 ]]; then
            echo -e "  ${GREEN}PASS${NC}  LLM C++ tests passed ($_CTEST_RAN ran)"
        else
            echo -e "  ${GREEN}PASS${NC}  LLM C++ CPU tests passed ($_CTEST_RAN ran, engine rebuild needed for GPU tests)"
        fi
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  LLM C++ tests: $_CTEST_FAILED / $_CTEST_RAN failed"
        echo "$output" | grep -E 'Failed|Not Run|error|ERROR' \
            | head -10 | sed 's/^/    /' || true
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# -- TTS C++ GPU Tests ------------------------------------------------------
# Description: Run Kokoro ONNX TTS inferencer tests
# Discovers model and voice paths using model registry helpers.
run_tts_tests() {
    echo ""
    echo -e "${BOLD}-- TTS C++ GPU Tests --${NC}"

    if [[ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  Build directory not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    # Discover model and voices using registry helpers
    local onnx_model voice_dir
    onnx_model=$(get_kokoro_onnx_model) || true
    voice_dir=$(get_kokoro_voice_dir) || true

    if [[ -z "$onnx_model" ]] || [[ -z "$voice_dir" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  Kokoro model or voices not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    export OMNIEDGE_ONNX_MODEL="$onnx_model"
    export OMNIEDGE_VOICE_DIR="$voice_dir"

    local tts_bin="$BUILD_DIR/tests/integration/inferencers/test_onnx_kokoro_inferencer"
    if ! check_binary "$tts_bin"; then
        log_info "Building test_onnx_kokoro_inferencer (EXCLUDE_FROM_ALL)..."
        cmake --build "$BUILD_DIR" --target test_onnx_kokoro_inferencer -j"$(nproc)" 2>/dev/null || true
    fi

    if ! check_binary "$tts_bin"; then
        echo -e "  ${YELLOW}SKIP${NC}  test_onnx_kokoro_inferencer binary not available"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return
    fi

    local output
    output=$("$tts_bin" 2>&1) || true

    if echo "$output" | grep -q "PASSED"; then
        echo -e "  ${GREEN}PASS${NC}  TTS C++ GPU tests passed"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  TTS C++ GPU tests"
        echo "$output" | grep -E "FAIL|Error" | head -5 | sed 's/^/    /' || true
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# -- VRAM drain -------------------------------------------------------------
# Description: Wait for GPU memory to drain after C++ tests
# The C++ LLM test loads a ~5.6 GB TRT-LLM engine. Even after the ctest
# process exits, CUDA memory deallocation may take a few seconds. Without
# this wait, the Python LLM test hits CUDA OOM on a 12 GB GPU.
drain_vram() {
    echo ""
    echo -e "${BOLD}-- VRAM Drain --${NC}"

    local wait_elapsed=0
    while [[ "$wait_elapsed" -lt 30 ]]; do
        local vram_used
        vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' ')

        if [[ "${vram_used:-9999}" -lt 500 ]]; then
            log_info "VRAM clear: ${vram_used} MiB used"
            break
        fi
        if [[ "$wait_elapsed" -eq 0 ]]; then
            log_info "Waiting for VRAM to drain (${vram_used} MiB used)..."
        fi
        sleep 1
        wait_elapsed=$((wait_elapsed + 1))
    done

    if [[ "$wait_elapsed" -ge 30 ]]; then
        local vram_used
        vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
            | head -1 | tr -d ' ' || echo "?")
        log_warn "VRAM still ${vram_used} MiB after 30s — Python tests may OOM"
    fi

    # Kill orphan TRT-LLM MPI workers left by C++ tests — they hold GPU
    # contexts that prevent the Python LLM test from allocating KV cache.
    pkill -f "tensorrt_llm.*mpi_worker" 2>/dev/null || true
    pkill -f "trtllm.*worker" 2>/dev/null || true
    sleep 1
}

# -- Python integration tests -----------------------------------------------
# Description: Run Python-based integration tests for LLM, STT, TTS
run_python_tests() {
    run_test "LLM Python Integration"  python3 "$SCRIPT_DIR/test_llm.py"
    run_test "STT Python Integration"  python3 "$SCRIPT_DIR/test_stt.py"
    run_test "TTS Python Integration"  python3 "$SCRIPT_DIR/test_tts.py"
}

# =============================================================================
#  Main execution
# =============================================================================
run_isp_tests
run_llm_tests
run_tts_tests
drain_vram
run_python_tests

# =============================================================================
#  Summary
# =============================================================================
print_summary

if [[ "$TOTAL_FAIL" -gt 0 ]]; then
    log_error "Integration tests have failures."
    exit 1
fi
if [[ "$TOTAL_SKIP" -gt 0 ]]; then
    log_warn "Some tests skipped due to missing models/deps."
fi
echo ""
exit 0
