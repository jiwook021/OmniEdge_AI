#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Test Runner
#
# Discovers and runs all GTest binaries registered below.
#
# Usage:
#   ./tests/run_tests.sh                     # run all registered tests
#   ./tests/run_tests.sh --mode conversation # run conversation-mode suite
#   ./tests/run_tests.sh --mode security     # run security-mode suite
#   ./tests/run_tests.sh --mode beauty       # run beauty-mode suite
#   ./tests/run_tests.sh --build             # rebuild before run
#   ./tests/run_tests.sh --filter ws_bridge  # additional regex filter
#   ./tests/run_tests.sh --ctest             # delegate to ctest
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR_RAW="${OE_BUILD_DIR:-$REPO_DIR/build}"
if [[ "$BUILD_DIR_RAW" = /* ]]; then
    BUILD_DIR="$BUILD_DIR_RAW"
else
    BUILD_DIR="$REPO_DIR/$BUILD_DIR_RAW"
fi
BUILD_DIR_NOTE=""

cmake_home_from_cache() {
    local cache="$1/CMakeCache.txt"
    [[ -f "$cache" ]] || return 1
    grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$cache" | cut -d'=' -f2-
}

fallback_build_dir_if_needed() {
    local configured_home
    configured_home="$(cmake_home_from_cache "$BUILD_DIR" || true)"
    if [[ -z "$configured_home" || "$configured_home" == "$REPO_DIR" ]]; then
        return
    fi

    local alt_dir="$REPO_DIR/build-user"
    local alt_home
    alt_home="$(cmake_home_from_cache "$alt_dir" || true)"
    if [[ "$alt_home" == "$REPO_DIR" ]]; then
        BUILD_DIR="$alt_dir"
        BUILD_DIR_NOTE="default build cache points to '$configured_home'; using '$alt_dir' instead"
    fi
}

fallback_build_dir_if_needed

source "$REPO_DIR/scripts/integration/common.sh"
export LD_LIBRARY_PATH="$(build_ld_library_path)"

# -- Argument parsing -------------------------------------------------------
DO_BUILD=0
USE_CTEST=0
FILTER=""
MODE_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)  DO_BUILD=1 ;;
        --ctest)  USE_CTEST=1 ;;
        --filter)
            shift
            FILTER="${1:-}"
            ;;
        --mode)
            shift
            MODE_FILTER="${1:-}"
            ;;
        *)
            FILTER="$1"
            ;;
    esac
    shift
done

if [[ -n "$MODE_FILTER" ]]; then
    case "$MODE_FILTER" in
        conversation|security|beauty) ;;
        *)
            echo "Unsupported mode: $MODE_FILTER" >&2
            echo "Expected one of: conversation, security, beauty" >&2
            exit 1
            ;;
    esac
fi

# -- Build (optional) ------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
    echo -e "${BOLD}==> Building all tests ...${NC}"
    cmake --build "$BUILD_DIR" --parallel "$(nproc)" 2>&1
    echo ""
fi

# -- ctest delegation (optional) -------------------------------------------
if [[ $USE_CTEST -eq 1 ]]; then
    echo -e "${BOLD}==> Running via ctest ...${NC}"
    cd "$BUILD_DIR"
    FILTER_ARGS=()
    [[ -n "$FILTER" ]] && FILTER_ARGS=(-R "$FILTER")
    if [[ -n "$MODE_FILTER" ]]; then
        echo -e "${YELLOW}Note:${NC} --mode filter is only supported by registry runner; ctest run ignores --mode."
    fi
    ctest --output-on-failure --timeout 120 "${FILTER_ARGS[@]}"
    exit $?
fi

# -- Test registry ---------------------------------------------------------
# Each entry is a pipe-delimited record:
#   "relative/binary | SUITE_TAG | Label | mode_tags | requirement"
#
# mode_tags:
#   all
#   conversation
#   security
#   beauty
#   comma-separated combinations, e.g. conversation,beauty
#
# requirement:
#   cpu (default)
#   gpu
#   shm
declare -a TESTS=(
    # ── core (pure algorithms, no GPU) ────────────────────────────────────
    "tests/core/common/test_oe_logger                  | core  | Logger singleton                    | all"
    "tests/core/common/test_ui_action                  | core  | UI action enum parse/name           | all"
    "tests/core/stt/test_mel_spectrogram               | core  | Mel spectrogram                     | conversation"
    "tests/core/stt/test_hallucination_filter          | core  | Hallucination filter                | conversation"
    "tests/core/tts/test_sentence_splitter             | core  | Sentence splitter                   | conversation"
    "tests/core/cv/test_face_gallery_sqlite            | core  | Face gallery SQLite CRUD            | conversation,beauty"
    "tests/core/statemachine/test_state_machine        | core  | State machine transitions           | all"
    "tests/core/statemachine/test_prompt_assembler     | core  | Prompt assembly                     | all"
    "tests/core/statemachine/test_session_persistence  | core  | Session state persistence           | all"

    # ── interfaces (contract / mock tests) ────────────────────────────────
    "tests/interfaces/cv/test_blur_api                 | iface | BlurInferencer contract             | conversation,beauty"
    "tests/interfaces/cv/test_blur_isp_pipeline        | iface | ISP parameter forwarding            | conversation,beauty"
    "tests/interfaces/cv/test_face_detection_api       | iface | FaceRecogInferencer contract        | conversation,beauty"
    "tests/interfaces/cv/test_face_matching            | iface | Face matching contract              | conversation,beauty"
    "tests/interfaces/stt/test_stt_empty_output        | iface | STT empty output suppression        | conversation | gpu"
    "tests/interfaces/tts/test_tts_degraded_inferencer | iface | TTS degraded mode                   | conversation | gpu"
    "tests/interfaces/ws_bridge/test_ws_command_validation | iface | WS command parsing               | all"

    # ── nodes (ZMQ / SHM wiring) ──────────────────────────────────────────
    "tests/nodes/stt/test_stt_node                     | node  | STT node wiring                     | conversation | gpu"
    "tests/nodes/tts/test_tts_node                     | node  | TTS node wiring                     | conversation"
    "tests/nodes/conversation/test_conversation_node   | node  | Conversation node wiring            | conversation"
    "tests/nodes/conversation/test_conversation_params | node  | Conversation variant parsing        | conversation"
    "tests/nodes/conversation/test_conversation_benchmark | node | Conversation benchmark smoke      | conversation"
    "tests/nodes/conversation/test_e2e_ingest_conversation | node | Conversation ingest E2E         | conversation | shm"
    "tests/nodes/cv/test_face_recog_integration        | node  | Face recog node                     | conversation,beauty | gpu"
    "tests/nodes/cv/test_video_denoise_node            | node  | Video denoise node                  | conversation"
    "tests/nodes/cv/test_face_filter_integration       | node  | Face filter integration             | conversation,beauty | gpu"
    "tests/nodes/cv/test_blur_deadline_fallback        | node  | Blur deadline fallback              | conversation,beauty"
    "tests/nodes/cv/test_security_vlm_node             | node  | Security VLM node                   | security"
    "tests/nodes/ingest/test_video_ingest_node         | node  | Video ingest node                   | all"
    "tests/nodes/ingest/test_audio_ingest_node         | node  | Audio ingest node                   | all"
    "tests/nodes/ingest/test_snapshot_command          | node  | Snapshot command                    | all"
    "tests/nodes/audio_denoise/test_audio_denoise_node | node  | Audio denoise node                  | conversation"
    "tests/nodes/ws_bridge/test_websocket_bridge_node  | node  | WS bridge node                      | all"
    "tests/nodes/ws_bridge/test_ws_bridge_relay        | node  | WS bridge relay                     | all"
    "tests/nodes/ws_bridge/test_frame_pacing           | node  | WS frame pacing                     | all"
    "tests/nodes/orchestrator/test_module_launcher     | node  | Module launcher                     | all"
    "tests/nodes/orchestrator/test_gpu_profiler        | node  | GPU profiler                        | all"
    "tests/nodes/orchestrator/test_omniedge_orchestrator | node | Orchestrator lifecycle            | all"
    "tests/nodes/orchestrator/test_priority_scheduler  | node  | Priority scheduler                  | all"
    "tests/nodes/orchestrator/test_pipeline_graph      | node  | Pipeline graph                      | all"
    "tests/nodes/orchestrator/test_graph_builder       | node  | Graph builder                       | all"
    "tests/nodes/orchestrator/test_pipeline_graph_switch | node | Graph switch plan                  | all"
    "tests/nodes/orchestrator/test_profile_transitions | node  | Profile transitions                 | all"

    # ── integration (real GPU, real SHM, real sockets) ───────────────────
    "tests/integration/transport/test_shm_mapping      | integ | POSIX shared memory                 | all"
    "tests/integration/transport/test_shm_double_buffer| integ | SHM double buffer                   | all"
    "tests/integration/transport/test_shm_jpeg_transport | integ | SHM JPEG round-trip              | all"
    "tests/integration/transport/test_zmq_publisher    | integ | ZMQ publisher                       | all"
    "tests/integration/transport/test_zmq_subscriber   | integ | ZMQ subscriber                      | all"
    "tests/integration/transport/test_message_router   | integ | MessageRouter pub/sub               | all"
    "tests/integration/hal/test_gpu_tier_probe         | integ | GPU tier detection                  | all"
    "tests/integration/hal/test_vram_tracker           | integ | VRAM budget tracking                | all"
    "tests/integration/hal/test_vram_gate              | integ | VRAM acquisition gate               | all"
    "tests/integration/hal/test_pinned_staging_buffer  | integ | Pinned staging buffer               | all"
    "tests/integration/pipeline/test_video_pipeline_e2e    | integ | Video ingest pipeline E2E        | all | shm"
    "tests/integration/pipeline/test_audio_pipeline_e2e    | integ | Audio ingest pipeline E2E        | all | shm"
)

# -- Timeouts ---------------------------------------------------------------
TIMEOUT_QUICK=30
TIMEOUT_MODEL=180

GPU_RUNTIME_OK=1
GPU_RUNTIME_REASON="available"
SHM_RUNTIME_OK=1
SHM_RUNTIME_REASON="available"

timeout_for() {
    local bin="$1"
    case "$bin" in
        *whisper*|*kokoro*|*pipeline*|*conversation*) echo "$TIMEOUT_MODEL" ;;
        *) echo "$TIMEOUT_QUICK" ;;
    esac
}

detect_gpu_runtime() {
    # Escape hatch for environments that guarantee working CUDA runtime.
    if [[ "${OE_REQUIRE_GPU_TESTS:-0}" == "1" ]]; then
        GPU_RUNTIME_OK=1
        GPU_RUNTIME_REASON="forced by OE_REQUIRE_GPU_TESTS=1"
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        GPU_RUNTIME_OK=0
        GPU_RUNTIME_REASON="nvidia-smi not found"
        return
    fi

    local check_err
    check_err="$(mktemp)"
    if ! python3 - <<'PY' 2>"$check_err"
import ctypes
import sys

cuda = None
for lib_name in ("libcudart.so.12", "libcudart.so"):
    try:
        cuda = ctypes.CDLL(lib_name)
        break
    except OSError:
        pass

if cuda is None:
    sys.stderr.write("libcudart not found")
    sys.exit(1)

cuda.cudaFree.argtypes = [ctypes.c_void_p]
cuda.cudaFree.restype = ctypes.c_int
cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
cuda.cudaGetErrorString.restype = ctypes.c_char_p

rc = cuda.cudaFree(ctypes.c_void_p(0))
if rc != 0:
    msg = cuda.cudaGetErrorString(rc)
    if isinstance(msg, bytes):
        msg = msg.decode("utf-8", errors="ignore")
    sys.stderr.write(f"cuda runtime init failed: {msg}")
    sys.exit(1)
PY
    then
        GPU_RUNTIME_OK=0
        GPU_RUNTIME_REASON="$(tr '\n' ' ' < "$check_err" | sed 's/[[:space:]]\+/ /g' | sed 's/^ *//;s/ *$//')"
        [[ -z "$GPU_RUNTIME_REASON" ]] && GPU_RUNTIME_REASON="cuda runtime init failed"
    fi
    rm -f "$check_err"
}

detect_shm_runtime() {
    local blocked=()
    local path
    for path in /dev/shm/oe.vid.ingest /dev/shm/oe.aud.ingest; do
        if [[ -e "$path" && ! -w "$path" ]]; then
            blocked+=("$path")
        fi
    done

    if [[ "${#blocked[@]}" -gt 0 ]]; then
        SHM_RUNTIME_OK=0
        SHM_RUNTIME_REASON="existing SHM segments are not writable (${blocked[*]})"
        return
    fi

    local probe
    probe="/dev/shm/oe.run_tests_probe.$$.$RANDOM"
    if ! ( : > "$probe" ) 2>/dev/null; then
        SHM_RUNTIME_OK=0
        SHM_RUNTIME_REASON="cannot create files under /dev/shm"
        return
    fi
    rm -f "$probe" 2>/dev/null || true
}

mode_matches() {
    local tags="$1"
    if [[ -z "$MODE_FILTER" ]]; then
        return 0
    fi
    if [[ -z "$tags" || "$tags" == "all" ]]; then
        return 0
    fi

    local IFS=','
    read -r -a parts <<< "$tags"
    for p in "${parts[@]}"; do
        if [[ "$p" == "$MODE_FILTER" ]]; then
            return 0
        fi
    done
    return 1
}

# -- Counters and state -----------------------------------------------------
PASS=0
FAIL=0
SKIP=0
TOTAL=0
declare -a FAILED_LIST=()
declare -a LOG_LINES=()

# -- Header -----------------------------------------------------------------
print_header() {
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║              OmniEdge_AI  Test Suite                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# -- Run a single test entry ------------------------------------------------
run_one() {
    local entry="$1"
    local relbin suite label mode_tags requirement
    relbin="$(echo "$entry" | cut -d'|' -f1 | xargs)"
    suite="$(echo "$entry" | cut -d'|' -f2 | xargs)"
    label="$(echo "$entry" | cut -d'|' -f3 | xargs)"
    mode_tags="$(echo "$entry" | cut -d'|' -f4 | xargs)"
    requirement="$(echo "$entry" | cut -d'|' -f5 | xargs || true)"
    requirement="${requirement:-cpu}"
    local binary="$BUILD_DIR/$relbin"

    if ! mode_matches "$mode_tags"; then
        return
    fi

    if [[ -n "$FILTER" ]] && ! echo "$relbin $suite $label" | grep -Eqi "$FILTER"; then
        return
    fi

    TOTAL=$((TOTAL + 1))

    if [[ "$requirement" == "gpu" && "$GPU_RUNTIME_OK" -eq 0 ]]; then
        SKIP=$((SKIP + 1))
        printf "  ${YELLOW}SKIP${NC}  [%-8s] %s (GPU runtime unavailable)\n" "$suite" "$label"
        return
    fi
    if [[ "$requirement" == "shm" && "$SHM_RUNTIME_OK" -eq 0 ]]; then
        SKIP=$((SKIP + 1))
        printf "  ${YELLOW}SKIP${NC}  [%-8s] %s (SHM runtime unavailable)\n" "$suite" "$label"
        return
    fi

    if [[ ! -f "$binary" ]]; then
        SKIP=$((SKIP + 1))
        printf "  ${YELLOW}SKIP${NC}  [%-8s] %s (binary not found)\n" "$suite" "$label"
        return
    fi

    local timeout_sec
    timeout_sec="$(timeout_for "$relbin")"
    local tmpout
    tmpout="$(mktemp)"
    local status=0
    timeout "$timeout_sec" "$binary" --gtest_color=yes 2>&1 | tee "$tmpout" > /dev/null || status=$?

    if [[ $status -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf "  ${GREEN}PASS${NC}  [%-8s] %s\n" "$suite" "$label"
    elif [[ $status -eq 124 ]]; then
        FAIL=$((FAIL + 1))
        FAILED_LIST+=("$label (TIMEOUT after ${timeout_sec}s)")
        printf "  ${RED}FAIL${NC}  [%-8s] %s  ${RED}(TIMEOUT)${NC}\n" "$suite" "$label"
        LOG_LINES+=("── $label ──")
        LOG_LINES+=("$(tail -20 "$tmpout")")
    else
        FAIL=$((FAIL + 1))
        FAILED_LIST+=("$label (exit $status)")
        printf "  ${RED}FAIL${NC}  [%-8s] %s  ${RED}(exit $status)${NC}\n" "$suite" "$label"
        LOG_LINES+=("── $label ──")
        LOG_LINES+=("$(tail -30 "$tmpout")")
    fi
    rm -f "$tmpout"
}

# -- Main -------------------------------------------------------------------
detect_gpu_runtime
detect_shm_runtime

print_header
echo -e "${BOLD}Running tests  (build=$BUILD_DIR)${NC}"
[[ -n "$MODE_FILTER" ]] && echo -e "${YELLOW}Mode: $MODE_FILTER${NC}"
[[ -n "$FILTER" ]] && echo -e "${YELLOW}Filter: $FILTER${NC}"
[[ -n "$BUILD_DIR_NOTE" ]] && echo -e "${YELLOW}${BUILD_DIR_NOTE}${NC}"
if [[ "$GPU_RUNTIME_OK" -eq 0 ]]; then
    echo -e "${YELLOW}GPU runtime unavailable: ${GPU_RUNTIME_REASON}${NC}"
    echo -e "${YELLOW}GPU-tagged tests will be skipped (set OE_REQUIRE_GPU_TESTS=1 to force).${NC}"
fi
if [[ "$SHM_RUNTIME_OK" -eq 0 ]]; then
    echo -e "${YELLOW}SHM runtime unavailable: ${SHM_RUNTIME_REASON}${NC}"
    echo -e "${YELLOW}SHM-tagged tests will be skipped.${NC}"
fi
echo ""

for entry in "${TESTS[@]}"; do
    run_one "$entry"
done

if [[ ${#LOG_LINES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}${BOLD}════ Failure output ════${NC}"
    for line in "${LOG_LINES[@]}"; do
        echo "$line"
    done
fi

echo ""
echo -e "${BOLD}════════════════════════════════════════${NC}"
printf "${BOLD}Results:  ${GREEN}%d passed${NC}  ${RED}%d failed${NC}  ${YELLOW}%d skipped${NC}  (of %d run)\n" \
    "$PASS" "$FAIL" "$SKIP" "$TOTAL"
echo -e "${BOLD}════════════════════════════════════════${NC}"

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}${BOLD}Failed tests:${NC}"
    for f in "${FAILED_LIST[@]}"; do
        echo "  • $f"
    done
    exit 1
fi

exit 0
