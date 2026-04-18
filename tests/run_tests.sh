#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Test Runner
#
# Discovers and runs all GTest binaries registered in the test registry below.
# Each entry maps a binary path, suite tag, and human-readable label.
#
# Usage:
#   ./tests/run_tests.sh                    # run all built tests
#   ./tests/run_tests.sh --build            # rebuild tests first, then run
#   ./tests/run_tests.sh --filter stt       # run only tests matching "stt"
#   ./tests/run_tests.sh --ctest            # delegate to ctest (xml output)
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"

# Source shared utilities for colours and build_ld_library_path
source "$REPO_DIR/scripts/integration/common.sh"

export LD_LIBRARY_PATH="$(build_ld_library_path)"

# -- Argument parsing -------------------------------------------------------
DO_BUILD=0
USE_CTEST=0
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)  DO_BUILD=1 ;;
        --ctest)  USE_CTEST=1 ;;
        --filter) shift; FILTER="${1:-}" ;;
        *)        FILTER="$1" ;;
    esac
    shift
done

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
    ctest --output-on-failure --timeout 120 "${FILTER_ARGS[@]}"
    exit $?
fi

# -- Test registry ---------------------------------------------------------
# Each entry is a pipe-delimited triple:
#   "relative/path/to/binary | SUITE_TAG | Human-readable label"
#
# SUITE_TAG is used for grouping in output:
#   core  — pure algorithm / unit tests (no GPU, no I/O)
#   iface — interface contract tests (API shape, mock backends)
#   node  — node wiring tests (ZMQ pub/sub, SHM plumbing)
#   integ — integration tests (real GPU, real SHM, real network)
#
# Binaries that don't exist are SKIPped (not built or EXCLUDE_FROM_ALL).
declare -a TESTS=(
    # ── core (pure algorithms, no GPU) ────────────────────────────────────
    "tests/core/common/test_oe_logger         | core    | Logger singleton"
    "tests/core/common/test_ui_action         | core    | UI action enum parse/name"
    "tests/core/stt/test_mel_spectrogram      | core    | Mel spectrogram"
    "tests/core/stt/test_hallucination_filter | core    | Hallucination filter"
    "tests/core/tts/test_sentence_splitter    | core    | Sentence splitter"
    "tests/core/cv/test_face_gallery_sqlite   | core    | Face gallery SQLite CRUD"
    "tests/core/statemachine/test_state_machine        | core | State machine transitions"
    "tests/core/statemachine/test_prompt_assembler     | core | Prompt assembly"
    "tests/core/statemachine/test_session_persistence  | core | Session state persistence"

    # ── interfaces (contract / mock tests) ────────────────────────────────
    "tests/interfaces/cv/test_blur_api             | iface   | BlurInferencer contract"
    "tests/interfaces/cv/test_blur_isp_pipeline    | iface   | ISP parameter forwarding"
    "tests/interfaces/cv/test_face_detection_api   | iface   | FaceRecogInferencer contract"
    "tests/interfaces/cv/test_face_matching        | iface   | Face matching contract"
    "tests/interfaces/stt/test_stt_empty_output    | iface   | STT empty output suppression"
    "tests/interfaces/tts/test_tts_degraded_inferencer | iface  | TTS degraded mode"
    "tests/interfaces/ws_bridge/test_ws_command_validation | iface | WS command parsing"

    # ── nodes (ZMQ / SHM wiring) ─────────────────────────────────────────
    "tests/nodes/stt/test_stt_node                     | node    | STT node wiring"
    "tests/nodes/tts/test_tts_node                     | node    | TTS node wiring"
    "tests/nodes/conversation/test_conversation_node   | node    | Conversation (Gemma-4) node wiring"
    "tests/nodes/conversation/test_conversation_params | node    | Conversation variant parsing"
    "tests/nodes/conversation/test_conversation_benchmark | node | Conversation benchmark smoke"
    "tests/nodes/cv/test_face_recog_integration        | node    | Face recog node"
    "tests/nodes/cv/test_video_denoise_node            | node    | Video denoise node"
    "tests/nodes/ingest/test_video_ingest_node         | node    | Video ingest node"
    "tests/nodes/ingest/test_audio_ingest_node         | node    | Audio ingest node"
    "tests/nodes/audio_denoise/test_audio_denoise_node | node    | Audio denoise node"
    "tests/nodes/ws_bridge/test_websocket_bridge_node  | node    | WS bridge node"
    "tests/nodes/orchestrator/test_module_launcher     | node    | Module launcher"
    "tests/nodes/orchestrator/test_gpu_profiler        | node    | GPU profiler"
    "tests/nodes/orchestrator/test_omniedge_orchestrator | node  | Orchestrator lifecycle"

    # ── integration (real GPU, real SHM, real sockets) ────────────────────
    "tests/integration/transport/test_shm_mapping          | integ | POSIX shared memory"
    "tests/integration/transport/test_shm_double_buffer    | integ | SHM double buffer"
    "tests/integration/transport/test_shm_jpeg_transport   | integ | SHM JPEG round-trip"
    "tests/integration/transport/test_zmq_publisher        | integ | ZMQ publisher"
    "tests/integration/transport/test_zmq_subscriber       | integ | ZMQ subscriber"
    "tests/integration/transport/test_message_router       | integ | MessageRouter pub/sub"
    "tests/integration/hal/test_gpu_tier_probe             | integ | GPU tier detection"
    "tests/integration/hal/test_vram_tracker               | integ | VRAM budget tracking"
    "tests/integration/hal/test_vram_gate                  | integ | VRAM acquisition gate"
    "tests/integration/hal/test_pinned_staging_buffer      | integ | Pinned staging buffer"
    "tests/integration/pipeline/test_stt_llm_tts_pipeline  | integ | STT→LLM→TTS pipeline"
)

declare -a SKIP_TESTS=()

# -- Timeouts ---------------------------------------------------------------
TIMEOUT_QUICK=30    # seconds — fast unit/contract tests
TIMEOUT_MODEL=180   # seconds — tests that load GPU models

# Description: Select timeout based on binary name
# Args: $1 - binary path
# Returns: prints timeout in seconds
timeout_for() {
    local bin="$1"
    case "$bin" in
        *whisper*|*kokoro*|*pipeline*) echo $TIMEOUT_MODEL ;;
        *) echo $TIMEOUT_QUICK ;;
    esac
}

# -- Counters and state -----------------------------------------------------
PASS=0; FAIL=0; SKIP=0; TOTAL=0
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
# Description: Parse a registry entry and run the GTest binary
# Args: $1 - pipe-delimited entry "binary_path | SUITE | label"
run_one() {
    local entry="$1"
    local relbin suite label
    relbin="$(  echo "$entry" | cut -d'|' -f1 | xargs)"
    suite="$(   echo "$entry" | cut -d'|' -f2 | xargs)"
    label="$(   echo "$entry" | cut -d'|' -f3 | xargs)"
    local binary="$BUILD_DIR/$relbin"

    # Apply filter if set
    if [[ -n "$FILTER" ]] && ! echo "$relbin $suite $label" | grep -qi "$FILTER"; then
        return
    fi

    TOTAL=$(( TOTAL + 1 ))

    # Skip if binary not built
    if [[ ! -f "$binary" ]]; then
        SKIP=$(( SKIP + 1 ))
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
        PASS=$(( PASS + 1 ))
        printf "  ${GREEN}PASS${NC}  [%-8s] %s\n" "$suite" "$label"
    elif [[ $status -eq 124 ]]; then
        FAIL=$(( FAIL + 1 ))
        FAILED_LIST+=("$label  (TIMEOUT after ${timeout_sec}s)")
        printf "  ${RED}FAIL${NC}  [%-8s] %s  ${RED}(TIMEOUT)${NC}\n" "$suite" "$label"
        LOG_LINES+=("── $label ──")
        LOG_LINES+=("$(tail -20 "$tmpout")")
    else
        FAIL=$(( FAIL + 1 ))
        FAILED_LIST+=("$label  (exit $status)")
        printf "  ${RED}FAIL${NC}  [%-8s] %s  ${RED}(exit $status)${NC}\n" "$suite" "$label"
        LOG_LINES+=("── $label ──")
        LOG_LINES+=("$(tail -30 "$tmpout")")
    fi
    rm -f "$tmpout"
}

# -- Main -------------------------------------------------------------------
print_header
echo -e "${BOLD}Running tests  (build=$BUILD_DIR)${NC}"
[[ -n "$FILTER" ]] && echo -e "${YELLOW}Filter: $FILTER${NC}"
echo ""

for entry in "${TESTS[@]}"; do
    run_one "$entry"
done

# -- Failure details --------------------------------------------------------
if [[ ${#LOG_LINES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}${BOLD}════ Failure output ════${NC}"
    for line in "${LOG_LINES[@]}"; do
        echo "$line"
    done
fi

# -- Summary ----------------------------------------------------------------
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
