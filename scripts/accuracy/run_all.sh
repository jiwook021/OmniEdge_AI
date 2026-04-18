#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI -- Model Accuracy Benchmark Runner
#
# Runs all accuracy evaluation scripts and prints a consolidated report.
# Each script exits 0 (pass), 1 (fail), or 77 (skip).
#
# Usage:
#   ./scripts/accuracy/run_all.sh                  # run all benchmarks
#   ./scripts/accuracy/run_all.sh --module llm     # run only LLM
#   ./scripts/accuracy/run_all.sh --json           # machine-readable JSON summary
#
# Environment:
#   OE_MODELS_DIR   -- root of downloaded models (default: ~/omniedge_models)
#   OE_ENGINES_DIR  -- root of TensorRT engines (default: $OE_MODELS_DIR/trt_engines)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=../integration/common.sh
source "$SCRIPT_DIR/../integration/common.sh"

# Consolidated log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$LOG_DIR/accuracy_report_${TIMESTAMP}.log"

# Parse args
MODULE_FILTER=""
JSON_OUTPUT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --module) shift; MODULE_FILTER="${1:-}" ;;
        --json)   JSON_OUTPUT=1 ;;
        *)        MODULE_FILTER="$1" ;;
    esac
    shift
done

declare -a RESULTS=()

run_eval() {
    local module_name="$1"
    local script_path="$2"
    local description="$3"

    # Filter
    if [[ -n "$MODULE_FILTER" ]] && [[ "$module_name" != *"$MODULE_FILTER"* ]]; then
        return
    fi

    echo "" | tee -a "$REPORT_FILE"
    echo -e "${BOLD}-- $description --${NC}" | tee -a "$REPORT_FILE"

    if [[ ! -f "$script_path" ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  Script not found: $script_path" | tee -a "$REPORT_FILE"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        RESULTS+=("{\"module\":\"$module_name\",\"status\":\"skip\",\"reason\":\"script not found\"}")
        return
    fi

    local start_time end_time elapsed_seconds
    start_time=$(date +%s)

    local output rc=0
    output=$(python3 "$script_path" 2>&1) || rc=$?

    end_time=$(date +%s)
    elapsed_seconds=$((end_time - start_time))

    echo "$output" | tee -a "$REPORT_FILE"

    case "$rc" in
        0)
            TOTAL_PASS=$((TOTAL_PASS + 1))
            echo -e "  ${GREEN}PASS${NC}  $module_name (${elapsed_seconds}s)" | tee -a "$REPORT_FILE"
            RESULTS+=("{\"module\":\"$module_name\",\"status\":\"pass\",\"elapsed_s\":$elapsed_seconds}")
            ;;
        77)
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            echo -e "  ${YELLOW}SKIP${NC}  $module_name (deps unavailable)" | tee -a "$REPORT_FILE"
            RESULTS+=("{\"module\":\"$module_name\",\"status\":\"skip\",\"reason\":\"deps unavailable\"}")
            ;;
        *)
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            echo -e "  ${RED}FAIL${NC}  $module_name (exit $rc, ${elapsed_seconds}s)" | tee -a "$REPORT_FILE"
            RESULTS+=("{\"module\":\"$module_name\",\"status\":\"fail\",\"exit_code\":$rc,\"elapsed_s\":$elapsed_seconds}")
            ;;
    esac
}

# =============================================================================
echo -e "${BOLD}OmniEdge_AI -- Model Accuracy Benchmarks${NC}" | tee "$REPORT_FILE"
echo "Timestamp: $(date -Iseconds)" | tee -a "$REPORT_FILE"
echo "Models:    $OE_MODELS_DIR" | tee -a "$REPORT_FILE"
echo "Engines:   $OE_ENGINES_DIR" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

run_eval "stt"       "$SCRIPT_DIR/eval_stt_wer.py"          "STT Word Error Rate (Whisper V3 Turbo)"
run_eval "vlm"       "$SCRIPT_DIR/eval_vlm_captioning.py"   "VLM Captioning Quality (Moondream2)"
run_eval "tts"       "$SCRIPT_DIR/eval_tts_quality.py"      "TTS Synthesis Quality (Kokoro ONNX)"
run_eval "vad"       "$SCRIPT_DIR/eval_vad_f1.py"           "VAD F1 Score (Silero VAD)"
run_eval "face"      "$SCRIPT_DIR/eval_face_recognition.py" "Face Recognition (SCRFD + AuraFace v1)"
run_eval "sam2"      "$SCRIPT_DIR/eval_sam2_segmentation.py" "SAM2 Interactive Segmentation (Segment Anything 2)"

# =============================================================================
# Summary
# =============================================================================
print_summary | tee -a "$REPORT_FILE"
echo "Report saved: $REPORT_FILE"

# -- Machine-readable JSON summary ------------------------------------------
# Printed when --json is passed.  Format (single line):
#   {
#     "pass":  <int>,          // number of benchmarks that passed
#     "fail":  <int>,          // number that failed (exit != 0 and != 77)
#     "skip":  <int>,          // number skipped (exit 77 or script missing)
#     "results": [             // per-module detail objects
#       { "module": "<name>", "status": "pass|fail|skip",
#         "elapsed_s": <int>,  // wall-clock seconds (omitted on skip)
#         "exit_code": <int>,  // present only on fail
#         "reason": "<text>"   // present only on skip
#       }, ...
#     ]
#   }
if [[ "$JSON_OUTPUT" -eq 1 ]]; then
    echo ""
    echo "{\"pass\":$TOTAL_PASS,\"fail\":$TOTAL_FAIL,\"skip\":$TOTAL_SKIP,\"results\":[$(IFS=,; echo "${RESULTS[*]}")]}"
fi

if [[ "$TOTAL_FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
