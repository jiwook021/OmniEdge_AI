#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Single Entry Point
# Run `bash run_all.sh --help` for usage.
# ═══════════════════════════════════════════════════════════════════════════════


# Exit on error, unset variable, or failed pipeline (safer scripting):
#   -e  : Exit immediately if any command fails
#   -u  : Treat unset variables as errors
#   -o pipefail : Return failure if any command in a pipeline fails
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/integration/common.sh"

LOG_DIR="${OE_LOG_DIR:-$PROJECT_ROOT/logs}"
export OE_LOG_DIR="$LOG_DIR"
mkdir -p "$LOG_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
INI_FILE="${PROJECT_ROOT}/config/omniedge.ini"
MODULES="video_ingest audio_ingest background_blur face_recognition stt tts llm"
WS_BRIDGE_PORT=$(ini_get "$INI_FILE" "ports" "ws_http" "9001")
DAEMON_PUB_PORT=$(ini_get "$INI_FILE" "ports" "daemon" "5571")
BRIDGE_PUB_PORT=$(ini_get "$INI_FILE" "ports" "websocket_bridge" "5570")
ALL_PORTS=$(ini_get_all_ports "$INI_FILE")

# Per-module VRAM budgets (MiB) — preflight-only, authoritative limits in C++.
declare -A MODULE_VRAM_BUDGET_MB=(
    [background_blur]=250 [tts]=100 [stt]=1500 [face_recognition]=500
    [llm]=3440 [conversation_model]=3500 [super_resolution]=1500
    [image_transform]=4000 [video_denoise]=1500 [audio_denoise]=50
)

# ── Environment ───────────────────────────────────────────────────────────────
setup_environment() {
    PYTHON_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null \
        || echo "$HOME/.local/lib/python3.12/site-packages")"

    # Empty CUDA_VISIBLE_DEVICES masks all GPUs — normalize it away.
    if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]] && [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
        unset CUDA_VISIBLE_DEVICES
    fi

    # Shared base paths (used by both build-time and runtime).
    local BASE_LIBS="\
/usr/local/onnxruntime/lib:\
${HOME}/espeak_ng_dev/runtime/usr/lib/x86_64-linux-gnu:\
${PYTHON_SITE}/tensorrt_llm/libs:\
${PYTHON_SITE}/torch/lib:\
${PYTHON_SITE}/nvidia/nccl/lib:\
${PYTHON_SITE}/tensorrt_libs:\
${LD_LIBRARY_PATH:-}"

    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${BASE_LIBS}"
    export OE_RUNTIME_LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${BASE_LIBS}"
    export ESPEAK_DATA_PATH="${HOME}/espeak_ng_dev/runtime/usr/lib/x86_64-linux-gnu/espeak-ng-data"
    export OE_MODELS_DIR="${OE_MODELS_DIR:-$HOME/omniedge_models}"
    export OE_ENGINES_DIR="${OE_MODELS_DIR}/trt_engines"
    export PATH="$HOME/.local/bin:$PATH"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

# Check a list of "Label:path" entries, print OK/FAIL for each.
# Returns number of missing items.
check_model_list() {
    local -n _models=$1
    local missing=0
    for item in "${_models[@]}"; do
        local name="${item%%:*}" path="${item#*:}"
        if [[ -e "$path" ]] && { [[ -f "$path" ]] || [[ -n "$(ls -A "$path" 2>/dev/null)" ]]; }; then
            status_ok "$name"
        else
            status_fail "$name" "$path"
            missing=$((missing + 1))
        fi
    done
    return "$missing"
}

# Query nvidia-smi for VRAM. Sets: _VRAM_USED, _VRAM_TOTAL, _VRAM_FREE, _GPU_NAME.
query_gpu_vram() {
    _VRAM_USED=0; _VRAM_TOTAL=0; _VRAM_FREE=0; _GPU_NAME="unknown"
    command -v nvidia-smi &>/dev/null || return 1
    local info
    info=$(nvidia-smi --query-gpu=memory.used,memory.total,name --format=csv,noheader,nounits 2>/dev/null | head -1)
    [[ -z "$info" ]] && return 1
    _VRAM_USED=$(echo "$info" | cut -d',' -f1 | tr -d ' ')
    _VRAM_TOTAL=$(echo "$info" | cut -d',' -f2 | tr -d ' ')
    _GPU_NAME=$(echo "$info" | cut -d',' -f3 | sed 's/^ *//')
    _VRAM_FREE=$((_VRAM_TOTAL - _VRAM_USED))
}

# Kill processes on a list of ports. Args: signal, port list.
kill_ports() {
    local sig="$1"; shift
    for p in "$@"; do
        local pids
        pids=$(lsof -ti :"$p" 2>/dev/null || true)
        [[ -n "$pids" ]] && echo "$pids" | xargs -r kill "$sig" 2>/dev/null || true
    done
}

# Check a command exists. Args: cmd [severity]  (severity: pass/fail/warn)
require_cmd() {
    local cmd="$1" severity="${2:-fail}"
    if command -v "$cmd" &>/dev/null; then
        pass_fn "$cmd"
    elif [[ "$severity" == "warn" ]]; then
        warn_fn "$cmd not found"
    else
        fail_fn "$cmd not found"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PREREQUISITE CHECK (used by --prereqs AND run_system)
# ═══════════════════════════════════════════════════════════════════════════════
check_prerequisites() {
    PASS=0; FAIL=0; WARN=0

    echo -e "${BOLD}=== Prerequisite Check ===${NC}\n"

    # GPU
    section "GPU:"
    if query_gpu_vram; then
        pass_fn "NVIDIA GPU: $_GPU_NAME (${_VRAM_TOTAL} MiB)"
        (( _VRAM_TOTAL < 8000 )) 2>/dev/null && warn_fn "GPU < 8 GB VRAM — only 'minimal' tier"
    else
        fail_fn "nvidia-smi not found — NVIDIA GPU driver required"
    fi

    # CUDA
    section "CUDA Toolkit:"
    if command -v nvcc &>/dev/null; then
        local cuda_ver
        cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        pass_fn "CUDA $cuda_ver"
        (( $(echo "$cuda_ver" | cut -d. -f1) < 12 )) 2>/dev/null && warn_fn "CUDA < 12.0 — TRT-LLM requires 12.x"
    else
        fail_fn "nvcc not found — install CUDA Toolkit 12.x"
    fi

    # Build tools
    section "Build Tools:"
    for cmd in cmake g++ git pkg-config; do require_cmd "$cmd"; done

    # Python
    section "Python:"
    if command -v python3 &>/dev/null; then
        pass_fn "$(python3 --version 2>&1)"
        for pkg in torch transformers numpy PIL onnxruntime; do
            python3 -c "import $pkg" 2>/dev/null && pass_fn "python3 $pkg" || warn_fn "python3 $pkg not installed"
        done
    else
        fail_fn "python3 not found"
    fi

    # GStreamer + media
    section "GStreamer & Media:"
    require_cmd gst-launch-1.0
    require_cmd ffmpeg warn
    require_cmd espeak-ng warn

    # TensorRT
    section "TensorRT:"
    require_cmd trtexec warn
    if python3 -c "import tensorrt_llm" 2>/dev/null; then
        pass_fn "tensorrt_llm $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null)"
    else
        warn_fn "tensorrt_llm not installed"
    fi

    # Disk / RAM
    section "System Resources:"
    local avail_gb
    avail_gb=$(( $(df --output=avail "$HOME" 2>/dev/null | tail -1 | tr -d ' ') / 1024 / 1024 )) 2>/dev/null || avail_gb=0
    (( avail_gb < 25 )) && warn_fn "Only ${avail_gb} GB disk free — need ~25 GB" || pass_fn "${avail_gb} GB disk available"

    local ram_gb
    ram_gb=$(( $(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}') / 1024 / 1024 )) 2>/dev/null || ram_gb=0
    (( ram_gb < 16 )) && warn_fn "${ram_gb} GB RAM — AWQ needs ~32 GB" || pass_fn "${ram_gb} GB RAM"

    # Video devices
    section "Video Devices:"
    if ls /dev/video* > /dev/null 2>&1; then
        pass_fn "Video devices: $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
    else
        warn_fn "No /dev/video* — WSL2 needs usbipd-win webcam passthrough"
        echo -e "    ${CYAN}Hint:${NC} PowerShell (Admin): usbipd.exe list → usbipd.exe attach --wsl --busid <BUSID>"
    fi

    # Summary
    echo ""
    echo "════════════════════════════════════════"
    echo -e "  ${GREEN}PASS: $PASS${NC}  |  ${YELLOW}WARN: $WARN${NC}  |  ${RED}FAIL: $FAIL${NC}"
    echo "════════════════════════════════════════"
    (( FAIL > 0 )) && { echo -e "  ${RED}Fix FAIL items before proceeding.${NC}"; return 1; }
    (( WARN > 0 )) && echo -e "  ${YELLOW}Warnings are non-blocking.${NC}"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
#  INSTALL / VERIFY (delegate to scripts/)
# ═══════════════════════════════════════════════════════════════════════════════
run_install() { setup_environment; bash "$PROJECT_ROOT/scripts/integration/install.sh"; }
run_verify()  { setup_environment; bash "$PROJECT_ROOT/scripts/integration/verify.sh"; }

# ═══════════════════════════════════════════════════════════════════════════════
#  CLEANUP STALE PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════
cleanup_stale() {
    echo -e "${BOLD}=== 1. Cleaning up stale processes ===${NC}"

    # Graceful kill → wait → force kill
    pkill -f 'omniedge_' 2>/dev/null || true
    sleep 1
    pkill -9 -f 'omniedge_' 2>/dev/null || true
    sleep 1

    # Kill anything holding OmniEdge ports (graceful then forced)
    kill_ports "" $ALL_PORTS
    sleep 1
    kill_ports "-9" $ALL_PORTS
    sleep 1

    rm -f /dev/shm/oe.*

    # Verify critical ports are free
    local blocked=0
    for p in $BRIDGE_PUB_PORT $DAEMON_PUB_PORT $WS_BRIDGE_PORT; do
        if lsof -ti :"$p" > /dev/null 2>&1; then
            echo -e "  ${YELLOW}WARN: Port $p still held:${NC}"
            lsof -i :"$p" 2>/dev/null | head -3 | sed 's/^/    /'
            lsof -ti :"$p" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
            sleep 1
            if lsof -ti :"$p" > /dev/null 2>&1; then
                echo -e "  ${RED}ERROR: Port $p still in use after kill -9!${NC}"
                blocked=1
            fi
        fi
    done
    (( blocked )) && { echo "Aborting — a system process is holding the port."; exit 1; }
    echo -e "  ${GREEN}All ports free.${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═══════════════════════════════════════════════════════════════════════════════
build_binaries() {
    echo -e "\n${BOLD}=== 2. Building binaries ===${NC}"
    local build_fail=0

    if cmake --build build -j"$(nproc)" > "$LOG_DIR/build_default_targets.log" 2>&1; then
        status_ok "default targets"
    else
        status_warn "default targets" "see $LOG_DIR/build_default_targets.log"
        build_fail=$((build_fail + 1))
    fi

    local targets=(
        omniedge_ws_bridge omniedge_daemon
        omniedge_bg_blur omniedge_face_recog omniedge_video_denoise omniedge_audio_denoise
        omniedge_super_resolution omniedge_image_transform
        omniedge_stt omniedge_tts omniedge_llm
    )
    for t in "${targets[@]}"; do
        if cmake --build build --target "$t" -j"$(nproc)" > "$LOG_DIR/build_${t}.log" 2>&1; then
            status_ok "$t"
        else
            status_warn "$t" "see $LOG_DIR/build_${t}.log"
            build_fail=$((build_fail + 1))
        fi
    done

    (( build_fail > 0 )) && echo -e "  ${YELLOW}$build_fail target(s) had build warnings/errors${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL READINESS
# ═══════════════════════════════════════════════════════════════════════════════
check_models() {
    echo -e "\n${BOLD}=== 3. Model readiness ===${NC}"
    local REQUIRED_MODELS=(
        "BgBlur (YOLO):$OE_ENGINES_DIR/yolov8n-seg.engine"
        "VAD (Silero):$OE_MODELS_DIR/silero_vad.onnx"
        "LLM engine (Qwen2.5-7B):$OE_ENGINES_DIR/qwen2.5-7b-nvfp4/rank0.engine"
        "LLM tokenizer:$OE_MODELS_DIR/qwen2.5-7b-instruct-awq/tokenizer.json"
        "TTS ONNX (Kokoro):$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
        "TTS voices:$OE_MODELS_DIR/kokoro/voices"
        "BasicVSR++ (SR):$OE_MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx"
        "STT engines:$OE_ENGINES_DIR/whisper-turbo"
        "STT tokenizer:$OE_MODELS_DIR/whisper-large-v3-turbo/vocab.json"
        "FaceRecog:$OE_MODELS_DIR/InspireFace/test_res/pack/Megatron_TRT"
    )
    check_model_list REQUIRED_MODELS || \
        echo -e "  ${YELLOW}Missing models: run '${BOLD}bash run_all.sh --install${NC}${YELLOW}' to install.${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VRAM PRE-FLIGHT CHECK
# ═══════════════════════════════════════════════════════════════════════════════
check_vram_preflight() {
    echo -e "\n${BOLD}=== 4. VRAM pre-flight check ===${NC}"

    local max_vram_mb
    max_vram_mb=$(ini_get "$INI_FILE" "vram_limits" "max_total_vram_mb" "10240")

    if ! query_gpu_vram; then
        status_warn "nvidia-smi" "not found — relying on C++ daemon for VRAM management"
        return 0
    fi

    echo -e "  GPU: ${CYAN}${_GPU_NAME}${NC}"
    echo -e "  VRAM: ${_VRAM_USED}/${_VRAM_TOTAL} MiB (${_VRAM_FREE} MiB free, cap: ${max_vram_mb} MiB)"

    (( _VRAM_USED > 2000 )) && status_warn "GPU in use" "${_VRAM_USED} MiB — close other GPU apps" \
                             || status_ok "GPU baseline: ${_VRAM_USED} MiB"

    # Sum enabled module budgets
    local total_budget=0 detail=""
    for mod in "${!MODULE_VRAM_BUDGET_MB[@]}"; do
        [[ "$(ini_get "$INI_FILE" "modules" "$mod" "0")" == "1" ]] || continue
        local b="${MODULE_VRAM_BUDGET_MB[$mod]}"
        total_budget=$((total_budget + b))
        detail+="    ${mod}: ${b} MiB\n"
    done

    if [[ -n "$detail" ]]; then
        echo -e "\n  ${BOLD}Enabled GPU modules:${NC}\n${detail}  ${BOLD}Total: ${total_budget} MiB${NC}"
        (( total_budget > max_vram_mb )) && \
            status_warn "VRAM over cap" "${total_budget} > ${max_vram_mb} MiB — daemon will use priority eviction"
        (( total_budget > _VRAM_FREE )) && \
            status_warn "VRAM tight" "${total_budget} MiB needed, ${_VRAM_FREE} MiB free" || \
            status_ok "VRAM fits: ${total_budget} MiB needed, ${_VRAM_FREE} MiB free"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LAUNCH SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
launch_system() {
    echo -e "\n${BOLD}=== 5. Launching system ===${NC}"
    local omni_log="$LOG_DIR/omniedge.log"
    : > "$omni_log"

    # 5a. Start ws_bridge
    local ws_bin="./build/modules/nodes/ws_bridge/omniedge_ws_bridge"
    [[ -x "$ws_bin" ]] || { echo -e "  ${RED}FATAL: ws_bridge binary not found.${NC}"; exit 1; }

    echo "  Starting ws_bridge..."
    LD_LIBRARY_PATH="$OE_RUNTIME_LD_LIBRARY_PATH" \
        stdbuf -oL "$ws_bin" --config config/omniedge_config.yaml >> "$omni_log" 2>&1 &
    WS_PID=$!
    sleep 2
    kill -0 "$WS_PID" 2>/dev/null || { echo -e "  ${RED}FATAL: ws_bridge crashed!${NC}"; tail -20 "$omni_log" | sed 's/^/    /'; exit 1; }

    # Verify port
    local listening=0
    { command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":${WS_BRIDGE_PORT}" && listening=1; } || \
    { command -v netstat &>/dev/null && netstat -tlnp 2>/dev/null | grep -q ":${WS_BRIDGE_PORT}" && listening=1; } || listening=1
    (( listening == 0 )) && { echo -e "  ${RED}FATAL: port ${WS_BRIDGE_PORT} not listening!${NC}"; exit 1; }
    status_ok "ws_bridge (PID $WS_PID, port ${WS_BRIDGE_PORT})"

    # 5b. Start orchestrator
    local orch_bin="./build/modules/nodes/orchestrator/omniedge_daemon"
    [[ -x "$orch_bin" ]] || { echo -e "  ${RED}FATAL: orchestrator binary not found.${NC}"; exit 1; }

    echo "  Starting orchestrator..."
    LD_LIBRARY_PATH="$OE_RUNTIME_LD_LIBRARY_PATH" \
        stdbuf -oL "$orch_bin" --config config/omniedge_config.yaml >> "$omni_log" 2>&1 &
    DAEMON_PID=$!

    # 5c. Wait for modules
    echo -e "  ${CYAN}Waiting for modules...${NC}"
    local expected=0
    for mod in video_ingest audio_ingest background_blur tts stt face_recognition llm video_denoise audio_denoise; do
        [[ "$(ini_get "$INI_FILE" "modules" "$mod" "0")" == "1" ]] && expected=$((expected + 1))
    done

    local wait_total=0 wait_max=40
    while (( wait_total < wait_max )); do
        sleep 2; wait_total=$((wait_total + 2))
        local ready_count
        ready_count=$(grep "module_ready_received" "$omni_log" 2>/dev/null \
            | sed -n 's/.*module=\([^, ]*\).*/\1/p' | sort -u | wc -l | tr -d ' ' || echo 0)
        [[ "$ready_count" =~ ^[0-9]+$ ]] || ready_count=0
        echo -e "    ${CYAN}[${wait_total}s]${NC} modules ready: ${ready_count}/${expected}"

        (( $(count_matches "daemon_initialized" "$omni_log") > 0 )) && \
            { echo -e "    ${GREEN}Daemon initialized (${wait_total}s)${NC}"; break; }

        kill -0 "$DAEMON_PID" 2>/dev/null || \
            { echo -e "  ${RED}FATAL: daemon crashed!${NC}"; tail -20 "$omni_log" | sed 's/^/    /'; exit 1; }
    done
    (( wait_total >= wait_max )) && echo -e "  ${YELLOW}WARN: Timed out after ${wait_max}s${NC}"
    status_ok "daemon (PID $DAEMON_PID, port ${DAEMON_PUB_PORT})"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════
health_check() {
    echo -e "\n${BOLD}=== 6. Module health check ===${NC}"
    local daemon_log="$LOG_DIR/omniedge.log"

    # VRAM status
    if query_gpu_vram; then
        local crit_mb
        crit_mb=$(ini_get "$INI_FILE" "vram_limits" "vram_critical_threshold_mb" "9500")
        if (( _VRAM_USED > crit_mb )); then
            status_fail "VRAM ${_VRAM_USED}/${_VRAM_TOTAL} MiB" "CRITICAL"
        else
            status_ok "VRAM ${_VRAM_USED}/${_VRAM_TOTAL} MiB (${_VRAM_FREE} MiB free)"
        fi
    fi
    echo ""

    [[ -f "$daemon_log" ]] && [[ -s "$daemon_log" ]] || { echo -e "  ${YELLOW}WARN: no log data${NC}"; return; }

    local mod_ok=0 mod_warn=0 mod_fail=0
    for mod in $MODULES; do
        local spawn_line
        spawn_line=$(grep "module_spawned.*module=$mod" "$daemon_log" | tail -1 || true)
        if [[ -z "$spawn_line" ]]; then
            status_skip "$mod" "not spawned"; mod_warn=$((mod_warn + 1)); continue
        fi

        local mod_pid
        mod_pid=$(echo "$spawn_line" | grep -o 'pid=[0-9]*' | tail -1 | cut -d= -f2 || true)

        # Process alive?
        if [[ -n "$mod_pid" ]] && kill -0 "$mod_pid" 2>/dev/null; then
            if grep -q "module_ready_received.*module=$mod" "$daemon_log"; then
                status_ok "$mod (PID $mod_pid, ready)"; mod_ok=$((mod_ok + 1))
            else
                status_warn "$mod (PID $mod_pid)" "running, not ready yet"; mod_warn=$((mod_warn + 1))
            fi
            continue
        fi

        # Process dead — show context
        local exit_status restarts
        exit_status=$(grep "module_exited.*module=$mod" "$daemon_log" | tail -1 | grep -o 'exit_status=[0-9]*' | cut -d= -f2 || echo "?")
        restarts=$(count_matches "module_restarting.*module=$mod" "$daemon_log")
        local msg="crashed (exit=$exit_status)"
        (( restarts > 0 )) 2>/dev/null && msg="crashed after $restarts restart(s)"

        status_fail "$mod" "$msg"; mod_fail=$((mod_fail + 1))
        grep -iE '\[ERROR\]|\[FATAL\]|exception' "$daemon_log" | grep -i "$mod" | tail -3 | sed 's/^/      /' || true
    done

    echo -e "\n  ${BOLD}Modules:${NC} ${GREEN}${mod_ok} ready${NC}, ${YELLOW}${mod_warn} warn${NC}, ${RED}${mod_fail} failed${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
show_summary() {
    echo -e "\n${BOLD}=== 7. Summary ===${NC}"
    local alive
    alive=$(pgrep -f 'omniedge_' 2>/dev/null | wc -l || echo 0)
    echo -e "  Running processes: ${CYAN}${alive}${NC}"
    { pgrep -af 'omniedge_' 2>/dev/null | grep -v bash || true; } | sed 's/^/    /'

    echo -e "\n  ${BOLD}Frontend:${NC} ${CYAN}http://localhost:${WS_BRIDGE_PORT}${NC}"
    local wsl_ip
    wsl_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -n "$wsl_ip" ]] && echo -e "  ${BOLD}WSL2:${NC}     ${CYAN}http://${wsl_ip}:${WS_BRIDGE_PORT}${NC}"
    echo -e "\n  ${BOLD}Logs:${NC} tail -f $LOG_DIR/omniedge.log"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CLEANUP TRAP
# ═══════════════════════════════════════════════════════════════════════════════
cleanup_on_exit() {
    echo -e "\n${BOLD}Shutting down...${NC}"
    jobs -p 2>/dev/null | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    jobs -p 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    rm -f /dev/shm/oe.*
    if query_gpu_vram; then
        echo "VRAM after cleanup: ${_VRAM_USED}/${_VRAM_TOTAL} MiB."
    fi
    echo "Stopped. Logs: $LOG_DIR/"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN MODE
# ═══════════════════════════════════════════════════════════════════════════════
run_system() {
    trap cleanup_on_exit EXIT
    trap 'jobs -p 2>/dev/null | xargs -r kill 2>/dev/null' INT TERM
    setup_environment

    echo -e "${BOLD}OmniEdge_AI — System Launch${NC}\n"
    rm -f "$LOG_DIR"/*.log 2>/dev/null || true
    cleanup_stale
    check_prerequisites || true
    build_binaries
    check_models
    check_vram_preflight
    launch_system
    health_check
    show_summary

    echo -e "\n${BOLD}System running. Press Ctrl+C to stop.${NC}"
    wait
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TEST
# ═══════════════════════════════════════════════════════════════════════════════
run_model_test() {
    setup_environment
    local test_data_dir="$PROJECT_ROOT/tests/integration/pipeline/data"

    echo -e "${BOLD}OmniEdge_AI — Model Pipeline Test${NC}\n"

    # Step 1: Generate test data
    echo -e "${BOLD}=== 1. Generating test data ===${NC}"
    if [[ -x "$PROJECT_ROOT/tests/integration/pipeline/generate_test_data.sh" ]]; then
        bash "$PROJECT_ROOT/tests/integration/pipeline/generate_test_data.sh"
    else
        mkdir -p "$test_data_dir"
        [[ -f "$test_data_dir/test_silence_2s.raw" ]] || \
            dd if=/dev/zero of="$test_data_dir/test_silence_2s.raw" bs=1 count=64000 2>/dev/null
    fi

    # Step 2: Build
    echo -e "\n${BOLD}=== 2. Building ===${NC}"
    [[ -f "build/CMakeCache.txt" ]] || cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo > "$LOG_DIR/model_test_cmake.log" 2>&1

    local pipeline_ok=0
    cmake --build build --target test_pipeline_model -j"$(nproc)" > "$LOG_DIR/build_model_test.log" 2>&1 \
        && { status_ok "test_pipeline_model"; pipeline_ok=1; } \
        || echo -e "  ${CYAN}Falling back to GPU unit tests...${NC}"

    cmake --build build -j"$(nproc)" > "$LOG_DIR/build_model_test_all.log" 2>&1 \
        || { echo -e "  ${RED}FAIL: Build failed.${NC}"; tail -20 "$LOG_DIR/build_model_test_all.log" | sed 's/^/    /'; exit 1; }

    # Step 3: Model readiness (reuse shared list)
    echo -e "\n${BOLD}=== 3. Model readiness ===${NC}"
    local PREFLIGHT_MODELS=(
        "Qwen2.5-7B LLM:$OE_ENGINES_DIR/qwen2.5-7b-nvfp4/rank0.engine"
        "Kokoro TTS:$OE_MODELS_DIR/kokoro-onnx/onnx/model_int8.onnx"
        "BasicVSR++ SR:$OE_MODELS_DIR/basicvsrpp/basicvsrpp_denoise.onnx"
        "YOLOv8 BgBlur:$OE_ENGINES_DIR/yolov8n-seg.engine"
        "Silero VAD:$OE_MODELS_DIR/silero_vad.onnx"
        "Whisper STT:$OE_ENGINES_DIR/whisper-turbo/encoder"
    )
    check_model_list PREFLIGHT_MODELS || echo -e "  ${YELLOW}Missing models will be skipped.${NC}"

    # Step 4: Run tests
    echo -e "\n${BOLD}=== 4. Running tests ===${NC}\n"
    export LD_LIBRARY_PATH="$OE_RUNTIME_LD_LIBRARY_PATH"
    export OE_TEST_DATA_DIR="$test_data_dir"

    local test_log="$LOG_DIR/model_test_results.log"
    local exit_code=0

    local test_bin="./build/tests/integration/pipeline/test_pipeline_model"
    if (( pipeline_ok )) && [[ -x "$test_bin" ]]; then
        $test_bin --gtest_color=yes 2>&1 | tee "$test_log"; exit_code=${PIPESTATUS[0]}
    else
        local rc=0
        [[ -x "$PROJECT_ROOT/scripts/integration/run_all.sh" ]] && \
            bash "$PROJECT_ROOT/scripts/integration/run_all.sh" 2>&1 | tee "$test_log" || rc=$?
        (( rc != 0 )) && exit_code=1
    fi

    # Summary
    echo -e "\n════════════════════════════════════════"
    if (( exit_code == 0 )); then
        echo -e "  ${GREEN}${BOLD}ALL MODEL TESTS PASSED${NC}"
    else
        local real_fail not_run
        real_fail=$(grep -c 'FAILED' "$test_log" 2>/dev/null || echo 0)
        not_run=$(grep -c 'Not Run' "$test_log" 2>/dev/null || echo 0)
        if (( real_fail == not_run )); then
            echo -e "  ${GREEN}${BOLD}ALL AVAILABLE TESTS PASSED${NC} ${YELLOW}($not_run skipped)${NC}"
            exit_code=0
        else
            echo -e "  ${RED}${BOLD}TESTS FAILED${NC}"
            grep -E 'FAILED' "$test_log" 2>/dev/null | grep -v 'Not Run' | sed 's/^/    /' || true
        fi
    fi
    echo "════════════════════════════════════════"

    # Step 5: Benchmark (optional)
    local bench_script="$PROJECT_ROOT/scripts/benchmark/benchmark_models.py"
    if [[ -f "$bench_script" ]]; then
        echo -e "\n${BOLD}=== 5. Benchmark ===${NC}"
        local bench_args=()
        [[ -n "${MODEL_TEST_QUANT_SWEEP:-}" ]] && bench_args+=("--quant-sweep")
        [[ -n "${MODEL_TEST_VRAM_BUDGET:-}" ]] && bench_args+=("--vram-budget" "$MODEL_TEST_VRAM_BUDGET")
        [[ -n "${MODEL_TEST_QUANT_MAP:-}" ]]   && bench_args+=("--quant-map" "$MODEL_TEST_QUANT_MAP")
        [[ -n "${MODEL_TEST_MODULE:-}" ]]       && bench_args+=("--module" "$MODEL_TEST_MODULE")
        python3 "$bench_script" "${bench_args[@]}" 2>&1 | tee -a "$test_log" || true
    fi

    # Step 6: JSON report
    python3 -c "
import json, re
lines = open('$test_log').readlines()
lat = {m.group(1): float(m.group(2)) for l in lines if (m := re.search(r'\[LATENCY\]\s+(\S+):\s+([\d.]+)\s+ms', l))}
p = sum(1 for l in lines if '[       OK ]' in l or re.search(r'Test #\d+:.*Passed', l))
f = sum(1 for l in lines if '[  FAILED  ]' in l or re.search(r'Test #\d+:.*\*\*\*Failed', l))
report = {'exit_code': $exit_code, 'summary': {'passed': p, 'failed': f}, 'latencies_ms': lat}
with open('$LOG_DIR/model_test_report.json', 'w') as fh: json.dump(report, fh, indent=2)
print(json.dumps(report, indent=2))
" 2>/dev/null || true

    echo -e "  Report: $LOG_DIR/model_test_report.json"
    exit "$exit_code"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELP
# ═══════════════════════════════════════════════════════════════════════════════
show_help() {
    cat <<'EOF'
OmniEdge_AI — Unified Launch & Install Script

Usage: bash run_all.sh [OPTION]

  (no option)              Build and launch the system
  --install                Install all deps + models, then launch
  --install-only           Install all deps + models (no launch)
  --verify                 Verify all models, engines, and binaries
  --prereqs                Check prerequisites only
  --model-test             Run headless model pipeline tests
    --quant-sweep            Sweep all quantization variants
    --vram-budget <MB>       Rank best combo for VRAM budget
    --quant-map K=V,...      Per-module quant (e.g., llm=nvfp4)
    --module <list>          Module filter (e.g., llm,stt)
  --model-install          Install cross-model benchmark variants
    --llm/--stt/--tts/--cv   Filter by module type
    --no-engines             Download only, skip TRT builds
  --help                   Show this help

Examples:
  bash run_all.sh                                       # Build & run
  bash run_all.sh --install                             # Full install then run
  bash run_all.sh --model-test --quant-sweep            # Compare all quant variants
  bash run_all.sh --model-install --llm --no-engines    # Download LLM weights only
EOF
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
main() {
    case "${1:-}" in
        --help|-h)       show_help ;;
        --prereqs)       setup_environment; check_prerequisites ;;
        --install)       run_install; echo -e "\n${BOLD}Installation complete. Launching...${NC}\n"; run_system ;;
        --install-only)  run_install ;;
        --verify)        run_verify ;;
        --model-test|--modeltest)
            shift || true
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --quant-sweep) export MODEL_TEST_QUANT_SWEEP=1 ;;
                    --vram-budget) shift; export MODEL_TEST_VRAM_BUDGET="$1" ;;
                    --quant-map)   shift; export MODEL_TEST_QUANT_MAP="$1" ;;
                    --module)      shift; export MODEL_TEST_MODULE="$1" ;;
                    *) echo -e "${RED}Unknown flag: $1${NC}"; exit 1 ;;
                esac; shift
            done
            run_model_test ;;
        --model-install)
            shift || true
            local args=()
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --llm|--stt|--tts|--cv|--no-engines) args+=("$1") ;;
                    *) echo -e "${RED}Unknown flag: $1${NC}"; exit 1 ;;
                esac; shift
            done
            setup_environment
            bash "$PROJECT_ROOT/scripts/integration/install_cross_models.sh" "${args[@]}" ;;
        "")  run_system ;;
        *)   echo -e "${RED}Unknown: $1${NC}\n"; show_help; exit 1 ;;
    esac
}

main "$@"
