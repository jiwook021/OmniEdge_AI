#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Shared shell utilities
#
# Source this file from other shell scripts:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/common.sh"                       # if caller lives in scripts/integration/
#   source "$SCRIPT_DIR/scripts/integration/common.sh"   # if caller lives in project root
#
# NOTE: This file is a library meant to be sourced, NOT executed directly.
# The sourcing script MUST have `set -euo pipefail` at the top.
#
# Provides:
#   Colours           : RED, GREEN, YELLOW, CYAN, BOLD, NC
#   Status printers   : status_ok, status_fail, status_warn, status_skip
#   Section headers   : section
#   Counter helpers   : pass_fn, fail_fn, warn_fn  (increment PASS/FAIL/WARN)
#   Install helpers   : ok, skip, warn, record_error
#   Env               : OE_MODELS_DIR, OE_ENGINES_DIR (exported with defaults)
#   Logging           : log_info, log_warn, log_error
#   Execution         : run_test (handles exit 0=pass, 77=skip, *=fail)
#   Checks            : check_binary, dir_exists, file_exists, check_dir, check_file
#   GPU               : resolve_gpu_tier
#   INI               : ini_get  (read a key from an INI [section])
#   Ports             : ini_get_all_ports  (all port values from [ports])
#   LD_LIBRARY_PATH   : build_ld_library_path  (construct runtime lib path)
#   Auto-install      : ensure_command, ensure_python_module, with_auto_install
#   Counters          : TOTAL_PASS, TOTAL_FAIL, TOTAL_SKIP (global, mutable)
#   Download          : download_hf  (idempotent HuggingFace download)
#   Parsing           : parse_ctest_summary  (extract pass/fail/ran from ctest output)
#   Networking        : wait_for_port  (poll until a port is listening)
# =============================================================================

# Guard against double-sourcing.
[[ -n "${_OE_COMMON_SOURCED:-}" ]] && return 0
readonly _OE_COMMON_SOURCED=1

# -- sudo shim for Docker (running as root, sudo not installed) -------------
if [ "$(id -u)" -eq 0 ] && ! command -v sudo &>/dev/null; then
    sudo() { "$@"; }
fi

# -- Colours ----------------------------------------------------------------
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# -- Environment ------------------------------------------------------------
export OE_MODELS_DIR="${OE_MODELS_DIR:-$HOME/omniedge_models}"
export OE_ENGINES_DIR="${OE_ENGINES_DIR:-$OE_MODELS_DIR/trt_engines}"

# -- Global counters (callers may reset these) ------------------------------
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

# -- Status line printers ---------------------------------------------------

# Description: Print a green OK status line
# Args: $1 - message
status_ok()   { echo -e "  ${GREEN}OK${NC}    $1"; }

# Description: Print a red FAIL status line with a hint
# Args: $1 - message, $2 - hint/path
status_fail() { echo -e "  ${RED}FAIL${NC}  $1  ${YELLOW}→ $2${NC}"; }

# Description: Print a yellow WARN status line with a hint
# Args: $1 - message, $2 - hint
status_warn() { echo -e "  ${YELLOW}WARN${NC}  $1  → $2"; }

# Description: Print a yellow SKIP status line with a reason
# Args: $1 - message, $2 - reason
status_skip() { echo -e "  ${YELLOW}SKIP${NC}  $1  → $2"; }

# -- Section header ---------------------------------------------------------

# Description: Print a bold section header with a blank line before it
# Args: $1 - section title
section() { echo ""; echo -e "${BOLD}$1${NC}"; }

# -- Counter-based printers (for prereq/verify scripts) --------------------
# These use and increment module-level PASS/FAIL/WARN counters.
# Callers must initialise those counters before use.

# Description: Print green PASS and increment PASS counter
# Args: $1 - message
pass_fn() { echo -e "  ${GREEN}PASS${NC}  $1"; PASS=$((PASS + 1)); }

# Description: Print red FAIL and increment FAIL counter
# Args: $1 - message
fail_fn() { echo -e "  ${RED}FAIL${NC}  $1"; FAIL=$((FAIL + 1)); }

# Description: Print yellow WARN and increment WARN counter
# Args: $1 - message
warn_fn()  { echo -e "  ${YELLOW}WARN${NC}  $1"; WARN=$((WARN + 1)); }

# -- Install-script helpers -------------------------------------------------

# Description: Print green OK (for install steps that succeeded)
# Args: $1 - message
ok()           { echo -e "  ${GREEN}OK${NC}    $1"; }

# Description: Print cyan SKIP (for install steps already completed)
# Args: $1 - message
skip()         { echo -e "  ${CYAN}SKIP${NC}  $1 (already present)"; }

# Description: Print yellow WARN (non-fatal install issue)
# Args: $1 - message
warn()         { echo -e "  ${YELLOW}WARN${NC}  $1"; }

# Description: Print red FAIL and increment ERRORS counter
# Args: $1 - message
# Globals: writes ERRORS
record_error() { echo -e "  ${RED}FAIL${NC}  $1"; ERRORS=$((ERRORS + 1)); }

# -- Logging ----------------------------------------------------------------

# Description: Print an INFO-level log message (green prefix)
# Args: $* - message text
log_info()  { echo -e "  ${GREEN}INFO${NC}  $*"; }

# Description: Print a WARN-level log message (yellow prefix)
# Args: $* - message text
log_warn()  { echo -e "  ${YELLOW}WARN${NC}  $*"; }

# Description: Print an ERROR-level log message to stderr (red prefix)
# Args: $* - message text
log_error() { echo -e "  ${RED}ERROR${NC} $*" >&2; }

# -- Test execution ---------------------------------------------------------

# Description: Run a command and interpret its exit code as pass/skip/fail
# Args: $1 - display name, $2.. - command and arguments
# Returns: always 0 (failures tracked via TOTAL_FAIL counter)
# Globals: writes TOTAL_PASS, TOTAL_SKIP, TOTAL_FAIL
#
# Exit code mapping:
#   0  -> PASS (increments TOTAL_PASS)
#   77 -> SKIP (increments TOTAL_SKIP, deps unavailable)
#   *  -> FAIL (increments TOTAL_FAIL)
run_test() {
    local display_name="$1"
    shift

    echo ""
    echo -e "${BOLD}-- ${display_name} --${NC}"

    local rc=0
    local start_time end_time elapsed_seconds
    start_time=$(date +%s)

    "$@" || rc=$?

    end_time=$(date +%s)
    elapsed_seconds=$((end_time - start_time))

    case "$rc" in
        0)
            TOTAL_PASS=$((TOTAL_PASS + 1))
            echo -e "  ${GREEN}PASS${NC}  ${display_name} (${elapsed_seconds}s)"
            ;;
        77)
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            echo -e "  ${YELLOW}SKIP${NC}  ${display_name} (deps unavailable)"
            ;;
        *)
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            echo -e "  ${RED}FAIL${NC}  ${display_name} (exit ${rc}, ${elapsed_seconds}s)"
            ;;
    esac

    # Always return 0 — failures are tracked via TOTAL_FAIL counter.
    return 0
}

# -- Summary printer --------------------------------------------------------

# Description: Print a summary bar with PASS/SKIP/FAIL counts
# Globals: reads TOTAL_PASS, TOTAL_SKIP, TOTAL_FAIL
print_summary() {
    echo ""
    echo "================================================================"
    echo -e "  ${GREEN}PASS: ${TOTAL_PASS}${NC}  |  ${YELLOW}SKIP: ${TOTAL_SKIP}${NC}  |  ${RED}FAIL: ${TOTAL_FAIL}${NC}"
    echo "================================================================"
}

# -- File / directory existence checks --------------------------------------

# Description: Check if a directory exists and is non-empty
# Args: $1 - directory path
# Returns: 0 if exists and non-empty, 1 otherwise
dir_exists()  { [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null)" ]]; }

# Description: Check if a regular file exists
# Args: $1 - file path
# Returns: 0 if exists, 1 otherwise
file_exists() { [[ -f "$1" ]]; }

# Description: Check if a binary exists and is executable
# Args: $1 - binary path
# Returns: 0 if executable, 1 otherwise
check_binary() {
    local binary_path="$1"
    [[ -f "$binary_path" ]] && [[ -x "$binary_path" ]]
}

# Description: Check a directory exists and print pass/fail status
# Args: $1 - label, $2 - path, $3 - severity (pass/fail/warn, default: fail)
# Globals: writes TOTAL_PASS, TOTAL_FAIL, TOTAL_SKIP via status helpers
check_dir() {
    local label="$1" path="$2" severity="${3:-fail}"
    if dir_exists "$path"; then
        status_ok "$label"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    elif [[ "$severity" == "warn" ]]; then
        status_warn "$label" "$path not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
    else
        status_fail "$label" "$path"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# Description: Check a file exists and print pass/fail status
# Args: $1 - label, $2 - path, $3 - severity (pass/fail/warn, default: fail)
# Globals: writes TOTAL_PASS, TOTAL_FAIL, TOTAL_SKIP via status helpers
check_file() {
    local label="$1" path="$2" severity="${3:-fail}"
    if file_exists "$path"; then
        status_ok "$label"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    elif [[ "$severity" == "warn" ]]; then
        status_warn "$label" "$path not found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
    else
        status_fail "$label" "$path"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# -- GPU tier resolution ----------------------------------------------------

# Description: Detect GPU generation from nvidia-smi output
# Returns: prints one of: "blackwell", "ada", "ampere", "turing", "unknown"
#
# GPU family pattern matching:
#   Blackwell : RTX 5090/5080/5070 (*RTX*50*), RTX PRO 6000/3000 (*RTX*PRO*),
#               any card with "Blackwell" in name
#   Ada       : RTX 4090/4080/4070/4060 (*RTX*40*), L40/L40S (*L40*),
#               any card with "Ada" in name
#   Ampere    : RTX 3090/3080/3070/3060 (*RTX*30*), A100/A6000 data center
#   Turing    : RTX 2080/2070/2060 (*RTX*20*), T4 inference card
resolve_gpu_tier() {
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")

    case "$gpu_name" in
        *RTX*50*|*RTX*PRO*6000*|*RTX*PRO*3000*|*Blackwell*)
            echo "blackwell" ;;
        *RTX*40*|*Ada*|*L40*)
            echo "ada" ;;
        *RTX*30*|*A100*|*A6000*)
            echo "ampere" ;;
        *RTX*20*|*T4*|*Turing*)
            echo "turing" ;;
        *)
            echo "unknown" ;;
    esac
}

# -- INI file reader --------------------------------------------------------

# Description: Read a value from a standard INI file
# Args: $1 - file path, $2 - section name, $3 - key name, $4 - default value (optional)
# Returns: prints the value (or default if not found)
#
# Strips inline comments (after ';') and surrounding whitespace.
#
# Example:
#   ini_get config/omniedge.ini ports llm 5561
#   ini_get config/omniedge.ini vram_limits max_total_vram_mb 10240
ini_get() {
    local file="$1" section="$2" key="$3" default="${4:-}"
    local in_section=0 value=""

    [[ -f "$file" ]] || { echo "$default"; return; }

    while IFS= read -r line; do
        # Strip inline comments (everything after ';')
        line="${line%%;*}"
        # Trim leading whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        # Trim trailing whitespace
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue

        # Detect [section] headers
        if [[ "$line" == "["*"]" ]]; then
            local sec="${line#[}"; sec="${sec%]}"
            [[ "$sec" == "$section" ]] && in_section=1 || in_section=0
            continue
        fi

        # Match key = value inside target section
        if (( in_section )); then
            local k="${line%%=*}"    # everything before first '='
            local v="${line#*=}"     # everything after first '='
            # Trim whitespace from key
            k="${k#"${k%%[![:space:]]*}"}"; k="${k%"${k##*[![:space:]]}"}"
            # Trim whitespace from value
            v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"
            [[ "$k" == "$key" ]] && value="$v"
        fi
    done < "$file"

    echo "${value:-$default}"
}

# -- INI section key lister -------------------------------------------------

# Description: List all keys in a given INI section
# Args: $1 - file path, $2 - section name
# Returns: prints space-separated list of keys
ini_get_section_keys() {
    local file="$1" section="$2"
    local in_section=0 keys=""

    [[ -f "$file" ]] || { echo ""; return; }

    while IFS= read -r line; do
        line="${line%%;*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue

        if [[ "$line" == "["*"]" ]]; then
            local sec="${line#[}"; sec="${sec%]}"
            [[ "$sec" == "$section" ]] && in_section=1 || in_section=0
            continue
        fi

        if (( in_section )); then
            local k="${line%%=*}"
            k="${k#"${k%%[![:space:]]*}"}"; k="${k%"${k##*[![:space:]]}"}"
            [[ -n "$k" ]] && keys="$keys $k"
        fi
    done < "$file"

    echo "$keys"
}

# -- Port helpers -----------------------------------------------------------

# Description: Read all port values from the [ports] section of an INI file
# Args: $1 - INI file path
# Returns: prints space-separated list of port numbers
#
# Used by cleanup/health-check scripts to avoid hardcoding port numbers.
# Example: ALL_PORTS=$(ini_get_all_ports config/omniedge.ini)
ini_get_all_ports() {
    local file="$1"
    local ports=""

    for key in $(ini_get_section_keys "$file" "ports"); do
        local val
        val=$(ini_get "$file" "ports" "$key")
        [[ -n "$val" ]] && ports="$ports $val"
    done

    echo "$ports"
}

# -- LD_LIBRARY_PATH builder ------------------------------------------------

# Description: Construct the LD_LIBRARY_PATH for OmniEdge runtime or build
# Args: none (uses environment variables)
# Returns: prints the colon-separated library path string
#
# Centralises the 8-10 line path construction duplicated across
# run_all.sh, scripts/integration/run_all.sh, and tests/run_tests.sh.
build_ld_library_path() {
    local python_site
    python_site="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null \
        || echo "$HOME/.local/lib/python3.12/site-packages")"

    local parts=(
        "/usr/lib/x86_64-linux-gnu"
        "/usr/local/lib"
        "/usr/local/cuda/lib64"
        "/usr/local/onnxruntime/lib"
        "${python_site}/tensorrt_llm/libs"
        "${python_site}/torch/lib"
        "${python_site}/nvidia/nccl/lib"
        "${python_site}/tensorrt_libs"
    )

    # Join with colons, append existing LD_LIBRARY_PATH if set
    local result=""
    for p in "${parts[@]}"; do
        result="${result:+${result}:}${p}"
    done
    [[ -n "${LD_LIBRARY_PATH:-}" ]] && result="${result}:${LD_LIBRARY_PATH}"

    echo "$result"
}

# -- Auto-install helpers ---------------------------------------------------

# Description: Check if a command exists; if not, attempt apt-get install
# Args: $1 - command name, $2 - apt package name (optional, defaults to $1)
# Returns: 0 if command available (possibly after install), 1 if still missing
ensure_command() {
    local cmd="$1" pkg="${2:-$1}"

    if command -v "$cmd" &>/dev/null; then
        return 0
    fi

    log_warn "$cmd not found — attempting: sudo apt-get install -y $pkg"
    sudo apt-get install -y "$pkg" 2>/dev/null || true

    if command -v "$cmd" &>/dev/null; then
        log_info "$cmd installed successfully"
        return 0
    fi

    log_error "$cmd still not available after install attempt"
    return 1
}

# Description: Check if a Python module is importable; if not, attempt pip install
# Args: $1 - module import name, $2 - pip package name (optional, defaults to $1)
# Returns: 0 if importable (possibly after install), 1 if still missing
ensure_python_module() {
    local import_name="$1" pip_name="${2:-$1}"

    if python3 -c "import $import_name" 2>/dev/null; then
        return 0
    fi

    log_warn "$import_name not found — attempting: pip install $pip_name"
    pip install "$pip_name" 2>/dev/null || true

    if python3 -c "import $import_name" 2>/dev/null; then
        log_info "$import_name installed successfully"
        return 0
    fi

    log_error "$import_name still not importable after install attempt"
    return 1
}

# Description: Run a check; if it fails and OE_AUTO_INSTALL=1, run installer then retry
# Args: $1 - check function name, $2 - install function name, $3 - description
# Returns: 0 if check passes (possibly after install+retry), 1 if still failing
#
# The check function should return 0 on success, non-zero on failure.
# The install function is only called when OE_AUTO_INSTALL=1.
with_auto_install() {
    local check_fn="$1" install_fn="$2" desc="$3"

    if "$check_fn"; then
        return 0
    fi

    if [[ "${OE_AUTO_INSTALL:-0}" != "1" ]]; then
        log_error "$desc: check failed (auto-install disabled)"
        return 1
    fi

    log_warn "$desc: check failed — running installer..."
    "$install_fn" || true

    log_info "$desc: retrying check..."
    if "$check_fn"; then
        log_info "$desc: check passed after install"
        return 0
    fi

    log_error "$desc: still failing after install"
    return 1
}

# -- Networking helpers -----------------------------------------------------

# Description: Wait for a TCP port to start listening
# Args: $1 - port number, $2 - timeout in seconds (default: 10)
# Returns: 0 if port is listening, 1 on timeout
wait_for_port() {
    local port="$1" timeout="${2:-10}"
    local elapsed=0

    while (( elapsed < timeout )); do
        if { command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":${port} "; } ||
           { command -v netstat &>/dev/null && netstat -tlnp 2>/dev/null | grep -q ":${port} "; }; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1
}

# -- ctest output parser ----------------------------------------------------

# Description: Parse ctest output to extract failed and total test counts
# Args: $1 - ctest output string
# Sets globals: _CTEST_FAILED, _CTEST_RAN
#
# Parses patterns like "2 tests failed out of 10" from ctest summary output.
# Falls back to 0 if pattern not found.
parse_ctest_summary() {
    local output="$1"
    _CTEST_FAILED=$(echo "$output" | grep -oP '\K\d+(?= tests failed)' || echo "0")
    _CTEST_RAN=$(echo "$output" | grep -oP 'out of \K\d+' || echo "0")
    [[ "$_CTEST_FAILED" =~ ^[0-9]+$ ]] || _CTEST_FAILED=0
    [[ "$_CTEST_RAN" =~ ^[0-9]+$ ]] || _CTEST_RAN=0
}

# -- HuggingFace download ---------------------------------------------------

# Description: Idempotent download from HuggingFace Hub
# Args: $1 - repo ID (e.g. "Qwen/Qwen2.5-7B"), $2 - local directory, $3 - description
# Returns: 0 on success, 1 on failure
download_hf() {
    local REPO="$1" LOCAL_DIR="$2" DESC="$3"

    if dir_exists "$LOCAL_DIR"; then
        echo -e "  ${GREEN}EXISTS${NC}  $DESC → $LOCAL_DIR"
        return 0
    fi

    echo -e "  ${CYAN}DOWNLOAD${NC}  $DESC (~$(basename "$REPO"))"
    mkdir -p "$LOCAL_DIR"
    if huggingface-cli download "$REPO" --local-dir "$LOCAL_DIR"; then
        echo -e "  ${GREEN}OK${NC}      $DESC"
    else
        echo -e "  ${RED}FAIL${NC}    $DESC — check network or repo name"
        return 1
    fi
}

# -- Grep match counter (safe for pipelines under set -e) -------------------

# Description: Count lines matching a pattern in a file (safe under set -e)
# Args: $1 - grep pattern, $2 - file path
# Returns: prints the match count (0 if file missing or no matches)
#
# Needed because `grep -c` returns exit code 1 when there are zero
# matches, which would abort the script under `set -e`.
count_matches() {
    local pattern="$1" file_path="$2"
    local count="0"

    if [[ -f "$file_path" ]]; then
        count=$(grep -c -- "$pattern" "$file_path" 2>/dev/null || true)
        count=$(printf '%s\n' "$count" | tail -n1)
    fi

    [[ "$count" =~ ^[0-9]+$ ]] || count="0"
    echo "$count"
}
