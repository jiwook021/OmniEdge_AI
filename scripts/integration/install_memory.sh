#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Memory safety utilities for install scripts
#
# Provides: available_memory_gb, total_ram_gb, safe_job_count,
#           require_memory_gb, ensure_swap_for_step
# ═══════════════════════════════════════════════════════════════════════════════
[[ -n "${_OE_INSTALL_MEMORY_SOURCED:-}" ]] && return 0
readonly _OE_INSTALL_MEMORY_SOURCED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Returns total available memory (RAM + swap) in GB.
available_memory_gb() {
    local mem_avail_kb swap_free_kb
    mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
    swap_free_kb=$(awk '/SwapFree/{print $2}' /proc/meminfo 2>/dev/null)
    echo $(( (${mem_avail_kb:-0} + ${swap_free_kb:-0}) / 1048576 ))
}

# Returns total RAM in GB.
total_ram_gb() {
    awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo
}

# Calculates safe parallel job count for compilation.
# Rule: ~1.5 GB per CUDA/TRT translation unit.
safe_job_count() {
    local avail_gb
    avail_gb=$(available_memory_gb)
    local max_by_mem=$(( avail_gb * 2 / 3 ))  # leave 1/3 for OS + GPU driver
    local max_by_cpu
    max_by_cpu=$(nproc)
    local jobs=$(( max_by_mem < max_by_cpu ? max_by_mem : max_by_cpu ))
    # Clamp to [1, nproc]
    if [ "$jobs" -lt 1 ]; then jobs=1; fi
    if [ "$jobs" -gt "$max_by_cpu" ]; then jobs="$max_by_cpu"; fi
    echo "$jobs"
}

# Guard: check if enough memory is available for a given step.
# Usage: require_memory_gb <needed_gb> <step_name>
# Returns 0 if OK, 1 if insufficient (and prints warning).
require_memory_gb() {
    local needed="$1"
    local step_name="$2"
    local avail
    avail=$(available_memory_gb)

    if [ "${FORCE_ALL:-0}" -eq 1 ]; then
        if [ "$avail" -lt "$needed" ]; then
            warn "${step_name} needs ~${needed} GB but only ${avail} GB available (--force-all overriding)"
        fi
        return 0
    fi

    if [ "$avail" -lt "$needed" ]; then
        record_error "${step_name} needs ~${needed} GB RAM+swap but only ${avail} GB available. Skipping to prevent OOM crash."
        echo -e "    ${YELLOW}Fix: Add swap (sudo fallocate -l 16G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile)${NC}"
        echo -e "    ${YELLOW}  or pass --force-all to override${NC}"
        return 1
    fi
    return 0
}

# Temporarily expand swap if RAM is below threshold.
# Creates a swap file that is cleaned up on exit via the trap.
ensure_swap_for_step() {
    # Swap management not applicable inside Docker containers
    [[ -f /.dockerenv ]] && return 0

    local needed_gb="$1"
    local avail
    avail=$(available_memory_gb)

    if [ "$avail" -ge "$needed_gb" ]; then
        return 0  # already have enough
    fi

    local deficit=$(( needed_gb - avail + 2 ))  # +2 GB headroom
    local swap_file="/tmp/oe_install_swap"

    # Don't create if we already have a temp swap or if not root-capable
    if [ -n "$OE_TEMP_SWAP" ] && [ -f "$OE_TEMP_SWAP" ]; then
        return 0  # already created
    fi

    echo -e "  ${CYAN}Creating temporary ${deficit}G swap file to prevent OOM...${NC}"
    if sudo fallocate -l "${deficit}G" "$swap_file" 2>/dev/null \
       && sudo chmod 600 "$swap_file" \
       && sudo mkswap "$swap_file" >/dev/null 2>&1 \
       && sudo swapon "$swap_file" 2>/dev/null; then
        OE_TEMP_SWAP="$swap_file"
        ok "Temporary swap: +${deficit}G (will be removed on exit)"
        return 0
    else
        warn "Could not create temporary swap — proceeding with available memory"
        rm -f "$swap_file" 2>/dev/null || true
        return 0  # non-fatal: let the memory guard catch it
    fi
}
