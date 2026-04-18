#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Beauty Profile Launcher
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/scripts/runtime/launch_profile.sh" \
    beauty \
    "bash run_beautymode.sh" \
    "$@"
