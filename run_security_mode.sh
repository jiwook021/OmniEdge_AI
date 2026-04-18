#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Security Profile Launcher
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/scripts/runtime/launch_profile.sh" \
    security \
    "bash run_security_mode.sh" \
    "$@"
