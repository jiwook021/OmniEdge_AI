#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Conversation Profile Launcher
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/scripts/runtime/launch_profile.sh" \
    conversation \
    "bash run_conversation.sh" \
    "$@"
