#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Legacy Compatibility Entrypoint
#
# Runtime launch is profile-first:
#   bash run_conversation.sh
#   bash run_security_mode.sh
#   bash run_beautymode.sh
#
# This script is kept only for backwards compatibility and forwards utility
# flags to the conversation launcher.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_compat_help() {
    cat <<'EOF'
OmniEdge_AI — `run_all.sh` is deprecated.

Use profile launchers:
  bash run_conversation.sh [OPTION]
  bash run_security_mode.sh [OPTION]
  bash run_beautymode.sh [OPTION]

Legacy compatibility:
  `run_all.sh` forwards non-launch utility flags to `run_conversation.sh`.
EOF
}

if [[ $# -eq 0 ]]; then
    print_compat_help
    echo ""
    echo "Direct launch is blocked. Pick a profile launcher above."
    exit 2
fi

echo "[DEPRECATED] run_all.sh forwards to run_conversation.sh for compatibility."
exec bash "${PROJECT_ROOT}/run_conversation.sh" "$@"
