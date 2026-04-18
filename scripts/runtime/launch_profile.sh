#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Shared Profile Launcher
#
# Usage:
#   bash scripts/runtime/launch_profile.sh <mode> <entrypoint-cmd> [args...]
#
# Modes:
#   conversation | security | beauty
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
ENTRYPOINT_CMD="${2:-}"
shift 2 || true

case "${MODE}" in
    conversation|security|beauty) ;;
    *)
        echo "Unsupported mode: ${MODE}" >&2
        echo "Expected one of: conversation, security, beauty" >&2
        exit 1
        ;;
esac

if [[ -z "${ENTRYPOINT_CMD}" ]]; then
    echo "Missing entrypoint command for mode '${MODE}'." >&2
    exit 1
fi

MODE_INI="${OE_MODE_INI:-${PROJECT_ROOT}/logs/omniedge.${MODE}.ini}"
if [[ -d "/opt/omniedge/share/frontend/${MODE}" ]]; then
    FRONTEND_DIR="/opt/omniedge/share/frontend/${MODE}"
else
    FRONTEND_DIR="${PROJECT_ROOT}/frontend/${MODE}"
fi

bash "${PROJECT_ROOT}/scripts/modes/generate_mode_ini.sh" \
    "${MODE}" "${MODE_INI}" "${FRONTEND_DIR}" >/dev/null

# Determine published HTTP port from generated mode INI ([ports] ws_http).
WS_HTTP_PORT="$(
    awk -F '=' '
        BEGIN { in_ports = 0 }
        /^[[:space:]]*\[ports\][[:space:]]*$/ { in_ports = 1; next }
        in_ports && /^[[:space:]]*\[/ { in_ports = 0 }
        in_ports && $1 ~ /^[[:space:]]*ws_http[[:space:]]*$/ {
            gsub(/[[:space:]]/, "", $2);
            print $2;
            exit;
        }
    ' "${MODE_INI}"
)"
WS_HTTP_PORT="${WS_HTTP_PORT:-9001}"

export OE_INI_FILE="${MODE_INI}"
export OE_YAML_CONFIG="${OE_YAML_CONFIG:-${PROJECT_ROOT}/config/omniedge_config.yaml}"
export OE_PROFILE_MODE="${MODE}"
export OE_PROFILE_URL="${OE_PROFILE_URL:-http://localhost:${WS_HTTP_PORT}}"
export OE_ENTRYPOINT_CMD="${OE_ENTRYPOINT_CMD:-${ENTRYPOINT_CMD}}"

exec bash "${PROJECT_ROOT}/scripts/runtime/profile_runtime.sh" "$@"
