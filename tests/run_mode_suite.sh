#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Single Profile Test Runner
#
# Usage:
#   bash tests/run_mode_suite.sh conversation [extra run_tests.sh args]
#   bash tests/run_mode_suite.sh security [extra run_tests.sh args]
#   bash tests/run_mode_suite.sh beauty [extra run_tests.sh args]
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
shift || true

case "${MODE}" in
    conversation)
        echo "== Conversation mode tests =="
        ;;
    security)
        echo "== Security mode tests =="
        ;;
    beauty)
        echo "== Beauty mode tests =="
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Expected one of: conversation, security, beauty" >&2
        exit 1
        ;;
esac

cd "${PROJECT_ROOT}"
exec bash "${PROJECT_ROOT}/tests/run_tests.sh" --mode "${MODE}" "$@"
