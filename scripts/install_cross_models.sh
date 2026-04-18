#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Install Cross-Model Benchmark Variants (wrapper)
#
# Delegates to the canonical script at scripts/integration/install_cross_models.sh.
# This wrapper exists so callers can use the shorter path from the project root.
#
# Usage:
#   bash scripts/install_cross_models.sh              # Install all
#   bash scripts/install_cross_models.sh --llm        # LLM variants only
#   bash scripts/install_cross_models.sh --tts        # TTS variants only
#   bash scripts/install_cross_models.sh --cv         # CV variants only
#   bash scripts/install_cross_models.sh --no-engines # Download only, skip TRT builds
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/integration/install_cross_models.sh" "$@"
