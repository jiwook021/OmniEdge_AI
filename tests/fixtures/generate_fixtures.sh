#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Test Fixture Generator
#
# Generates test data files required by the GPU integration tests:
#   - tests/fixtures/speech_hello_16khz.pcm   (STT: float32 PCM, 16 kHz mono)
#   - tests/fixtures/test_frame_1920x1080.bgr (CV: raw BGR24, 1920x1080)
#
# Requirements: espeak-ng + ffmpeg  (for speech fixture)
#               ffmpeg OR python3   (for video fixture)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import shared colour variables (RED, GREEN, CYAN, BOLD, NC) from common.sh.
# We keep local ok()/skip() below because their signatures differ from the
# install-oriented helpers in common.sh.
source "$SCRIPT_DIR/../../scripts/integration/common.sh"

ok()   { echo -e "  ${GREEN}OK${NC}    $1"; }
skip() { echo -e "  ${CYAN}SKIP${NC}  $1 — $2"; }

echo "=== Generating Test Fixtures ==="
echo ""

# ── 1. Speech fixture: "hello" at 16 kHz mono float32 ─────────────────────
SPEECH_OUT="$SCRIPT_DIR/speech_hello_16khz.pcm"
if [ -f "$SPEECH_OUT" ]; then
    ok "speech_hello_16khz.pcm (already exists)"
else
    if command -v espeak-ng &>/dev/null && command -v ffmpeg &>/dev/null; then
        TMPWAV=$(mktemp /tmp/oe_fixture_XXXXXX.wav)
        espeak-ng -v en -s 150 -w "$TMPWAV" "Hello" 2>/dev/null
        # Convert to float32 PCM at 16 kHz mono (what Whisper expects)
        ffmpeg -y -i "$TMPWAV" -f f32le -acodec pcm_f32le -ar 16000 -ac 1 \
            "$SPEECH_OUT" 2>/dev/null
        rm -f "$TMPWAV"
        ok "speech_hello_16khz.pcm (espeak-ng → float32 PCM)"
    else
        skip "speech_hello_16khz.pcm" "requires espeak-ng + ffmpeg"
    fi
fi

# ── 2. Video fixture: 1920x1080 BGR24 test frame ──────────────────────────
VIDEO_OUT="$SCRIPT_DIR/test_frame_1920x1080.bgr"
WIDTH=1920
HEIGHT=1080
FRAME_BYTES=$((WIDTH * HEIGHT * 3))

if [ -f "$VIDEO_OUT" ]; then
    ok "test_frame_1920x1080.bgr (already exists)"
else
    if command -v ffmpeg &>/dev/null; then
        # SMPTE color bars — recognizable pattern for CV tests
        ffmpeg -y -f lavfi -i "smptebars=size=${WIDTH}x${HEIGHT}:duration=1:rate=1" \
            -vframes 1 -f rawvideo -pix_fmt bgr24 "$VIDEO_OUT" 2>/dev/null
        ok "test_frame_1920x1080.bgr (SMPTE bars via ffmpeg)"
    elif command -v python3 &>/dev/null; then
        python3 -c "
import sys
w, h = $WIDTH, $HEIGHT
frame = bytearray(w * h * 3)
for y in range(h):
    for x in range(w):
        idx = (y * w + x) * 3
        frame[idx]   = (x * 255 // w) & 0xFF  # B gradient
        frame[idx+1] = (y * 255 // h) & 0xFF  # G gradient
        frame[idx+2] = 180                     # R constant
sys.stdout.buffer.write(frame)
" > "$VIDEO_OUT"
        ok "test_frame_1920x1080.bgr (python gradient fallback)"
    else
        skip "test_frame_1920x1080.bgr" "requires ffmpeg or python3"
    fi
fi

echo ""
echo "=== Fixture Summary ==="
ls -lhS "$SCRIPT_DIR/"*.pcm "$SCRIPT_DIR/"*.bgr 2>/dev/null | sed 's/^/  /'
echo ""
echo "Done. Run GPU tests with: ctest -L gpu"
