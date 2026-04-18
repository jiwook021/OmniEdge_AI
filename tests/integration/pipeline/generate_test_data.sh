#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Model Test Data Generator
#
# Pre-processes test datasets into raw formats for the headless pipeline test:
#   Audio:  16 kHz, 16-bit signed LE, mono raw PCM  (.raw)
#   Video:  BGR24 raw binary blobs                   (.bgr24)
#
# Requirements: ffmpeg, espeak-ng (optional, for TTS-generated audio)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import shared colour variables (RED, GREEN, CYAN, BOLD, NC) and ok() from
# common.sh.  Local skip()/fail() are kept because they use a two-arg "label
# — reason" format that differs from the install helpers in common.sh.
source "$SCRIPT_DIR/../../../scripts/integration/common.sh"

DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

skip() { echo -e "  ${CYAN}SKIP${NC}  $1 — $2"; }
fail() { echo -e "  ${RED}FAIL${NC}  $1 — $2"; }

echo -e "${BOLD}=== Generating Model Test Data ===${NC}"
echo ""

# ── Audio test data ────────────────────────────────────────────────────────
echo -e "${BOLD}Audio (16 kHz, 16-bit signed LE, mono):${NC}"

# Fallback chain: espeak-ng+ffmpeg (best) -> ffmpeg sine tone -> fail
generate_speech_audio() {
    local text="$1"
    local output="$2"
    local label="$3"

    if [ -f "$output" ]; then
        ok "$label (already exists)"
        return 0
    fi

    if command -v espeak-ng &>/dev/null && command -v ffmpeg &>/dev/null; then
        # Generate speech via espeak-ng, pipe through ffmpeg for format conversion
        local tmpwav
        tmpwav=$(mktemp /tmp/oe_testgen_XXXXXX.wav)
        espeak-ng -v en -s 150 -w "$tmpwav" "$text" 2>/dev/null
        ffmpeg -y -i "$tmpwav" -ar 16000 -ac 1 -f s16le -acodec pcm_s16le "$output" 2>/dev/null
        rm -f "$tmpwav"
        ok "$label (espeak-ng + ffmpeg)"
    elif command -v ffmpeg &>/dev/null; then
        # Generate a 2-second sine wave as fallback test audio
        ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" \
            -ar 16000 -ac 1 -f s16le -acodec pcm_s16le "$output" 2>/dev/null
        ok "$label (sine tone fallback — no espeak-ng)"
    else
        fail "$label" "ffmpeg not found"
        return 1
    fi
}

generate_speech_audio "Hello world." \
    "$DATA_DIR/test_hello_world.raw" \
    "test_hello_world.raw"

generate_speech_audio "Turn on the lights." \
    "$DATA_DIR/test_turn_on_lights.raw" \
    "test_turn_on_lights.raw"

# Silence: 2 seconds of zeros at 16 kHz 16-bit mono = 64000 bytes
SILENCE_FILE="$DATA_DIR/test_silence_2s.raw"
if [ ! -f "$SILENCE_FILE" ]; then
    dd if=/dev/zero of="$SILENCE_FILE" bs=1 count=64000 2>/dev/null
    ok "test_silence_2s.raw (2s silence)"
else
    ok "test_silence_2s.raw (already exists)"
fi

echo ""

# ── Video test data ────────────────────────────────────────────────────────
echo -e "${BOLD}Video (BGR24 raw, 640x480):${NC}"

# Fallback chain: ffmpeg (SMPTE/solid) -> python gradient -> dd zero-fill
generate_test_frame() {
    local output="$1"
    local label="$2"
    local pattern="$3"  # "person" or "empty"
    local width=640
    local height=480
    local frame_bytes=$((width * height * 3))

    if [ -f "$output" ]; then
        ok "$label (already exists)"
        return 0
    fi

    if command -v ffmpeg &>/dev/null; then
        if [ "$pattern" = "person" ]; then
            # Generate a test pattern with a bright rectangle simulating a person
            # smptebars has identifiable colour regions; models can attempt detection
            ffmpeg -y -f lavfi -i "smptebars=size=${width}x${height}:duration=1:rate=1" \
                -vframes 1 -f rawvideo -pix_fmt bgr24 "$output" 2>/dev/null
            ok "$label (SMPTE bars — person proxy)"
        else
            # Solid dark grey frame — no person content
            ffmpeg -y -f lavfi -i "color=c=0x333333:s=${width}x${height}:d=1:r=1" \
                -vframes 1 -f rawvideo -pix_fmt bgr24 "$output" 2>/dev/null
            ok "$label (solid grey — empty room proxy)"
        fi
    else
        # Raw binary fallback: fill with gradient pattern
        python3 -c "
import sys
w, h = $width, $height
frame = bytearray(w * h * 3)
for y in range(h):
    for x in range(w):
        idx = (y * w + x) * 3
        frame[idx]   = (x * 255 // w) & 0xFF  # B
        frame[idx+1] = (y * 255 // h) & 0xFF  # G
        frame[idx+2] = 128                     # R
sys.stdout.buffer.write(frame)
" > "$output" 2>/dev/null && ok "$label (python gradient fallback)" \
        || { dd if=/dev/zero of="$output" bs=1 count=$frame_bytes 2>/dev/null; ok "$label (zero-fill fallback)"; }
    fi
}

generate_test_frame \
    "$DATA_DIR/test_person_640x480.bgr24" \
    "test_person_640x480.bgr24" \
    "person"

generate_test_frame \
    "$DATA_DIR/test_empty_room_640x480.bgr24" \
    "test_empty_room_640x480.bgr24" \
    "empty"

echo ""

# ── Summary ────────────────────────────────────────────────────────────────
echo -e "${BOLD}=== Test Data Summary ===${NC}"
echo ""
if [ -d "$DATA_DIR" ]; then
    local_count=$(find "$DATA_DIR" -type f | wc -l)
    echo -e "  Files generated: ${GREEN}${local_count}${NC} in $DATA_DIR/"
    echo ""
    ls -lhS "$DATA_DIR/" 2>/dev/null | tail -n +2 | sed 's/^/  /'
fi
echo ""
echo -e "${GREEN}Done.${NC} Test data ready for: bash run_all.sh --model-test"
