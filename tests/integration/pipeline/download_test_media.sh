#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI -- Download Real Test Media for Pipeline Integration Tests
#
# Downloads public domain test video (Big Buck Bunny) at multiple resolutions
# and extracts audio track for end-to-end pipeline testing.
#
# Requirements: ffmpeg, curl or wget
#
# Test matrix:
#   1080p60  (1920x1080 @ 60fps)  -- stress test: max throughput
#   1080p30  (1920x1080 @ 30fps)  -- standard production config
#   720p30   ( 1280x720 @ 30fps)  -- reduced resolution baseline
#   Audio    (16 kHz mono WAV)    -- extracted from source video
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../../scripts/integration/common.sh
source "$SCRIPT_DIR/../../../scripts/integration/common.sh"

DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

# Local fail() that exits (common.sh record_error does not exit).
fail() { echo -e "  ${RED}FAIL${NC}  $1 -- $2"; exit 1; }

echo -e "${BOLD}=== Downloading Pipeline Test Media ===${NC}"
echo ""

# -- Prerequisite check -------------------------------------------------------

if ! command -v ffmpeg &>/dev/null; then
    fail "ffmpeg" "ffmpeg is required but not found"
fi

download() {
    local url="$1"
    local dest="$2"
    if command -v curl &>/dev/null; then
        curl -fsSL -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget -q -O "$dest" "$url"
    else
        fail "downloader" "neither curl nor wget found"
    fi
}

# Description: Transcode a source video clip to a given fps and resolution.
# Args: $1 - input file, $2 - output file, $3 - fps, $4 - scale (WxH),
#        $5 - human-readable label for status output
# Skips if the output file already exists.
transcode_clip() {
    local input="$1" output="$2" fps="$3" scale="$4" label="$5"
    if [ -f "$output" ]; then
        ok "$label (already exists)"
        return 0
    fi
    ffmpeg -y -i "$input" -t 10 -vf "fps=${fps},scale=${scale}" \
        -c:v libx264 -preset ultrafast -crf 18 \
        -an \
        "$output" 2>/dev/null
    if [ -f "$output" ]; then
        ok "$label"
    else
        fail "$label" "ffmpeg conversion failed"
    fi
}

# -- Source video: Big Buck Bunny 1080p60 (Creative Commons, ~6 MB 10s clip) --
# We download the full 1080p60 MP4 and then trim to 10 seconds to keep test
# artifacts small while still exercising the full decode pipeline.

SOURCE_URL="https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4"
SOURCE_FILE="$DATA_DIR/big_buck_bunny_source.mp4"

echo -e "${BOLD}Source video:${NC}"
if [ -f "$SOURCE_FILE" ]; then
    ok "big_buck_bunny_source.mp4 (already exists)"
else
    echo -n "  Downloading Big Buck Bunny 1080p source... "
    if download "$SOURCE_URL" "$SOURCE_FILE" 2>/dev/null; then
        ok "big_buck_bunny_source.mp4"
    else
        # Fallback: generate a synthetic 10s 1080p60 test video with ffmpeg
        echo ""
        echo -e "  ${CYAN}INFO${NC}  Download failed, generating synthetic test video..."
        ffmpeg -y -f lavfi \
            -i "testsrc2=size=1920x1080:rate=60:duration=10" \
            -c:v libx264 -preset ultrafast -crf 18 \
            -f lavfi -i "sine=frequency=440:duration=10:sample_rate=16000" \
            -c:a aac -b:a 64k \
            -shortest \
            "$SOURCE_FILE" 2>/dev/null
        ok "big_buck_bunny_source.mp4 (synthetic fallback)"
    fi
fi

# -- 1080p60 test clip (10 seconds) -------------------------------------------

echo ""
echo -e "${BOLD}Video test clips:${NC}"

transcode_clip "$SOURCE_FILE" "$DATA_DIR/test_1080p60.mp4" 60 "1920:1080" \
    "test_1080p60.mp4 (1920x1080 @ 60fps, 10s)"

transcode_clip "$SOURCE_FILE" "$DATA_DIR/test_1080p30.mp4" 30 "1920:1080" \
    "test_1080p30.mp4 (1920x1080 @ 30fps, 10s)"

transcode_clip "$SOURCE_FILE" "$DATA_DIR/test_720p30.mp4"  30 "1280:720" \
    "test_720p30.mp4 (1280x720 @ 30fps, 10s)"

# -- Audio extraction: 16 kHz mono WAV ----------------------------------------

echo ""
echo -e "${BOLD}Audio test clip:${NC}"

AUDIO_WAV="$DATA_DIR/test_audio_16k.wav"
if [ -f "$AUDIO_WAV" ]; then
    ok "test_audio_16k.wav (already exists)"
else
    # Try to extract audio from source video first; fall back to sine tone
    if ffprobe -i "$SOURCE_FILE" -show_streams -select_streams a 2>/dev/null | grep -q "codec_type=audio"; then
        ffmpeg -y -i "$SOURCE_FILE" \
            -t 10 \
            -vn \
            -ar 16000 -ac 1 \
            "$AUDIO_WAV" 2>/dev/null
        ok "test_audio_16k.wav (extracted from source, 16kHz mono)"
    else
        # Source has no audio track -- generate synthetic speech-like audio
        ffmpeg -y -f lavfi \
            -i "sine=frequency=300:duration=10:sample_rate=16000" \
            -ar 16000 -ac 1 \
            "$AUDIO_WAV" 2>/dev/null
        ok "test_audio_16k.wav (synthetic sine 300Hz, 16kHz mono)"
    fi
fi

# -- Summary -------------------------------------------------------------------

echo ""
echo -e "${BOLD}=== Test Media Summary ===${NC}"
echo ""
if [ -d "$DATA_DIR" ]; then
    count=$(find "$DATA_DIR" -name "test_*" -type f | wc -l)
    echo -e "  Files: ${GREEN}${count}${NC} in $DATA_DIR/"
    echo ""
    ls -lhS "$DATA_DIR"/test_* 2>/dev/null | sed 's/^/  /'
fi
echo ""
echo -e "${GREEN}Done.${NC} Test media ready for pipeline integration tests."
