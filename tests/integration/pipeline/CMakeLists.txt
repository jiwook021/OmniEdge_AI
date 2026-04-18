# =========================================================================
# tests/integration/pipeline — Cross-module pipeline end-to-end tests
#
# These tests feed real video/audio files through the actual ingest nodes,
# verifying the full GStreamer → SHM circular buffer pipeline path.
#
# Prerequisites:
#   - GStreamer runtime (gst-launch-1.0, decodebin, videoconvert, wavparse)
#   - ffmpeg (for on-the-fly test media generation)
#   - models/silero_vad.onnx (for audio pipeline VAD)
#
# Generate test media:
#   bash tests/integration/pipeline/download_test_media.sh
# =========================================================================

# --- Video Pipeline E2E ---------------------------------------------------

oe_add_test(test_video_pipeline_e2e
    SOURCES         test_video_pipeline_e2e.cpp
    LINK_LIBRARIES  omniedge_common omniedge_ingest_common
    RESOURCE_LOCK   "gstreamer_pipeline_e2e"
    LABELS          "integration;pipeline"
)

# --- Audio Pipeline E2E ---------------------------------------------------

oe_add_test(test_audio_pipeline_e2e
    SOURCES         test_audio_pipeline_e2e.cpp
    LINK_LIBRARIES  omniedge_common omniedge_ingest_common
    RESOURCE_LOCK   "gstreamer_pipeline_e2e"
    LABELS          "integration;pipeline"
)
