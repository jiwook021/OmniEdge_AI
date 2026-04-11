# =========================================================================
# modules/nodes/ingest — Video + Audio ingest nodes
#
# Targets:
#   omniedge_ingest_common — STATIC lib (for tests)
#   omniedge_video_ingest  — executable
#   omniedge_audio_ingest  — executable
# =========================================================================

oe_add_node_common(omniedge_ingest_common
    SOURCES   src/video_ingest_node.cpp
              src/audio_ingest_node.cpp
    CORE_LIB  omniedge_core_ingest
)

oe_add_node_exe(omniedge_video_ingest
    SOURCES    src/main_video.cpp
    COMMON_LIB omniedge_ingest_common
)

oe_add_node_exe(omniedge_audio_ingest
    SOURCES    src/main_audio.cpp
    COMMON_LIB omniedge_ingest_common
)
