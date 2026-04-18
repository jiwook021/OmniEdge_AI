# =========================================================================
# modules/nodes/ingest — Video + Audio + Screen ingest nodes
#
# Targets:
#   omniedge_ingest_common  — STATIC lib (for tests)
#   omniedge_video_ingest   — executable
#   omniedge_audio_ingest   — executable
#   omniedge_screen_ingest  — executable
# =========================================================================

oe_add_node_common(omniedge_ingest_common
    SOURCES   src/video_ingest_node.cpp
              src/audio_ingest_node.cpp
              src/screen_ingest_node.cpp
    CORE_LIB  omniedge_core_ingest
)

# Screen ingest needs OpenCV for JPEG decoding
find_package(OpenCV REQUIRED COMPONENTS imgcodecs imgproc)
target_link_libraries(omniedge_ingest_common PUBLIC ${OpenCV_LIBS})

oe_add_node_exe(omniedge_video_ingest
    SOURCES    src/main_video.cpp
    COMMON_LIB omniedge_ingest_common
)

oe_add_node_exe(omniedge_audio_ingest
    SOURCES    src/main_audio.cpp
    COMMON_LIB omniedge_ingest_common
)

oe_add_node_exe(omniedge_screen_ingest
    SOURCES    src/main_screen.cpp
    COMMON_LIB omniedge_ingest_common
)
