# =========================================================================
# Section 1 — Comment Header
# modules/nodes/ingest — Video + Audio ingest nodes
#
# Targets:
#   omniedge_ingest_common — STATIC lib (for tests)
#   omniedge_video_ingest  — executable
#   omniedge_audio_ingest  — executable
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_ingest_common STATIC
    src/video_ingest_node.cpp
    src/audio_ingest_node.cpp
)
target_include_directories(omniedge_ingest_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_ingest_common PUBLIC
    omniedge_core_ingest
    omniedge_common
)
oe_set_target_warnings(omniedge_ingest_common)

# =========================================================================
# Section 3 — Executables
# =========================================================================
add_executable(omniedge_video_ingest src/main_video.cpp)
target_link_libraries(omniedge_video_ingest PRIVATE omniedge_ingest_common yaml-cpp)
oe_set_target_warnings(omniedge_video_ingest)
oe_set_release_debug_flags(omniedge_video_ingest)

add_executable(omniedge_audio_ingest src/main_audio.cpp)
target_link_libraries(omniedge_audio_ingest PRIVATE omniedge_ingest_common yaml-cpp)
oe_set_target_warnings(omniedge_audio_ingest)
oe_set_release_debug_flags(omniedge_audio_ingest)
