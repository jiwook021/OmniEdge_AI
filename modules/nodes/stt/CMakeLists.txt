# =========================================================================
# Section 1 — Comment Header
# modules/nodes/stt — STT node (wires core STT + transport)
#
# Targets:
#   omniedge_stt_common — STATIC lib (for tests)
#   omniedge_stt        — executable
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_stt_common STATIC src/stt_node.cpp)

target_include_directories(omniedge_stt_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_stt_common PUBLIC
    omniedge_core_stt
    omniedge_common
)
oe_set_target_warnings(omniedge_stt_common)

# =========================================================================
# Section 3 — Executable
# =========================================================================
add_executable(omniedge_stt
    src/main.cpp
    ${CMAKE_SOURCE_DIR}/modules/core/stt/src/trt_whisper_inferencer.cpp
)
target_link_libraries(omniedge_stt PRIVATE omniedge_stt_common yaml-cpp)
oe_set_target_warnings(omniedge_stt)
oe_set_release_debug_flags(omniedge_stt)

# =========================================================================
# Section 4 — Optional real inferencer dependencies
# =========================================================================
if(OE_TENSORRT_FOUND)
    target_include_directories(omniedge_stt PRIVATE ${OE_TENSORRT_INCLUDE_DIR})
    target_link_libraries(omniedge_stt PRIVATE ${OE_TENSORRT_LIB})
endif()
