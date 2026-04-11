# =========================================================================
# modules/nodes/stt — STT node (wires core STT + transport)
#
# Targets:
#   omniedge_stt_common — STATIC lib (for tests)
#   omniedge_stt        — executable
# =========================================================================

oe_add_node_common(omniedge_stt_common
    SOURCES   src/stt_node.cpp
    CORE_LIB  omniedge_core_stt
)

oe_add_node_exe(omniedge_stt
    SOURCES    src/main.cpp
               ${CMAKE_SOURCE_DIR}/modules/core/stt/src/trt_whisper_inferencer.cpp
    COMMON_LIB omniedge_stt_common
)

# Optional real inferencer dependencies
if(OE_TENSORRT_FOUND)
    target_include_directories(omniedge_stt PRIVATE ${OE_TENSORRT_INCLUDE_DIR})
    target_link_libraries(omniedge_stt PRIVATE ${OE_TENSORRT_LIB})
endif()
