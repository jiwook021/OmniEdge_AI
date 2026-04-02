# =========================================================================
# Section 1 — Comment Header
# modules/nodes/audio_denoise — Audio denoise node
#
# Targets:
#   omniedge_audio_denoise_common — STATIC lib (for tests)
#   omniedge_audio_denoise        — executable
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_audio_denoise_common STATIC src/audio_denoise_node.cpp)

target_include_directories(omniedge_audio_denoise_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_audio_denoise_common PUBLIC
    omniedge_core_audio_denoise
    omniedge_common
)
oe_set_target_warnings(omniedge_audio_denoise_common)

# =========================================================================
# Section 3 — Executable (compiles the real ONNX DTLN inferencer)
# =========================================================================
add_executable(omniedge_audio_denoise
    src/main.cpp
    ${CMAKE_SOURCE_DIR}/modules/core/audio_denoise/src/onnx_dtln_inferencer.cpp
)
target_link_libraries(omniedge_audio_denoise PRIVATE omniedge_audio_denoise_common yaml-cpp)
oe_set_target_warnings(omniedge_audio_denoise)
oe_set_release_debug_flags(omniedge_audio_denoise)

# =========================================================================
# Section 4 — Optional real inferencer dependencies
# =========================================================================
if(OE_ONNXRUNTIME_FOUND)
    target_include_directories(omniedge_audio_denoise PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_audio_denoise PRIVATE ${OE_ONNXRUNTIME_LIB})
endif()
