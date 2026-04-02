# =========================================================================
# Section 1 — Comment Header
# modules/nodes/tts — TTS node (Kokoro ONNX)
#
# Targets:
#   omniedge_tts_common — STATIC lib (for tests)
#   omniedge_tts        — executable
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_tts_common STATIC src/tts_node.cpp)

target_include_directories(omniedge_tts_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_tts_common PUBLIC
    omniedge_core_tts
    omniedge_common
)
oe_set_target_warnings(omniedge_tts_common)

# =========================================================================
# Section 3 — Executable
# =========================================================================
add_executable(omniedge_tts src/main.cpp)
target_link_libraries(omniedge_tts PRIVATE omniedge_tts_common yaml-cpp)
oe_set_target_warnings(omniedge_tts)
oe_set_release_debug_flags(omniedge_tts)

# =========================================================================
# Section 4 — Optional real inferencer sources + dependencies
#   - ONNX Runtime + espeak-ng for real Kokoro inferencer
# =========================================================================
target_sources(omniedge_tts PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/core/tts/src/onnx_kokoro_inferencer.cpp
)

if(OE_ONNXRUNTIME_FOUND)
    target_include_directories(omniedge_tts PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_tts PRIVATE ${OE_ONNXRUNTIME_LIB})
endif()
if(OE_ESPEAK_FOUND)
    target_include_directories(omniedge_tts PRIVATE ${OE_ESPEAK_INCLUDE_DIR})
    target_link_libraries(omniedge_tts PRIVATE ${OE_ESPEAK_LIB})
endif()
