# =========================================================================
# modules/nodes/tts — TTS node (Kokoro ONNX)
#
# Targets:
#   omniedge_tts_common — STATIC lib (for tests)
#   omniedge_tts        — executable
# =========================================================================

oe_add_node_common(omniedge_tts_common
    SOURCES   src/tts_node.cpp
    CORE_LIB  omniedge_core_tts
)

oe_add_node_exe(omniedge_tts
    SOURCES    src/main.cpp
               ${CMAKE_SOURCE_DIR}/modules/core/tts/src/onnx_kokoro_inferencer.cpp
    COMMON_LIB omniedge_tts_common
)

# Optional real inferencer dependencies (ONNX Runtime + espeak-ng)
if(OE_ONNXRUNTIME_FOUND)
    target_include_directories(omniedge_tts PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_tts PRIVATE ${OE_ONNXRUNTIME_LIB})
endif()
if(OE_ESPEAK_FOUND)
    target_include_directories(omniedge_tts PRIVATE ${OE_ESPEAK_INCLUDE_DIR})
    target_link_libraries(omniedge_tts PRIVATE ${OE_ESPEAK_LIB})
endif()
