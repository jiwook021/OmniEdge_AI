# =========================================================================
# modules/nodes/audio_denoise — Audio denoise node
#
# Targets:
#   omniedge_audio_denoise_common — STATIC lib (for tests)
#   omniedge_audio_denoise        — executable
# =========================================================================

oe_add_node_common(omniedge_audio_denoise_common
    SOURCES   src/audio_denoise_node.cpp
    CORE_LIB  omniedge_core_audio_denoise
)

# Real inferencer requires ONNX Runtime — build with real or skip
if(OE_ONNXRUNTIME_FOUND)
    oe_add_node_exe(omniedge_audio_denoise
        SOURCES    src/main.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/audio_denoise/src/onnx_dtln_inferencer.cpp
        COMMON_LIB omniedge_audio_denoise_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB}
    )
    target_include_directories(omniedge_audio_denoise PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(STATUS "[OmniEdge] ONNX Runtime not found — omniedge_audio_denoise skipped")
endif()
