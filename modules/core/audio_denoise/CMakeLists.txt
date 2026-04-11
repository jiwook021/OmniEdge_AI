# =========================================================================
# Section 1 — Comment Header
# modules/core/audio_denoise — DTLN inferencer headers (INTERFACE only)
#
# Target: omniedge_core_audio_denoise (INTERFACE — headers only)
# Note: onnx_dtln_inferencer.cpp requires ONNX Runtime and is compiled
#       directly by targets that have the dependency.
#
# No oe_set_target_warnings() — INTERFACE library has no compilation unit.
# =========================================================================

# =========================================================================
# Section 2 — Library declaration + includes + link
# =========================================================================
add_library(omniedge_core_audio_denoise INTERFACE)

target_include_directories(omniedge_core_audio_denoise INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(omniedge_core_audio_denoise INTERFACE
    omniedge_interfaces
    omniedge_common
)
