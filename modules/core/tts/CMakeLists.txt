# =========================================================================
# Section 1 — Comment Header
# modules/core/tts — Sentence splitter (CPU-safe)
#
# Target: omniedge_core_tts (STATIC)
# Note: onnx_kokoro_inferencer.cpp is NOT included — it requires ONNX Runtime
#       and espeak-ng, and is compiled directly by targets that need it.
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_core_tts STATIC
    src/sentence_splitter.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_core_tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_core_tts PUBLIC
    omniedge_interfaces
    omniedge_common
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_tts)
