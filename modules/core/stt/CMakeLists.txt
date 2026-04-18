# =========================================================================
# Section 1 — Comment Header
# modules/core/stt — Mel spectrogram, hallucination filter (CPU-safe)
#
# Target: omniedge_core_stt (STATIC)
# Note: trt_whisper_inferencer.cpp is NOT included — it requires TensorRT
#       and is compiled directly by targets that need it.
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_core_stt STATIC
    src/mel_spectrogram.cpp
    src/hallucination_filter.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_core_stt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_core_stt PUBLIC
    omniedge_interfaces
    omniedge_common
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_stt)
