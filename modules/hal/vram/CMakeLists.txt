# =========================================================================
# Section 1 — Comment Header
# modules/hal/vram — VRAM tracking, gating, thresholds
#
# Target: omniedge_hal_vram (STATIC)
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_hal_vram STATIC
    src/vram_gate.cpp
    src/vram_tracker.cpp
    src/priority_scheduler.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_hal_vram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_hal_vram PUBLIC
    omniedge_hal_gpu
    omniedge_logger
    tl::expected
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings_no_error(omniedge_hal_vram)
