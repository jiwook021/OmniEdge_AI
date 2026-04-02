# =========================================================================
# Section 1 — Comment Header
# modules/interfaces — Pure virtual inferencer headers (INTERFACE only)
#
# Target: omniedge_interfaces (INTERFACE)
# Include: #include "stt_inferencer.hpp", "conversation_inferencer.hpp", etc.
#
# No oe_set_target_warnings() — INTERFACE library has no compilation unit.
# =========================================================================

# =========================================================================
# Section 2 — Library declaration + includes + link
# =========================================================================
add_library(omniedge_interfaces INTERFACE)

target_include_directories(omniedge_interfaces INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(omniedge_interfaces INTERFACE tl::expected)
