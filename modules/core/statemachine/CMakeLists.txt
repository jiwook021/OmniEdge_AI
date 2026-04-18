# =========================================================================
# Section 1 — Comment Header
# modules/core/statemachine — FSM, prompt assembler, session persistence
#
# Target: omniedge_core_statemachine (STATIC)
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_core_statemachine STATIC
    src/state_machine.cpp
    src/prompt_assembler.cpp
    src/session_persistence.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_core_statemachine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_core_statemachine PUBLIC
    omniedge_logger
    omniedge_transport_zmq
    nlohmann_json::nlohmann_json
    tl::expected
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_statemachine)
