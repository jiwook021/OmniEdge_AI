# =========================================================================
# modules/nodes/orchestrator — OmniEdge daemon (process lifecycle, mode
#                               switching, VRAM orchestration)
#
# Targets:
#   omniedge_orchestrator_common — STATIC lib (for tests)
#   omniedge_daemon              — executable
#
# NOTE: Does not use oe_add_node_common() because it links a custom set
# of infrastructure libraries and uses oe_set_target_warnings_no_error.
# =========================================================================

add_library(omniedge_orchestrator_common STATIC
    src/omniedge_daemon.cpp
    src/module_launcher.cpp
    src/mode_orchestrator.cpp
    src/shm_registry.cpp
)
target_include_directories(omniedge_orchestrator_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_orchestrator_common PUBLIC
    omniedge_core_statemachine
    omniedge_hal_gpu
    omniedge_hal_vram
    omniedge_common
    yaml-cpp
)
oe_set_target_warnings_no_error(omniedge_orchestrator_common)

oe_add_node_exe(omniedge_daemon
    SOURCES    src/main.cpp
    COMMON_LIB omniedge_orchestrator_common
)
