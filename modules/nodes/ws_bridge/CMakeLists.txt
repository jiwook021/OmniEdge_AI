# =========================================================================
# Section 1 — Comment Header
# modules/nodes/ws_bridge — WebSocket bridge node
#
# Targets:
#   omniedge_ws_bridge_common — STATIC lib (for tests)
#   omniedge_ws_bridge        — executable
# Optional deps: libjpeg (JPEG decoding for face registration relay)
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_ws_bridge_common STATIC
    src/websocket_bridge_node.cpp
)
target_include_directories(omniedge_ws_bridge_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_ws_bridge_common PUBLIC
    omniedge_transport_ws
    omniedge_common
)

find_package(JPEG QUIET)
if(JPEG_FOUND)
    target_link_libraries(omniedge_ws_bridge_common PUBLIC JPEG::JPEG)
endif()

oe_set_target_warnings(omniedge_ws_bridge_common)

# =========================================================================
# Section 3 — Executable
# =========================================================================
add_executable(omniedge_ws_bridge src/main.cpp)
target_link_libraries(omniedge_ws_bridge PRIVATE omniedge_ws_bridge_common yaml-cpp)
oe_set_target_warnings(omniedge_ws_bridge)
oe_set_release_debug_flags(omniedge_ws_bridge)
