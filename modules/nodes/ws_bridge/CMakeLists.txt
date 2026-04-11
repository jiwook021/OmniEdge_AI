# =========================================================================
# modules/nodes/ws_bridge — WebSocket bridge node
#
# Targets:
#   omniedge_ws_bridge_common — STATIC lib (for tests)
#   omniedge_ws_bridge        — executable
# Optional deps: libjpeg (JPEG decoding for face registration relay)
# =========================================================================

oe_add_node_common(omniedge_ws_bridge_common
    SOURCES   src/websocket_bridge_node.cpp
    CORE_LIB  omniedge_transport_ws
)

find_package(JPEG QUIET)
if(JPEG_FOUND)
    target_link_libraries(omniedge_ws_bridge_common PUBLIC JPEG::JPEG)
endif()

oe_add_node_exe(omniedge_ws_bridge
    SOURCES    src/main.cpp
    COMMON_LIB omniedge_ws_bridge_common
)
