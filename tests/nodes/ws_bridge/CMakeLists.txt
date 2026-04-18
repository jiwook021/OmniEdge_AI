# tests/nodes/ws_bridge — WebSocket bridge node wiring test

oe_add_test(test_websocket_bridge_node
    SOURCES         test_websocket_bridge_node.cpp
    LINK_LIBRARIES  omniedge_ws_bridge_common
    RESOURCE_LOCK   zmq_ports_wsbridge
)

oe_add_test(test_frame_pacing
    SOURCES         test_frame_pacing.cpp
    LINK_LIBRARIES  omniedge_ws_bridge_common
)

oe_add_test(test_ws_bridge_relay
    SOURCES         test_ws_bridge_relay.cpp
    LINK_LIBRARIES  omniedge_ws_bridge_common
    RESOURCE_LOCK   zmq_ports_wsbridge_relay
)
