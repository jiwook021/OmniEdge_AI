# tests/interfaces/ws_bridge — WS command JSON parsing contract

oe_add_test(test_ws_command_validation
    SOURCES         test_ws_command_validation.cpp
    LINK_LIBRARIES  omniedge_ws_bridge_common
)
