# tests/nodes/tts — TTS node wiring test with mock inferencer

oe_add_test(test_tts_node
    SOURCES         test_tts_node.cpp
    LINK_LIBRARIES  omniedge_tts_common
    RESOURCE_LOCK   zmq_test_ports
)
