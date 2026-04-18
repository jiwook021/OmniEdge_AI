# tests/nodes/stt — STT node wiring test with mock inferencer

oe_add_test(test_stt_node
    SOURCES         test_stt_node.cpp
    LINK_LIBRARIES  omniedge_stt_common
    RESOURCE_LOCK   zmq_test_ports
)
