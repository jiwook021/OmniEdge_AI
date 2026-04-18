# tests/interfaces/stt — STTInferencer contract (empty output suppression)

oe_add_test(test_stt_empty_output
    SOURCES         test_stt_empty_output.cpp
    LINK_LIBRARIES  omniedge_stt_common
    RESOURCE_LOCK   zmq_test_ports
)
