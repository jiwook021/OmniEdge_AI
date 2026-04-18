# tests/interfaces/tts — TTSInferencer contract (degraded mode / error recovery)

oe_add_test(test_tts_degraded_inferencer
    SOURCES         test_tts_degraded_inferencer.cpp
    LINK_LIBRARIES  omniedge_tts_common
    RESOURCE_LOCK   zmq_test_ports
)
