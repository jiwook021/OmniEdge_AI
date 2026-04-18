# tests/nodes/audio_denoise — Audio denoise node wiring test

oe_add_test(test_audio_denoise_node
    SOURCES         test_audio_denoise_node.cpp
    LINK_LIBRARIES  omniedge_audio_denoise_common
)
