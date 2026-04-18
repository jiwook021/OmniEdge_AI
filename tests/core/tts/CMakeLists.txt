# tests/core/tts — Sentence splitter (CPU-only, no mocks)

oe_add_test(test_sentence_splitter
    SOURCES         test_sentence_splitter.cpp
    LINK_LIBRARIES  omniedge_core_tts omniedge_interfaces
)
