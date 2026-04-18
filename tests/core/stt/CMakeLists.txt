# tests/core/stt — Mel spectrogram + hallucination filter (CPU-only, no mocks)

oe_add_test(test_mel_spectrogram
    SOURCES         test_mel_spectrogram.cpp
    LINK_LIBRARIES  omniedge_core_stt omniedge_interfaces
)

oe_add_test(test_hallucination_filter
    SOURCES         test_hallucination_filter.cpp
    LINK_LIBRARIES  omniedge_core_stt omniedge_interfaces
)
