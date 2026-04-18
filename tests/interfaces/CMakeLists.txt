# =========================================================================
# tests/interfaces — Contract tests (mock + real impl)
#
# Link: mocks + interfaces + omniedge_common (for shared types)
# Aggregates subdirectories in alphabetical order.
# =========================================================================

# =========================================================================
# Subdirectories
# =========================================================================
add_subdirectory(cv)
add_subdirectory(stt)
add_subdirectory(tts)
add_subdirectory(ws_bridge)

# =========================================================================
# Cross-interface edge case tests
# =========================================================================
oe_add_test(test_interface_edge_cases
    SOURCES         test_interface_edge_cases.cpp
    LINK_LIBRARIES  omniedge_interfaces omniedge_common
)
