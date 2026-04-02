# =========================================================================
# Section 1 — Comment Header
# modules/nodes/conversation — Unified conversation node
#
# Targets:
#   omniedge_conversation_common — STATIC lib with stub inferencers (for tests)
#   omniedge_conversation        — executable
# =========================================================================

# =========================================================================
# Section 2 — Common library (shared with tests)
# =========================================================================
add_library(omniedge_conversation_common STATIC
    src/conversation_node.cpp
)
target_include_directories(omniedge_conversation_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_conversation_common PUBLIC
    omniedge_core_conversation
    omniedge_common
    nlohmann_json::nlohmann_json
    yaml-cpp
)
oe_set_target_warnings(omniedge_conversation_common)

# =========================================================================
# Section 3 — Executable
#   - Uses stub inferencers (real GPU inferencers compiled separately)
# =========================================================================
add_executable(omniedge_conversation src/main.cpp)
target_link_libraries(omniedge_conversation PRIVATE omniedge_conversation_common)
oe_set_target_warnings(omniedge_conversation)
oe_set_release_debug_flags(omniedge_conversation)
