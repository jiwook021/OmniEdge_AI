# =========================================================================
# modules/nodes/conversation — Unified conversation node
#
# Targets:
#   omniedge_conversation_common — STATIC lib with stub inferencers (for tests)
#   omniedge_conversation        — executable
# =========================================================================

oe_add_node_common(omniedge_conversation_common
    SOURCES    src/conversation_node.cpp
    CORE_LIB   omniedge_core_conversation
    EXTRA_LIBS nlohmann_json::nlohmann_json yaml-cpp
)

oe_add_node_exe(omniedge_conversation
    SOURCES    src/main.cpp
    COMMON_LIB omniedge_conversation_common
)
