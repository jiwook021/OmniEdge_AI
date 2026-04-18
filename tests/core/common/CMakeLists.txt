# tests/core/common — Logger + UI action enum tests

oe_add_test(test_oe_logger
    SOURCES         test_oe_logger.cpp
    LINK_LIBRARIES  omniedge_logger
)

oe_add_test(test_ui_action
    SOURCES         test_ui_action.cpp
    LINK_LIBRARIES  omniedge_core_statemachine omniedge_interfaces
)
