# =========================================================================
# tests/common — ConfigValidator, ModuleNodeBase CRTP, EventBus tests
# =========================================================================

oe_add_test(test_validated_config
    SOURCES         test_validated_config.cpp
    LINK_LIBRARIES  omniedge_logger
)

oe_add_test(test_module_node_base
    SOURCES         test_module_node_base.cpp
    LINK_LIBRARIES  omniedge_common
    RESOURCE_LOCK   zmq_ports
)

oe_add_test(test_event_bus
    SOURCES         test_event_bus.cpp
    LINK_LIBRARIES  omniedge_logger
)

oe_add_test(test_subprocess_manager
    SOURCES         test_subprocess_manager.cpp
    LINK_LIBRARIES  omniedge_logger
)
