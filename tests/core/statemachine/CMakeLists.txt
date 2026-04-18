# tests/core/statemachine — FSM, prompt assembler, session persistence

oe_add_test(test_state_machine
    SOURCES         test_state_machine.cpp
    LINK_LIBRARIES  omniedge_core_statemachine omniedge_interfaces
)

oe_add_test(test_prompt_assembler
    SOURCES         test_prompt_assembler.cpp
    LINK_LIBRARIES  omniedge_core_statemachine omniedge_interfaces
)

oe_add_test(test_session_persistence
    SOURCES         test_session_persistence.cpp
    LINK_LIBRARIES  omniedge_core_statemachine omniedge_interfaces
    RESOURCE_LOCK   session_persistence_dir
)
