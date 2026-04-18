# tests/core/pipeline_orchestrator — Unit tests for pipeline orchestrator
# Label: orchestrator_unit (CPU-only, no GPU required)

oe_add_test(test_switch_plan
    SOURCES         test_switch_plan.cpp
    LINK_LIBRARIES  omniedge_core_pipeline_orchestrator
    LABELS          orchestrator_unit
)

oe_add_test(test_crash_handler
    SOURCES         test_crash_handler.cpp
    LINK_LIBRARIES  omniedge_core_pipeline_orchestrator
    LABELS          orchestrator_unit
)

oe_add_test(test_pipeline_orchestrator_core
    SOURCES         test_pipeline_orchestrator.cpp
    LINK_LIBRARIES  omniedge_core_pipeline_orchestrator nlohmann_json::nlohmann_json
    LABELS          orchestrator_unit
)
