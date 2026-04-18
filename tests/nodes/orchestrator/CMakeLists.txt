# tests/nodes/orchestrator — Daemon wiring tests

oe_add_test(test_module_launcher
    SOURCES         test_module_launcher.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

oe_add_test(test_gpu_profiler
    SOURCES         test_gpu_profiler.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
    GPU_SPLIT       GpuProfilerTest
)

oe_add_test(test_omniedge_orchestrator
    SOURCES         test_omniedge_orchestrator.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

oe_add_test(test_priority_scheduler
    SOURCES         test_priority_scheduler.cpp
    LINK_LIBRARIES  omniedge_common
)

oe_add_test(test_pipeline_graph
    SOURCES         test_pipeline_graph.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

oe_add_test(test_graph_builder
    SOURCES         test_graph_builder.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

oe_add_test(test_pipeline_graph_switch
    SOURCES         test_pipeline_graph_switch.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

oe_add_test(test_profile_transitions
    SOURCES         test_profile_transitions.cpp
    LINK_LIBRARIES  omniedge_orchestrator_common
)

