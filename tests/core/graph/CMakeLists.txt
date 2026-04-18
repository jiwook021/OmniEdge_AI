# tests/core/graph — Unit tests for core::graph library
# Label: graph_unit (CPU-only, no GPU required)

oe_add_test(test_pipeline_vertex
    SOURCES         test_pipeline_vertex.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_pipeline_edge
    SOURCES         test_pipeline_edge.cpp
    LINK_LIBRARIES  omniedge_core_graph
    INCLUDE_DIRS    ${CMAKE_SOURCE_DIR}/modules/common/include
    LABELS          graph_unit
)

oe_add_test(test_edge_state_machine
    SOURCES         test_edge_state_machine.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_core_pipeline_graph
    SOURCES         test_core_pipeline_graph.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_graph_topology
    SOURCES         test_graph_topology.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_graph_diff
    SOURCES         test_graph_diff.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_graph_validator
    SOURCES         test_graph_validator.cpp
    LINK_LIBRARIES  omniedge_core_graph
    LABELS          graph_unit
)

oe_add_test(test_graph_serialization
    SOURCES         test_graph_serialization.cpp
    LINK_LIBRARIES  omniedge_core_graph
    INCLUDE_DIRS    ${CMAKE_SOURCE_DIR}/modules/common/include
    LABELS          graph_unit
)
