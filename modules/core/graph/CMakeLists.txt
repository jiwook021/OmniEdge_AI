# ---------------------------------------------------------------------------
# omniedge_core_graph — Pure graph data structures and algorithms.
#
# DEPENDENCY RULE: This library links ONLY to tl::expected, spdlog, and
# nlohmann_json. NO transport, NO hal, NO ZMQ, NO CUDA.
# This guarantees unit tests run with zero runtime infrastructure.
# ---------------------------------------------------------------------------

add_library(omniedge_core_graph STATIC
    src/pipeline_edge.cpp
    src/edge_state_machine.cpp
    src/pipeline_graph.cpp
    src/graph_topology.cpp
    src/graph_diff.cpp
    src/graph_validator.cpp
    src/graph_serialization.cpp
)

target_include_directories(omniedge_core_graph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(omniedge_core_graph PUBLIC
    tl::expected
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

target_compile_features(omniedge_core_graph PUBLIC cxx_std_20)
