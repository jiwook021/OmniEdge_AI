add_library(omniedge_core_pipeline_orchestrator STATIC
    src/switch_plan.cpp
    src/crash_handler.cpp
    src/pipeline_orchestrator.cpp
)

target_include_directories(omniedge_core_pipeline_orchestrator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(omniedge_core_pipeline_orchestrator PUBLIC
    omniedge_core_graph
)

target_compile_features(omniedge_core_pipeline_orchestrator PUBLIC cxx_std_20)
oe_set_target_warnings(omniedge_core_pipeline_orchestrator)
