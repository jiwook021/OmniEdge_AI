# =========================================================================
# tests/nodes/conversation — Conversation node wiring tests
#
# Accepted deviation: raw add_executable used because these tests compile
# cross-module sources from modules/nodes/conversation/src/ and
# modules/core/conversation/src/ — not supported by oe_add_test().
# =========================================================================

# =========================================================================
# Level 1 — Pure logic tests (GenerationParams, config validation)
# =========================================================================
add_executable(test_conversation_params
    test_conversation_params.cpp
    ${CMAKE_SOURCE_DIR}/modules/nodes/conversation/src/conversation_node.cpp
)
target_compile_features(test_conversation_params PRIVATE cxx_std_20)
oe_set_target_warnings_no_error(test_conversation_params)
target_include_directories(test_conversation_params PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_conversation_params PRIVATE
    GTest::gtest_main omniedge_conversation_common
    nlohmann_json::nlohmann_json yaml-cpp
)
add_test(NAME ConversationParamsTests COMMAND test_conversation_params)

# =========================================================================
# Level 2 — Node integration tests
# =========================================================================
add_executable(test_conversation_node
    test_conversation_node.cpp
    ${CMAKE_SOURCE_DIR}/modules/nodes/conversation/src/conversation_node.cpp
)
target_compile_features(test_conversation_node PRIVATE cxx_std_20)
oe_set_target_warnings_no_error(test_conversation_node)
target_include_directories(test_conversation_node PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_conversation_node PRIVATE
    CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(test_conversation_node PRIVATE
    GTest::gtest_main omniedge_conversation_common
    nlohmann_json::nlohmann_json yaml-cpp
)
add_test(NAME ConversationNodeTests COMMAND test_conversation_node)
set_tests_properties(ConversationNodeTests PROPERTIES RESOURCE_LOCK zmq_test_ports)

# =========================================================================
# Benchmark — Token throughput
# =========================================================================
add_executable(test_conversation_benchmark
    test_conversation_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/modules/nodes/conversation/src/conversation_node.cpp
)
target_compile_features(test_conversation_benchmark PRIVATE cxx_std_20)
oe_set_target_warnings_no_error(test_conversation_benchmark)
target_include_directories(test_conversation_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_conversation_benchmark PRIVATE
    GTest::gtest_main omniedge_conversation_common
    nlohmann_json::nlohmann_json yaml-cpp
)
add_test(NAME ConversationBenchmarkTests COMMAND test_conversation_benchmark)
set_tests_properties(ConversationBenchmarkTests PROPERTIES RESOURCE_LOCK zmq_test_ports)

# =========================================================================
# Level 3 — End-to-end: real ingest nodes → SHM → conversation node
#
# Runs VideoIngestNode (GStreamer file source) and AudioIngestNode
# (GStreamer WAV source + Silero VAD) feeding into ConversationNode.
# Requires GStreamer runtime, silero_vad.onnx, and test fixtures.
# =========================================================================
add_executable(test_e2e_ingest_conversation
    test_e2e_ingest_conversation.cpp
    ${CMAKE_SOURCE_DIR}/modules/nodes/conversation/src/conversation_node.cpp
)
target_compile_features(test_e2e_ingest_conversation PRIVATE cxx_std_20)
oe_set_target_warnings_no_error(test_e2e_ingest_conversation)
target_include_directories(test_e2e_ingest_conversation PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_e2e_ingest_conversation PRIVATE
    CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(test_e2e_ingest_conversation PRIVATE
    GTest::gtest_main
    omniedge_conversation_common
    omniedge_ingest_common
    nlohmann_json::nlohmann_json yaml-cpp
)
add_test(NAME E2eIngestConversationTests COMMAND test_e2e_ingest_conversation)
set_tests_properties(E2eIngestConversationTests PROPERTIES
    RESOURCE_LOCK zmq_test_ports
    LABELS "e2e;gstreamer"
)
