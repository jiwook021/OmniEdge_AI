# =========================================================================
# Section 1 — Comment Header
# modules/core/conversation — Conversation inferencer stubs (CPU-only builds)
#
# Target: omniedge_core_conversation (STATIC)
# Note: Real model inferencers are compiled separately where GPU + model
#       libraries are available. The stubs satisfy the linker for
#       mock-based tests.
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_core_conversation STATIC
    src/qwen_omni_inferencer_stub.cpp
    src/gemma_e4b_inferencer_stub.cpp
    src/audio_accumulator.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_core_conversation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_core_conversation PUBLIC
    omniedge_interfaces
    omniedge_common
    omniedge_transport_shm
    omniedge_transport_zmq
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_conversation)
