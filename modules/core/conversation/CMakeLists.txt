# =========================================================================
# Section 1 — Comment Header
# modules/core/conversation — Conversation inferencers
#
# Target: omniedge_core_conversation (STATIC)
#
# OE_USE_REAL_CONVERSATION=ON:  Real GemmaInferencer — spawns model_generate.py
#                               (HuggingFace transformers subprocess, gated on
#                               HF_TOKEN for google/gemma-4-*-it). TRT-LLM is
#                               NOT required — TRT-LLM 1.2.0 does not support
#                               Gemma-4; conversation uses pure HF transformers.
# OE_USE_REAL_CONVERSATION=OFF: CPU-only stub for mock-based unit tests.
# =========================================================================

option(OE_USE_REAL_CONVERSATION
    "Use real GemmaInferencer (requires HF transformers Python runtime + HF_TOKEN)" OFF)

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
if(OE_USE_REAL_CONVERSATION)
    add_library(omniedge_core_conversation STATIC
        src/gemma_inferencer.cpp
        src/audio_accumulator.cpp
    )
    # Install the Python worker script alongside binaries
    install(FILES src/model_generate.py DESTINATION share)
else()
    add_library(omniedge_core_conversation STATIC
        src/gemma_inferencer_stub.cpp
        src/audio_accumulator.cpp
    )
endif()

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
