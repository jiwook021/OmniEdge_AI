# =========================================================================
# Section 1 — Comment Header
# modules/transport/shm — POSIX shared memory (ShmMapping, ShmDoubleBuffer,
#                          ShmFrameReader)
#
# Target: omniedge_transport_shm (STATIC)
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_transport_shm STATIC
    src/shm_mapping.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_transport_shm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# shm_mapping.hpp includes zmq/audio_constants.hpp and common/constants/video_constants.hpp
target_include_directories(omniedge_transport_shm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/transport/zmq/include>
)

# =========================================================================
# Section 4 — Link libraries
# =========================================================================
target_link_libraries(omniedge_transport_shm PUBLIC
    omniedge_logger
    rt
)

# =========================================================================
# Section 5 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_transport_shm)
