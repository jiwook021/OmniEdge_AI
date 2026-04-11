# =========================================================================
# Section 1 — Comment Header
# modules/common — Logger + shared utility headers
#
# Target: omniedge_logger (STATIC)
# Provides: include/common/ headers to all downstream targets
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_logger STATIC
    src/oe_logger.cpp
    src/subprocess_manager.cpp
    src/diagnostic_ring.cpp
    src/file.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_logger PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Vendored dependencies
# =========================================================================

# inih — vendored INI file parser (third_party/inih, BSD-3-Clause)
set(INIH_DIR "${CMAKE_SOURCE_DIR}/third_party/inih")
add_library(inih STATIC "${INIH_DIR}/ini.c" "${INIH_DIR}/INIReader.cpp")
target_include_directories(inih PUBLIC "${INIH_DIR}")
set_target_properties(inih PROPERTIES POSITION_INDEPENDENT_CODE ON)

# =========================================================================
# Section 5 — Link libraries
# =========================================================================
target_link_libraries(omniedge_logger PUBLIC tl::expected inih)

if(TARGET spdlog::spdlog)
    target_link_libraries(omniedge_logger PUBLIC spdlog::spdlog)
else()
    # Header-only spdlog (system install without cmake config)
    find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h)
    if(SPDLOG_INCLUDE_DIR)
        target_include_directories(omniedge_logger PUBLIC ${SPDLOG_INCLUDE_DIR})
    endif()
endif()

# =========================================================================
# Section 6 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_logger)
