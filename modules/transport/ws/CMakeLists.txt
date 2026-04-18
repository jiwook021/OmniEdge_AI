# =========================================================================
# Section 1 — Comment Header
# modules/transport/ws — uWebSockets server
#
# Target: omniedge_transport_ws (STATIC)
# Optional deps: uWebSockets/uSockets, OpenSSL, zlib
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_transport_ws STATIC
    src/ws_server.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_transport_ws PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Required link libraries
# =========================================================================
target_link_libraries(omniedge_transport_ws PUBLIC
    omniedge_logger
)

# =========================================================================
# Section 5 — Optional dependency detection and linking
# =========================================================================

# uWebSockets — header-only library backed by uSockets
find_path(UWS_INCLUDE_DIR uwebsockets/App.h
    HINTS /usr/local/include /usr/include
    PATH_SUFFIXES uWebSockets
)
if(UWS_INCLUDE_DIR)
    target_include_directories(omniedge_transport_ws PUBLIC ${UWS_INCLUDE_DIR})
endif()

find_library(USOCKETS_LIB NAMES uSockets usockets HINTS /usr/local/lib /usr/lib)
if(USOCKETS_LIB)
    target_link_libraries(omniedge_transport_ws PUBLIC ${USOCKETS_LIB})
endif()

# uSockets depends on OpenSSL + zlib
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    target_link_libraries(omniedge_transport_ws PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(omniedge_transport_ws PUBLIC ZLIB::ZLIB)
endif()

# =========================================================================
# Section 6 — Warning flags
#   - _no_error: system fmt/spdlog headers trigger -Warray-bounds false
#     positives under GCC 13 with -O2+
# =========================================================================
oe_set_target_warnings_no_error(omniedge_transport_ws)
