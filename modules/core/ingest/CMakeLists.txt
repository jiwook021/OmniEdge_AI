# =========================================================================
# Section 1 — Comment Header
# modules/core/ingest — GStreamer pipeline builder, Silero VAD
#
# Target: omniedge_core_ingest (STATIC)
# Optional deps: GStreamer, ONNX Runtime (Silero VAD)
# =========================================================================

# =========================================================================
# Section 2 — Source files + library declaration
# =========================================================================
add_library(omniedge_core_ingest STATIC
    src/gstreamer_pipeline.cpp
    src/silero_vad.cpp
)

# =========================================================================
# Section 3 — Include directories
# =========================================================================
target_include_directories(omniedge_core_ingest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 4 — Required link libraries
# =========================================================================
target_link_libraries(omniedge_core_ingest PUBLIC
    omniedge_interfaces
    omniedge_common
)

# =========================================================================
# Section 5 — Optional dependency detection and linking
# =========================================================================

# GStreamer (optional — graceful skip if not found)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GST QUIET gstreamer-1.0 gstreamer-app-1.0)
    if(GST_FOUND)
        target_include_directories(omniedge_core_ingest PUBLIC ${GST_INCLUDE_DIRS})
        target_link_libraries(omniedge_core_ingest PUBLIC ${GST_LIBRARIES})
    endif()
endif()

# ONNX Runtime for Silero VAD
if(TARGET onnxruntime::onnxruntime)
    target_link_libraries(omniedge_core_ingest PUBLIC onnxruntime::onnxruntime)
elseif(OE_ONNXRUNTIME_FOUND)
    target_include_directories(omniedge_core_ingest PUBLIC ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_core_ingest PUBLIC ${OE_ONNXRUNTIME_LIB})
endif()

# =========================================================================
# Section 6 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_ingest)
