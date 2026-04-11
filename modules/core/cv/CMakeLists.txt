# =========================================================================
# Section 1 — Comment Header
# modules/core/cv — CV algorithms (CPU-safe subset)
#
# Target: omniedge_core_cv (STATIC)
# Optional deps: SQLite3
#
# GPU/ONNX inferencers NOT included in this library:
#   yolo_seg_engine.cu             — CUDA + TensorRT
#   onnx_basicvsrpp_inferencer.cpp — ONNX Runtime
# These are compiled directly by the targets that need them.
# =========================================================================

# =========================================================================
# Section 2 — Optional dependency detection
# =========================================================================
find_package(SQLite3 QUIET)

# =========================================================================
# Section 3 — Source file list
# =========================================================================
set(CORE_CV_SOURCES src/face_gallery.cpp)

# =========================================================================
# Section 4 — Library declaration
# =========================================================================
add_library(omniedge_core_cv STATIC ${CORE_CV_SOURCES})

# =========================================================================
# Section 5 — Include directories
# =========================================================================
target_include_directories(omniedge_core_cv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# =========================================================================
# Section 6 — Required link libraries
# =========================================================================
target_link_libraries(omniedge_core_cv PUBLIC
    omniedge_interfaces
    omniedge_common
)

# =========================================================================
# Section 7 — Conditional dependency linking
# =========================================================================
if(TARGET SQLite3::SQLite3)
    target_link_libraries(omniedge_core_cv PUBLIC SQLite3::SQLite3)
elseif(SQLite3_FOUND AND SQLite3_LIBRARIES)
    target_include_directories(omniedge_core_cv PUBLIC ${SQLite3_INCLUDE_DIRS})
    target_link_libraries(omniedge_core_cv PUBLIC ${SQLite3_LIBRARIES})
endif()

# =========================================================================
# Section 8 — Warning flags
# =========================================================================
oe_set_target_warnings(omniedge_core_cv)
