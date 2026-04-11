# =========================================================================
# modules/nodes/cv — CV node binaries (blur, face recog, face filter,
#                    video denoise, SAM2)
#
# Library targets (for tests):
#   omniedge_cv_common             — blur + face recog + face filter nodes
#   omniedge_video_denoise_common  — video denoise node
#   omniedge_sam2_common           — SAM2 segmentation node
#
# Executables:
#   omniedge_bg_blur, omniedge_face_recog, omniedge_face_filter,
#   omniedge_video_denoise, omniedge_sam2
# =========================================================================

# =========================================================================
# Common libraries (shared with tests)
# =========================================================================

oe_add_node_common(omniedge_cv_common
    SOURCES   src/background_blur_node.cpp
              src/face_recog_node.cpp
              src/face_filter_node.cpp
    CORE_LIB  omniedge_core_cv
)

oe_add_node_common(omniedge_video_denoise_common
    SOURCES   src/video_denoise_node.cpp
    CORE_LIB  omniedge_core_cv
)

oe_add_node_common(omniedge_sam2_common
    SOURCES   src/sam2_node.cpp
    CORE_LIB  omniedge_core_cv
)

# =========================================================================
# Unconditional executables (stub inferencers)
# =========================================================================

oe_add_node_exe(omniedge_bg_blur
    SOURCES    src/main_bg_blur.cpp
               ${CMAKE_SOURCE_DIR}/modules/core/cv/src/tensorrt_blur_inferencer_stub.cpp
    COMMON_LIB omniedge_cv_common
)

oe_add_node_exe(omniedge_face_filter
    SOURCES    src/main_face_filter.cpp
               ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_filter_inferencer_stub.cpp
    COMMON_LIB omniedge_cv_common
)

# =========================================================================
# GPU-gated executables
# =========================================================================

# Single OpenCV find — face_recog needs core+imgproc+dnn,
# video_denoise adds imgcodecs.  Consolidate to avoid clobbering variables.
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs dnn)

# --- omniedge_face_recog ---
if(OE_ONNXRUNTIME_FOUND AND TARGET opencv_core AND TARGET opencv_dnn)
    oe_add_node_exe(omniedge_face_recog
        SOURCES    src/main_face_recog.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_face_recog_inferencer.cpp
        COMMON_LIB omniedge_cv_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} opencv_core opencv_imgproc opencv_dnn
    )
    target_include_directories(omniedge_face_recog PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(STATUS "[OmniEdge] ONNX Runtime or OpenCV not found — omniedge_face_recog skipped")
    add_executable(omniedge_face_recog EXCLUDE_FROM_ALL src/main_face_recog.cpp)
    target_link_libraries(omniedge_face_recog PRIVATE omniedge_cv_common yaml-cpp)
    oe_set_target_warnings(omniedge_face_recog)
    oe_set_release_debug_flags(omniedge_face_recog)
endif()

# --- omniedge_video_denoise ---
if(OE_ONNXRUNTIME_FOUND AND TARGET opencv_core AND TARGET opencv_dnn)
    oe_add_node_exe(omniedge_video_denoise
        SOURCES    src/main_video_denoise.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_basicvsrpp_inferencer.cpp
        COMMON_LIB omniedge_video_denoise_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn
    )
    target_include_directories(omniedge_video_denoise PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(STATUS "[OmniEdge] ONNX Runtime or OpenCV not found — omniedge_video_denoise skipped")
    add_executable(omniedge_video_denoise EXCLUDE_FROM_ALL src/main_video_denoise.cpp)
    target_link_libraries(omniedge_video_denoise PRIVATE omniedge_video_denoise_common yaml-cpp)
    oe_set_target_warnings(omniedge_video_denoise)
    oe_set_release_debug_flags(omniedge_video_denoise)
endif()

# --- omniedge_sam2 ---
find_package(JPEG QUIET)
if(OE_ONNXRUNTIME_FOUND AND JPEG_FOUND)
    oe_add_node_exe(omniedge_sam2
        SOURCES    src/main_sam2.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_sam2_inferencer.cpp
        COMMON_LIB omniedge_sam2_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} JPEG::JPEG
    )
    target_include_directories(omniedge_sam2 PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    add_executable(omniedge_sam2
        src/main_sam2.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/sam2_inferencer_stub.cpp
    )
    target_link_libraries(omniedge_sam2 PRIVATE omniedge_sam2_common yaml-cpp)
    oe_set_target_warnings(omniedge_sam2)
endif()
