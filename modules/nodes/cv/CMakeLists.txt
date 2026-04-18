# =========================================================================
# modules/nodes/cv — CV node binaries (blur, face recog, face filter,
#                    video denoise, SAM2, security camera, beauty)
#
# Library targets (for tests):
#   omniedge_cv_common             — blur + face recog + face filter nodes
#   omniedge_video_denoise_common  — video denoise node
#   omniedge_sam2_common           — SAM2 segmentation node
#   omniedge_security_common       — security camera node
#   omniedge_beauty_common         — beauty (face beautification) node
#
# Executables:
#   omniedge_bg_blur, omniedge_face_recog, omniedge_face_filter,
#   omniedge_video_denoise, omniedge_sam2, omniedge_security_camera,
#   omniedge_beauty
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

oe_add_node_common(omniedge_beauty_common
    SOURCES   src/beauty_node.cpp
    CORE_LIB  omniedge_core_cv
)

# =========================================================================
# Unconditional executables (stub inferencers)
# =========================================================================

if(OE_ONNXRUNTIME_FOUND)
    oe_add_node_exe(omniedge_bg_blur
        SOURCES    src/main_bg_blur.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_selfie_seg_blur_inferencer.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/basicvsrpp_preprocess.cu
        COMMON_LIB omniedge_cv_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} CUDA::cudart CUDA::nvjpeg
                   omniedge_hal_gpu_cuda
    )
    target_include_directories(omniedge_bg_blur PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(STATUS "[OmniEdge] ONNX Runtime not found — omniedge_bg_blur falls back to stub")
    oe_add_node_exe(omniedge_bg_blur
        SOURCES    src/main_bg_blur.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/selfie_seg_blur_inferencer_stub.cpp
        COMMON_LIB omniedge_cv_common
    )
endif()

oe_add_node_exe(omniedge_face_filter
    SOURCES    src/main_face_filter.cpp
               ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_filter_inferencer_stub.cpp
    COMMON_LIB omniedge_cv_common
)

# omniedge_beauty stub — only when ONNX RT is not available (GPU variant below)
if(NOT OE_ONNXRUNTIME_FOUND)
    oe_add_node_exe(omniedge_beauty
        SOURCES    src/main_beauty.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/beauty_inferencer_stub.cpp
        COMMON_LIB omniedge_beauty_common
    )
endif()

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
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_recog_preprocess.cu
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
if(OE_ONNXRUNTIME_FOUND)
    oe_add_node_exe(omniedge_video_denoise
        SOURCES    src/main_video_denoise.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_basicvsrpp_inferencer.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/basicvsrpp_preprocess.cu
        COMMON_LIB omniedge_video_denoise_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} CUDA::cudart CUDA::nvjpeg
    )
    target_include_directories(omniedge_video_denoise PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(STATUS "[OmniEdge] ONNX Runtime not found — omniedge_video_denoise skipped")
    add_executable(omniedge_video_denoise EXCLUDE_FROM_ALL src/main_video_denoise.cpp)
    target_link_libraries(omniedge_video_denoise PRIVATE omniedge_video_denoise_common yaml-cpp)
    oe_set_target_warnings(omniedge_video_denoise)
    oe_set_release_debug_flags(omniedge_video_denoise)
endif()

# --- omniedge_sam2 ---
find_package(JPEG QUIET)
if(TARGET omniedge_beauty_common AND JPEG_FOUND)
    target_link_libraries(omniedge_beauty_common PUBLIC JPEG::JPEG)
endif()
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

# --- omniedge_beauty (GPU variant: ONNX RT + beauty CUDA kernels + nvJPEG) ---
if(OE_ONNXRUNTIME_FOUND)
    oe_add_node_exe(omniedge_beauty_gpu
        SOURCES    src/main_beauty.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_beauty_inferencer.cpp
                   ${CMAKE_SOURCE_DIR}/modules/core/cv/src/beauty_preprocess.cu
        COMMON_LIB omniedge_beauty_common
        EXTRA_LIBS ${OE_ONNXRUNTIME_LIB} CUDA::cudart CUDA::nvjpeg
    )
    # Eigen3 (header-only, for TPS solver)
    find_package(Eigen3 QUIET)
    target_include_directories(omniedge_beauty_gpu PRIVATE
        ${OE_ONNXRUNTIME_INCLUDE_DIR}
    )
    # Eigen3 headers are third-party — mark SYSTEM to suppress internal warnings
    target_include_directories(omniedge_beauty_gpu SYSTEM PRIVATE
        ${EIGEN3_INCLUDE_DIR}
    )
    # Override stub binary: GPU variant is the default when ONNX RT is available
    set_target_properties(omniedge_beauty_gpu PROPERTIES OUTPUT_NAME omniedge_beauty)
else()
    message(STATUS "[OmniEdge] ONNX Runtime not found — omniedge_beauty_gpu skipped (stub only)")
endif()

# =========================================================================
# Security Camera (YOLO detection + NVENC recording + event logging)
#
# Disabled by default — security_camera_node.cpp needs API updates.
# Enable with -DOE_BUILD_SECURITY_CAMERA=ON once code is updated.
# =========================================================================

option(OE_BUILD_SECURITY_CAMERA "Build the security camera module" ON)

if(OE_BUILD_SECURITY_CAMERA)
    # GStreamer (required for SecurityRecorder NVENC pipeline)
    find_package(PkgConfig QUIET)
    set(_SECURITY_GST_FOUND FALSE)
    if(PkgConfig_FOUND)
        pkg_check_modules(SECURITY_GST QUIET gstreamer-1.0 gstreamer-app-1.0)
        if(SECURITY_GST_FOUND)
            set(_SECURITY_GST_FOUND TRUE)
        endif()
    endif()

    if(OE_ONNXRUNTIME_FOUND AND _SECURITY_GST_FOUND AND TARGET opencv_core)
        oe_add_node_common(omniedge_security_common
            SOURCES   src/security_camera_node.cpp
            CORE_LIB  omniedge_core_cv
        )

        oe_add_node_exe(omniedge_security_camera
            SOURCES    src/main_security_camera.cpp
                       ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_security_inferencer.cpp
                       ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_event_logger.cpp
                       ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_recorder.cpp
                       ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_detection_annotator.cpp
            COMMON_LIB omniedge_security_common
            EXTRA_LIBS CUDA::cudart CUDA::nvjpeg
                       ${OE_ONNXRUNTIME_LIB}
                       opencv_core opencv_imgproc opencv_imgcodecs
                       ${SECURITY_GST_LIBRARIES}
        )
        target_include_directories(omniedge_security_camera PRIVATE
            ${OE_ONNXRUNTIME_INCLUDE_DIR}
            ${SECURITY_GST_INCLUDE_DIRS}
        )

        # --- omniedge_security_vlm — LP agent loop: clip extract + Gemma VLM ---
        # Pure downstream consumer; only needs OpenCV for JPEG encoding plus the
        # core/cv VisionWorkerClient (which owns the Python sidecar subprocess).
        oe_add_node_common(omniedge_security_vlm_common
            SOURCES    src/security_vlm_node.cpp
            CORE_LIB   omniedge_core_cv
            EXTRA_LIBS opencv_core opencv_imgproc opencv_imgcodecs
        )

        oe_add_node_exe(omniedge_security_vlm
            SOURCES    src/main_security_vlm.cpp
            COMMON_LIB omniedge_security_vlm_common
        )
    else()
        message(STATUS "[OmniEdge] ONNX Runtime, GStreamer, or OpenCV not found — omniedge_security_camera skipped")
    endif()
else()
    message(STATUS "[OmniEdge] Security camera disabled (OE_BUILD_SECURITY_CAMERA=OFF)")
endif()
