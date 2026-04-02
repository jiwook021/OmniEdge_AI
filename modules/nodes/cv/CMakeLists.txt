# =========================================================================
# Section 1 — Comment Header
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
# Section 2 — Common libraries (shared with tests)
# =========================================================================

# --- omniedge_cv_common: blur + face recog + face filter nodes ---
add_library(omniedge_cv_common STATIC
    src/background_blur_node.cpp
    src/face_recog_node.cpp
    src/face_filter_node.cpp
)
target_include_directories(omniedge_cv_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_cv_common PUBLIC
    omniedge_core_cv
    omniedge_common
)
oe_set_target_warnings(omniedge_cv_common)

# --- omniedge_video_denoise_common ---
add_library(omniedge_video_denoise_common STATIC
    src/video_denoise_node.cpp
)
target_include_directories(omniedge_video_denoise_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_video_denoise_common PUBLIC
    omniedge_core_cv
    omniedge_common
)
oe_set_target_warnings(omniedge_video_denoise_common)

# --- omniedge_sam2_common ---
add_library(omniedge_sam2_common STATIC
    src/sam2_node.cpp
)
target_include_directories(omniedge_sam2_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(omniedge_sam2_common PUBLIC
    omniedge_core_cv
    omniedge_common
)
oe_set_target_warnings(omniedge_sam2_common)

# =========================================================================
# Section 3 — Executables
# =========================================================================

# --- omniedge_bg_blur ---
# Uses stub inferencer until TensorRT implementation is ready
add_executable(omniedge_bg_blur
    src/main_bg_blur.cpp
    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/tensorrt_blur_inferencer_stub.cpp
)
target_link_libraries(omniedge_bg_blur PRIVATE omniedge_cv_common yaml-cpp)
oe_set_target_warnings(omniedge_bg_blur)
oe_set_release_debug_flags(omniedge_bg_blur)

# --- omniedge_face_filter ---
# Uses stub inferencer until ONNX FaceMesh implementation is ready
add_executable(omniedge_face_filter
    src/main_face_filter.cpp
    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_filter_inferencer_stub.cpp
)
target_link_libraries(omniedge_face_filter PRIVATE omniedge_cv_common yaml-cpp)
oe_set_target_warnings(omniedge_face_filter)
oe_set_release_debug_flags(omniedge_face_filter)

# =========================================================================
# Section 4 — Optional executables (GPU-gated)
# =========================================================================

# --- omniedge_face_recog ---
# ONNX Runtime SCRFD+AdaFace (primary) or InspireFace (legacy fallback)
# InspireFace is resolved by oe_find_inspireface() in root CMakeLists.txt
if(OE_ONNXRUNTIME_FOUND)
    add_executable(omniedge_face_recog
        src/main_face_recog.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_face_recog_inferencer.cpp
    )
    target_link_libraries(omniedge_face_recog PRIVATE omniedge_cv_common yaml-cpp)
    target_include_directories(omniedge_face_recog PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_face_recog PRIVATE ${OE_ONNXRUNTIME_LIB})

    # Also compile InspireFace inferencer if SDK available (legacy variant)
    if(OE_INSPIREFACE_FOUND)
        target_sources(omniedge_face_recog PRIVATE
            ${CMAKE_SOURCE_DIR}/modules/core/cv/src/inspire_face_inferencer.cpp)
        target_include_directories(omniedge_face_recog PRIVATE ${OE_INSPIREFACE_INCLUDE_DIR})
        target_link_libraries(omniedge_face_recog PRIVATE ${OE_INSPIREFACE_LIB})
        target_compile_definitions(omniedge_face_recog PRIVATE OE_HAS_INSPIREFACE=1)
    endif()
    oe_set_target_warnings(omniedge_face_recog)
    oe_set_release_debug_flags(omniedge_face_recog)

elseif(OE_INSPIREFACE_FOUND)
    # Fallback: InspireFace-only build (no ONNX Runtime)
    add_executable(omniedge_face_recog
        src/main_face_recog.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/inspire_face_inferencer.cpp
    )
    target_link_libraries(omniedge_face_recog PRIVATE omniedge_cv_common yaml-cpp)
    target_include_directories(omniedge_face_recog PRIVATE ${OE_INSPIREFACE_INCLUDE_DIR})
    target_link_libraries(omniedge_face_recog PRIVATE ${OE_INSPIREFACE_LIB})
    target_compile_definitions(omniedge_face_recog PRIVATE OE_HAS_INSPIREFACE=1)
    oe_set_target_warnings(omniedge_face_recog)
    oe_set_release_debug_flags(omniedge_face_recog)

else()
    message(STATUS "[OmniEdge] Neither ONNX Runtime nor InspireFace found — omniedge_face_recog skipped")
    add_executable(omniedge_face_recog EXCLUDE_FROM_ALL src/main_face_recog.cpp)
    target_link_libraries(omniedge_face_recog PRIVATE omniedge_cv_common yaml-cpp)
endif()

# --- omniedge_video_denoise ---
# Compiles the ONNX BasicVSR++ inferencer directly
# Requires: ONNX Runtime (CUDA EP) + OpenCV (resize, cvtColor, imencode)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OE_ONNXRUNTIME_FOUND AND OpenCV_FOUND)
    add_executable(omniedge_video_denoise
        src/main_video_denoise.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_basicvsrpp_inferencer.cpp
    )
    target_link_libraries(omniedge_video_denoise PRIVATE
        omniedge_video_denoise_common yaml-cpp ${OpenCV_LIBS})
    target_include_directories(omniedge_video_denoise PRIVATE
        ${OE_ONNXRUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(omniedge_video_denoise PRIVATE ${OE_ONNXRUNTIME_LIB})
    oe_set_target_warnings(omniedge_video_denoise)
    oe_set_release_debug_flags(omniedge_video_denoise)
else()
    message(STATUS "[OmniEdge] ONNX Runtime or OpenCV not found — omniedge_video_denoise skipped")
    add_executable(omniedge_video_denoise EXCLUDE_FROM_ALL src/main_video_denoise.cpp)
    target_link_libraries(omniedge_video_denoise PRIVATE omniedge_video_denoise_common yaml-cpp)
endif()

# --- omniedge_sam2 ---
# Uses stub inferencer until full ONNX SAM2 models are available
if(OE_ONNXRUNTIME_FOUND AND JPEG_FOUND)
    add_executable(omniedge_sam2
        src/main_sam2.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_sam2_inferencer.cpp
    )
    target_link_libraries(omniedge_sam2 PRIVATE
        omniedge_sam2_common yaml-cpp JPEG::JPEG)
    target_include_directories(omniedge_sam2 PRIVATE ${OE_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(omniedge_sam2 PRIVATE ${OE_ONNXRUNTIME_LIB})
    oe_set_target_warnings(omniedge_sam2)
    oe_set_release_debug_flags(omniedge_sam2)
else()
    add_executable(omniedge_sam2
        src/main_sam2.cpp
        ${CMAKE_SOURCE_DIR}/modules/core/cv/src/sam2_inferencer_stub.cpp
    )
    target_link_libraries(omniedge_sam2 PRIVATE omniedge_sam2_common yaml-cpp)
    oe_set_target_warnings(omniedge_sam2)
endif()
