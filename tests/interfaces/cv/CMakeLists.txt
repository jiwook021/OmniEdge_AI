# =========================================================================
# tests/interfaces/cv — BlurInferencer + FaceRecogInferencer contract tests
# =========================================================================

set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")

# =========================================================================
# CPU-only contract tests (always available)
# =========================================================================
oe_add_test(test_blur_api
    SOURCES     test_blur_api.cpp
    LINK_LIBRARIES  omniedge_cv_common
    INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}
    COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
)

oe_add_test(test_blur_isp_pipeline
    SOURCES     test_blur_isp_pipeline.cpp
    LINK_LIBRARIES  omniedge_cv_common
    INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}
    COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
)

oe_add_test(test_face_detection_api
    SOURCES     test_face_detection_api.cpp
    LINK_LIBRARIES  omniedge_cv_common
    INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}
    COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
)

oe_add_test(test_face_matching
    SOURCES     test_face_matching.cpp
    LINK_LIBRARIES  omniedge_cv_common
    INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}
    COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
)

# =========================================================================
# GPU-gated tests (require ONNX Runtime)
# =========================================================================
find_package(OpenCV QUIET COMPONENTS core imgproc dnn)
if(OE_ONNXRUNTIME_FOUND AND OpenCV_FOUND)
    oe_add_test(test_onnx_face_recog_inferencer
        SOURCES     test_onnx_face_recog_inferencer.cpp
                    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/onnx_face_recog_inferencer.cpp
                    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_recog_preprocess.cu
        LINK_LIBRARIES  omniedge_cv_common
        INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR} ${OE_ONNXRUNTIME_INCLUDE_DIR}
                        ${OpenCV_INCLUDE_DIRS}
        COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
        LABELS      gpu
    )
    target_link_libraries(test_onnx_face_recog_inferencer PRIVATE
        ${OE_ONNXRUNTIME_LIB} ${OpenCV_LIBS})
endif()

# =========================================================================
# SAM2 API test (source file not yet implemented)
# =========================================================================
if(TARGET omniedge_sam2_common AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_sam2_api.cpp")
    oe_add_test(test_sam2_api
        SOURCES     test_sam2_api.cpp
        LINK_LIBRARIES  omniedge_sam2_common
        INCLUDE_DIRS    ${CMAKE_CURRENT_SOURCE_DIR}
        COMPILE_DEFS    OE_PROJECT_ROOT="${PROJECT_ROOT_DIR}"
    )
endif()
