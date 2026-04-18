# tests/nodes/cv — CV node wiring tests (face recog + video denoise + blur)

oe_add_test(test_face_recog_integration
    SOURCES         test_face_recog_integration.cpp
    LINK_LIBRARIES  omniedge_cv_common
    RESOURCE_LOCK   zmq_test_ports
)

oe_add_test(test_video_denoise_node
    SOURCES         test_video_denoise_node.cpp
    LINK_LIBRARIES  omniedge_video_denoise_common
)

oe_add_test(test_video_denoise_benchmark
    SOURCES         test_video_denoise_benchmark.cpp
    LINK_LIBRARIES  omniedge_video_denoise_common
)

oe_add_test(test_blur_deadline_fallback
    SOURCES         test_blur_deadline_fallback.cpp
    LINK_LIBRARIES  omniedge_cv_common
)

oe_add_test(test_face_filter_integration
    SOURCES         test_face_filter_integration.cpp
    LINK_LIBRARIES  omniedge_cv_common
    RESOURCE_LOCK   zmq_test_ports
)

# SAM2 integration test
if(TARGET omniedge_sam2_common AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_sam2_integration.cpp")
    oe_add_test(test_sam2_integration
        SOURCES         test_sam2_integration.cpp
        LINK_LIBRARIES  omniedge_sam2_common
        RESOURCE_LOCK   zmq_test_ports
    )
endif()

# SAM2 benchmark test
if(TARGET omniedge_sam2_common AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_sam2_benchmark.cpp")
    oe_add_test(test_sam2_benchmark
        SOURCES         test_sam2_benchmark.cpp
        LINK_LIBRARIES  omniedge_sam2_common
    )
endif()
