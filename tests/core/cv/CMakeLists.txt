# =========================================================================
# tests/core/cv — Face gallery SQLite CRUD + face filter stub +
#                 security event logger / annotator / recorder (CPU-only)
# =========================================================================

# =========================================================================
# SQLite3-gated tests
# =========================================================================
find_package(SQLite3 QUIET)

if(TARGET SQLite3::SQLite3)
    set(OE_SQLITE3_LIB SQLite3::SQLite3)
elseif(SQLite3_FOUND AND SQLite3_LIBRARIES)
    set(OE_SQLITE3_LIB ${SQLite3_LIBRARIES})
else()
    message(WARNING "SQLite3 not found — test_face_gallery_sqlite disabled")
endif()

if(OE_SQLITE3_LIB)
    oe_add_test(test_face_gallery_sqlite
        SOURCES     test_face_gallery_sqlite.cpp
                    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_gallery.cpp
        LINK_LIBRARIES  omniedge_core_cv omniedge_interfaces ${OE_SQLITE3_LIB}
    )
endif()

# =========================================================================
# Face filter stub tests (always available)
# =========================================================================
oe_add_test(test_face_filter_stub
    SOURCES     test_face_filter_stub.cpp
                ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_filter_inferencer_stub.cpp
    LINK_LIBRARIES  omniedge_core_cv omniedge_interfaces omniedge_common
)

oe_add_test(test_face_filter_benchmark
    SOURCES     test_face_filter_benchmark.cpp
                ${CMAKE_SOURCE_DIR}/modules/core/cv/src/face_filter_inferencer_stub.cpp
    LINK_LIBRARIES  omniedge_core_cv omniedge_interfaces omniedge_common
)

# =========================================================================
# Security Camera Mode tests
# =========================================================================

# --- SecurityEventLogger (no GPU, no GStreamer, no OpenCV) ----------------
oe_add_test(test_security_event_logger
    SOURCES     test_security_event_logger.cpp
                ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_event_logger.cpp
    LINK_LIBRARIES  omniedge_core_cv omniedge_common
)

oe_add_test(test_vision_worker_client
    SOURCES     test_vision_worker_client.cpp
    LINK_LIBRARIES  omniedge_core_cv omniedge_common
)

# --- SecurityDetectionAnnotator (needs OpenCV imgproc + imgcodecs) --------
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(TARGET opencv_core AND TARGET opencv_imgproc AND TARGET opencv_imgcodecs)
    oe_add_test(test_security_detection_annotator
        SOURCES     test_security_detection_annotator.cpp
                    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_detection_annotator.cpp
        LINK_LIBRARIES  omniedge_core_cv omniedge_common
                        opencv_core opencv_imgproc opencv_imgcodecs
    )
else()
    message(STATUS "OpenCV not found — test_security_detection_annotator disabled")
endif()

# --- SecurityRecorder (needs GStreamer) -----------------------------------
include(FindPkgConfig OPTIONAL)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(_TEST_GST QUIET gstreamer-1.0 gstreamer-app-1.0)
endif()
if(_TEST_GST_FOUND)
    oe_add_test(test_security_recorder
        SOURCES     test_security_recorder.cpp
                    ${CMAKE_SOURCE_DIR}/modules/core/cv/src/security_recorder.cpp
        LINK_LIBRARIES  omniedge_core_cv omniedge_common
                        ${_TEST_GST_LIBRARIES}
    )
    target_include_directories(test_security_recorder PRIVATE
        ${_TEST_GST_INCLUDE_DIRS}
    )
else()
    message(STATUS "GStreamer not found — test_security_recorder disabled")
endif()
