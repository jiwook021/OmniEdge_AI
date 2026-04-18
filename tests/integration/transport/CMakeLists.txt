# =========================================================================
# tests/integration/transport — Real SHM + ZMQ tests
#
# GTest and CUDAToolkit are resolved at root — no find_package() needed.
# =========================================================================

oe_add_test(test_shm_mapping
    SOURCES     test_shm_mapping.cpp
    LINK_LIBRARIES  omniedge_common
)

oe_add_test(test_shm_double_buffer
    SOURCES     test_shm_double_buffer.cpp
    LINK_LIBRARIES  omniedge_common
)

oe_add_test(test_shm_circular_buffer
    SOURCES     test_shm_circular_buffer.cpp
    LINK_LIBRARIES  omniedge_common
)

oe_add_test(test_shm_jpeg_transport
    SOURCES     test_shm_jpeg_transport.cpp
    LINK_LIBRARIES  omniedge_cv_common
    INCLUDE_DIRS    ${CMAKE_SOURCE_DIR}/tests/interfaces/cv
    COMPILE_DEFS    OE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

oe_add_test(test_zmq_publisher
    SOURCES     test_zmq_publisher.cpp
    LINK_LIBRARIES  omniedge_common
    RESOURCE_LOCK   zmq_publisher_ports
)

oe_add_test(test_zmq_subscriber
    SOURCES     test_zmq_subscriber.cpp
    LINK_LIBRARIES  omniedge_common
    RESOURCE_LOCK   zmq_subscriber_ports
)

oe_add_test(test_message_router
    SOURCES     test_message_router.cpp
    LINK_LIBRARIES  omniedge_common
    RESOURCE_LOCK   zmq_test_ports
)

oe_add_test(test_transport_edge_cases
    SOURCES     test_transport_edge_cases.cpp
    LINK_LIBRARIES  omniedge_common
    RESOURCE_LOCK   zmq_test_ports
)
