# modules/transport/zmq — ZMQ pub/sub, MessageRouter, port settings
#
# Target: omniedge_transport_zmq (STATIC)

add_library(omniedge_transport_zmq STATIC
    src/message_router.cpp
    src/zmq_publisher.cpp
    src/zmq_subscriber.cpp
)

target_include_directories(omniedge_transport_zmq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(omniedge_transport_zmq PUBLIC
    omniedge_logger
    omniedge_hal_gpu
    nlohmann_json::nlohmann_json
    rt
)

if(TARGET cppzmq)
    target_link_libraries(omniedge_transport_zmq PUBLIC cppzmq)
elseif(cppzmq_FOUND)
    target_include_directories(omniedge_transport_zmq PUBLIC ${cppzmq_INCLUDE_DIRS})
    target_link_libraries(omniedge_transport_zmq PUBLIC ${cppzmq_LIBRARIES})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(libzmq REQUIRED IMPORTED_TARGET libzmq)
    target_link_libraries(omniedge_transport_zmq PUBLIC PkgConfig::libzmq)
endif()

oe_set_target_warnings(omniedge_transport_zmq)
