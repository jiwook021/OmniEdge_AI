# =========================================================================
# tests/integration — Full system tests (real ZMQ, real GPU, real inferencers)
#
# Aggregates subdirectories in alphabetical order.
# =========================================================================

add_subdirectory(hal)
add_subdirectory(inferencers)
add_subdirectory(pipeline)
add_subdirectory(transport)
