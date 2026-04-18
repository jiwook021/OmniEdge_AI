# tests/nodes/ingest — Ingest node wiring tests

oe_add_test(test_video_ingest_node
    SOURCES         test_video_ingest_node.cpp
    LINK_LIBRARIES  omniedge_common omniedge_ingest_common
)

oe_add_test(test_audio_ingest_node
    SOURCES         test_audio_ingest_node.cpp
    LINK_LIBRARIES  omniedge_common omniedge_ingest_common
)

oe_add_test(test_snapshot_command
    SOURCES         test_snapshot_command.cpp
    LINK_LIBRARIES  omniedge_common omniedge_ingest_common
)
