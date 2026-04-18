# =========================================================================
# tests/integration/hal — Real GPU, VRAM, pinned memory, compositor
#
# GTest and CUDAToolkit are resolved at root — no find_package() needed.
# =========================================================================

# =========================================================================
# Standard tests using oe_add_test()
# =========================================================================
oe_add_test(test_gpu_tier_probe
    SOURCES     test_gpu_tier_probe.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   ProbeGpuTest
)

oe_add_test(test_vram_tracker
    SOURCES     test_vram_tracker.cpp
    LINK_LIBRARIES  omniedge_common
)

oe_add_test(test_vram_gate
    SOURCES     test_vram_gate.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   VramGateGpuTest
)

oe_add_test(test_pinned_staging_buffer
    SOURCES     test_pinned_staging_buffer.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   PinnedStagingBufferGpuTest
)

oe_add_test(test_gpu_diagnostics
    SOURCES     test_gpu_diagnostics.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   ProbeGpuDiagnosticsTest
)

oe_add_test(test_cuda_guard
    SOURCES     test_cuda_guard.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   CudaGuardGpuTest
)

oe_add_test(test_cuda_fence
    SOURCES     test_cuda_fence.cpp
    LINK_LIBRARIES  omniedge_common CUDA::cudart
    GPU_SPLIT   CudaFenceGpuTest
)

# =========================================================================
# Accepted deviation: GPU compositor uses raw add_executable
#   - CUDA .cu file needs CUDA_SEPARABLE_COMPILATION property
#   - Not supported by oe_add_test()
# =========================================================================
find_package(OpenCV REQUIRED COMPONENTS core)

add_executable(test_gpu_compositor
    test_gpu_compositor.cu
    ${CMAKE_SOURCE_DIR}/modules/hal/gpu/src/gpu_compositor.cu
)
target_compile_features(test_gpu_compositor PRIVATE cxx_std_20)
target_compile_options(test_gpu_compositor PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:  -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -Wall,-Wextra>
)
target_include_directories(test_gpu_compositor PRIVATE
    ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_gpu_compositor PRIVATE
    GTest::gtest_main omniedge_common ${OpenCV_LIBS} CUDA::cudart
)
set_target_properties(test_gpu_compositor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME GpuCompositorTests COMMAND test_gpu_compositor)
set_tests_properties(GpuCompositorTests PROPERTIES LABELS gpu RESOURCE_LOCK gpu_device)
