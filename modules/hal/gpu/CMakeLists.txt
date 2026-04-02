# =========================================================================
# Section 1 — Comment Header
# modules/hal/gpu — CUDA streams, pinned buffers, GPU tier, profiler,
#                   compositor
#
# Targets:
#   omniedge_hal_gpu      — STATIC (C++ sources)
#   omniedge_hal_gpu_cuda — STATIC (CUDA .cu, separable compilation)
# =========================================================================

# =========================================================================
# Section 2 — Required dependency detection
# =========================================================================
find_library(NVML_LIB nvidia-ml
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib/wsl/lib /usr/local/cuda/lib64
          /usr/local/cuda/targets/x86_64-linux/lib/stubs)
if(NOT NVML_LIB)
    message(FATAL_ERROR
        "libnvidia-ml not found. Install the NVIDIA driver or CUDA toolkit stubs.\n"
        "  Searched: /usr/lib/x86_64-linux-gnu, /usr/lib/wsl/lib, /usr/local/cuda/lib64")
endif()

# =========================================================================
# Section 3 — Source files + library declaration
# =========================================================================
add_library(omniedge_hal_gpu STATIC
    src/gpu_tier.cpp
    src/gpu_diagnostics.cpp
    src/gpu_profiler.cpp
)

# =========================================================================
# Section 4 — Include directories
# =========================================================================
target_include_directories(omniedge_hal_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# gpu_tier.hpp includes vram/vram_thresholds.hpp (header-only).
# Add vram include path directly to avoid circular dep (vram lib depends on gpu).
target_include_directories(omniedge_hal_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/hal/vram/include>
)

# =========================================================================
# Section 5 — Link libraries
# =========================================================================
target_link_libraries(omniedge_hal_gpu PUBLIC
    omniedge_logger
    CUDA::cudart
    ${NVML_LIB}
)

# =========================================================================
# Section 6 — Warning flags
#   - _no_error: GCC 13 -Warray-bounds false positive in fmt/spdlog with -O2
# =========================================================================
oe_set_target_warnings_no_error(omniedge_hal_gpu)

# =========================================================================
# Section 7 — CUDA compositor kernel (optional, requires OpenCV CUDA)
# =========================================================================
find_package(OpenCV QUIET COMPONENTS core cudaarithm)

if(OpenCV_FOUND AND TARGET opencv_cudaarithm)
    add_library(omniedge_hal_gpu_cuda STATIC src/gpu_compositor.cu)

    target_include_directories(omniedge_hal_gpu_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omniedge_hal_gpu_cuda PUBLIC
        omniedge_logger CUDA::cudart ${OpenCV_LIBS}
    )
    set_target_properties(omniedge_hal_gpu_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    oe_set_target_warnings(omniedge_hal_gpu_cuda)
else()
    # INTERFACE placeholder so dependents don't break
    add_library(omniedge_hal_gpu_cuda INTERFACE)
    message(STATUS "[hal/gpu] OpenCV CUDA modules not found — gpu_compositor disabled")
endif()
