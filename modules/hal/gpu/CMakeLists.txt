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
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_LIBRARY_DIR}/stubs
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/stubs
        /usr/local/cuda/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/stubs
        /usr/local/cuda/lib64/stubs
        /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
        /usr/lib/wsl/lib
        /usr/local/cuda/lib64
)
if(NOT NVML_LIB)
    message(FATAL_ERROR
        "libnvidia-ml not found. Install the NVIDIA driver or CUDA toolkit stubs.\n"
        "  CUDAToolkit_LIBRARY_DIR = ${CUDAToolkit_LIBRARY_DIR}\n"
        "  CUDA_TOOLKIT_ROOT_DIR  = ${CUDA_TOOLKIT_ROOT_DIR}")
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
# Section 7 — CUDA compositor kernel
#
# Uses only opencv_core (for cv::cuda::GpuMat as a device-memory handle).
# Gaussian blur, ISP BCS/sharpen and mask composite are implemented as
# native CUDA kernels inside gpu_compositor.cu, so the optional OpenCV
# CUDA contrib modules (cudaarithm / cudafilters) are not required.
# =========================================================================
find_package(OpenCV QUIET COMPONENTS core)

if(OpenCV_FOUND AND TARGET opencv_core)
    add_library(omniedge_hal_gpu_cuda STATIC src/gpu_compositor.cu)

    target_include_directories(omniedge_hal_gpu_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omniedge_hal_gpu_cuda PUBLIC
        omniedge_logger CUDA::cudart opencv_core
    )
    set_target_properties(omniedge_hal_gpu_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    oe_set_target_warnings_no_error(omniedge_hal_gpu_cuda)
else()
    # INTERFACE placeholder so dependents don't break
    add_library(omniedge_hal_gpu_cuda INTERFACE)
    message(STATUS "[hal/gpu] OpenCV core not found — gpu_compositor disabled")
endif()
