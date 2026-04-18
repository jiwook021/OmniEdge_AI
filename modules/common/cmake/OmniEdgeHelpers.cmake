# =============================================================================
# OmniEdge_AI — Shared CMake Helpers
#
# Provides reusable functions and macros to eliminate duplication across
# module and test CMakeLists.txt files.
#
# Usage: include(OmniEdgeHelpers) in the root CMakeLists.txt.
# =============================================================================

include_guard(GLOBAL)

# Use -march=native for local dev builds (default ON).
# Docker builds pass -DOE_MARCH_NATIVE=OFF → uses portable -march=x86-64-v3.
option(OE_MARCH_NATIVE "Use -march=native instead of portable -march=x86-64-v3" ON)

# -----------------------------------------------------------------------------
# oe_set_target_warnings(<target> [PRIVATE|PUBLIC])
#
# Apply the project's standard warning flags to a target.
# Defaults to PRIVATE visibility.
# -----------------------------------------------------------------------------
function(oe_set_target_warnings TARGET)
    set(VISIBILITY PRIVATE)
    if(ARGN)
        list(GET ARGN 0 VISIBILITY)
    endif()

    # nvcc-generated stub code uses non-standard #line directives that
    # -Wpedantic rejects, so CUDA host compilation stays pedantic-free.
    target_compile_options(${TARGET} ${VISIBILITY}
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic -Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
    )
endfunction()

# -----------------------------------------------------------------------------
# oe_set_target_warnings_no_error(<target> [PRIVATE|PUBLIC])
#
# Same as above but without -Werror. Used for test targets where
# third-party test macros may trigger warnings.
# -----------------------------------------------------------------------------
function(oe_set_target_warnings_no_error TARGET)
    set(VISIBILITY PRIVATE)
    if(ARGN)
        list(GET ARGN 0 VISIBILITY)
    endif()

    target_compile_options(${TARGET} ${VISIBILITY}
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endfunction()

# -----------------------------------------------------------------------------
# oe_set_release_debug_flags(<target>)
#
# Add Release optimizations and Debug sanitizer flags.
# Used by production binaries (not libraries or tests).
# -----------------------------------------------------------------------------
function(oe_set_release_debug_flags TARGET)
    target_compile_options(${TARGET} PRIVATE
        $<$<CONFIG:Release>:-O3 $<IF:$<BOOL:${OE_MARCH_NATIVE}>,-march=native,-march=x86-64-v3>>
    )
    # Debug sanitizers — skip if the root OE_SANITIZE option already adds them
    # globally, to avoid duplicate instrumentation.
    if(NOT OE_SANITIZE)
        target_compile_options(${TARGET} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
        target_link_options(${TARGET} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    endif()
endfunction()

# =============================================================================
# Dependency Finders
#
# These functions resolve common dependencies once and cache the results.
# Call them from the root CMakeLists.txt so every subdirectory inherits them.
# =============================================================================

# -----------------------------------------------------------------------------
# oe_find_yaml_cpp()
#
# Find yaml-cpp via system package, or fetch from GitHub.
# After this call, `yaml-cpp` or `yaml-cpp::yaml-cpp` target is available.
# -----------------------------------------------------------------------------
function(oe_find_yaml_cpp)
    if(TARGET yaml-cpp OR TARGET yaml-cpp::yaml-cpp)
        return()  # Already found or fetched
    endif()

    find_package(yaml-cpp QUIET)
    if(NOT yaml-cpp_FOUND)
        message(STATUS "[OmniEdge] yaml-cpp not found — fetching from GitHub")
        include(FetchContent)
        FetchContent_Declare(yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG        0.8.0
            GIT_SHALLOW    TRUE
        )
        set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(yaml-cpp)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# oe_find_onnxruntime()
#
# Find ONNX Runtime via find_package, manual detection, or env var.
# After this call, `onnxruntime::onnxruntime` IMPORTED target is available,
# or OE_ONNXRUNTIME_FOUND is FALSE.
#
# Search order:
#   1. find_package(onnxruntime)
#   2. Manual detection using ONNXRUNTIME_ROOT or $ENV{ONNXRUNTIME_ROOT}
#   3. Standard system paths
#
# Exports to parent scope:
#   OE_ONNXRUNTIME_FOUND        — TRUE if found
#   OE_ONNXRUNTIME_INCLUDE_DIR  — include path
#   OE_ONNXRUNTIME_LIB          — library path
# -----------------------------------------------------------------------------
function(oe_find_onnxruntime)
    if(TARGET onnxruntime::onnxruntime)
        set(OE_ONNXRUNTIME_FOUND TRUE PARENT_SCOPE)
        return()
    endif()

    # Determine root directory
    set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    if(NOT ORT_ROOT)
        set(ORT_ROOT "/usr/local/onnxruntime")
    endif()

    # Find header
    find_path(OE_ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS
            ${ORT_ROOT}/include
            /usr/local/onnxruntime/include
            /usr/local/include/onnxruntime
            /usr/include/onnxruntime
            /opt/onnxruntime/include
    )

    # Find library (try exact name, then versioned SO)
    find_library(OE_ONNXRUNTIME_LIB onnxruntime
        HINTS
            ${ORT_ROOT}/lib
            /usr/local/onnxruntime/lib
            /usr/local/lib
            /usr/lib
            /opt/onnxruntime/lib
    )
    if(NOT OE_ONNXRUNTIME_LIB)
        file(GLOB _ort_versioned_libs
            "${ORT_ROOT}/lib/libonnxruntime.so.*"
            "/usr/local/onnxruntime/lib/libonnxruntime.so.*"
            "/usr/local/lib/libonnxruntime.so.*"
        )
        if(_ort_versioned_libs)
            list(GET _ort_versioned_libs 0 OE_ONNXRUNTIME_LIB)
        endif()
    endif()

    if(OE_ONNXRUNTIME_INCLUDE_DIR AND OE_ONNXRUNTIME_LIB)
        message(STATUS "[OmniEdge] Found ONNX Runtime: ${OE_ONNXRUNTIME_LIB}")
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION             "${OE_ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${OE_ONNXRUNTIME_INCLUDE_DIR}"
        )
        set(OE_ONNXRUNTIME_FOUND TRUE PARENT_SCOPE)
    else()
        message(STATUS "[OmniEdge] ONNX Runtime not found (header: ${OE_ONNXRUNTIME_INCLUDE_DIR}, lib: ${OE_ONNXRUNTIME_LIB})")
        set(OE_ONNXRUNTIME_FOUND FALSE PARENT_SCOPE)
    endif()

    set(OE_ONNXRUNTIME_INCLUDE_DIR "${OE_ONNXRUNTIME_INCLUDE_DIR}" PARENT_SCOPE)
    set(OE_ONNXRUNTIME_LIB "${OE_ONNXRUNTIME_LIB}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_find_tensorrt()
#
# Find TensorRT (NvInfer.h + libnvinfer).
# After this call, `nvinfer` IMPORTED target may be available.
#
# Exports to parent scope:
#   OE_TENSORRT_FOUND       — TRUE if both header and lib found
#   OE_TENSORRT_INCLUDE_DIR — include path
#   OE_TENSORRT_LIB         — library path
# -----------------------------------------------------------------------------
function(oe_find_tensorrt)
    if(TARGET nvinfer)
        set(OE_TENSORRT_FOUND TRUE PARENT_SCOPE)
        return()
    endif()

    find_path(OE_TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS
            /usr/local/tensorrt/include
            /usr/include/x86_64-linux-gnu
            /usr/include
            /usr/local/include
            /usr/local/cuda/include
    )
    find_library(OE_TENSORRT_LIB nvinfer
        HINTS
            /usr/local/tensorrt/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
    )

    if(OE_TENSORRT_INCLUDE_DIR AND OE_TENSORRT_LIB)
        message(STATUS "[OmniEdge] Found TensorRT: ${OE_TENSORRT_LIB}")
        add_library(nvinfer UNKNOWN IMPORTED GLOBAL)
        set_target_properties(nvinfer PROPERTIES
            IMPORTED_LOCATION             "${OE_TENSORRT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${OE_TENSORRT_INCLUDE_DIR}"
        )
        set(OE_TENSORRT_FOUND TRUE PARENT_SCOPE)
    else()
        message(STATUS "[OmniEdge] TensorRT not found (header: ${OE_TENSORRT_INCLUDE_DIR}, lib: ${OE_TENSORRT_LIB})")
        set(OE_TENSORRT_FOUND FALSE PARENT_SCOPE)
    endif()

    set(OE_TENSORRT_INCLUDE_DIR "${OE_TENSORRT_INCLUDE_DIR}" PARENT_SCOPE)
    set(OE_TENSORRT_LIB "${OE_TENSORRT_LIB}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_find_trtllm()
#
# Find TensorRT-LLM headers + tokenizers-cpp library.
#
# Search order for TRT-LLM headers:
#   1. -DTRTLLM_INCLUDE_DIRS=... (explicit CMake cache)
#   2. $TENSORRT_LLM_HOME/cpp/include (env var from install script)
#   3. find_path() system search
#
# Search order for tokenizers-cpp:
#   1. -DTOKENIZERS_CPP_LIB=... (explicit cache)
#   2. $TOKENIZERS_CPP_HOME/build/libtokenizers_cpp.a (env var)
#   3. find_library() system search
#
# Exports to parent scope:
#   OE_TRTLLM_FOUND             — TRUE if both found
#   OE_TRTLLM_INCLUDE_DIRS      — TRT-LLM header directory
#   OE_TOKENIZERS_CPP_LIB       — tokenizers-cpp library path
#   OE_TOKENIZERS_CPP_INCLUDE   — tokenizers-cpp include directory
# -----------------------------------------------------------------------------
function(oe_find_trtllm)
    # --- TRT-LLM include directory ---
    set(_trtllm_inc "${TRTLLM_INCLUDE_DIRS}")
    if(NOT _trtllm_inc OR NOT EXISTS "${_trtllm_inc}/tensorrt_llm")
        if(DEFINED ENV{TENSORRT_LLM_HOME} AND EXISTS "$ENV{TENSORRT_LLM_HOME}/cpp/include/tensorrt_llm")
            set(_trtllm_inc "$ENV{TENSORRT_LLM_HOME}/cpp/include")
        else()
            find_path(_trtllm_inc_found tensorrt_llm/executor/executor.h
                HINTS "$ENV{HOME}/TensorRT-LLM/cpp/include"
                      "$ENV{HOME}/tensorrt-llm/TensorRT-LLM/cpp/include"
                      /usr/local/include
            )
            if(_trtllm_inc_found)
                set(_trtllm_inc "${_trtllm_inc_found}")
            endif()
        endif()
    endif()

    # --- tokenizers-cpp library ---
    set(_tok_lib "${TOKENIZERS_CPP_LIB}")
    set(_tok_inc "${TOKENIZERS_CPP_INCLUDE_DIRS}")
    if(NOT _tok_lib OR NOT EXISTS "${_tok_lib}")
        if(DEFINED ENV{TOKENIZERS_CPP_HOME} AND EXISTS "$ENV{TOKENIZERS_CPP_HOME}/build/libtokenizers_cpp.a")
            set(_tok_lib "$ENV{TOKENIZERS_CPP_HOME}/build/libtokenizers_cpp.a")
            set(_tok_inc "$ENV{TOKENIZERS_CPP_HOME}/include")
        else()
            find_library(_tok_lib_found NAMES tokenizers_cpp HINTS /usr/local/lib)
            find_path(_tok_inc_found tokenizers_cpp.h HINTS /usr/local/include)
            if(_tok_lib_found)
                set(_tok_lib "${_tok_lib_found}")
            endif()
            if(_tok_inc_found)
                set(_tok_inc "${_tok_inc_found}")
            endif()
        endif()
    endif()

    # --- Verdict ---
    if(EXISTS "${_tok_lib}" AND EXISTS "${_trtllm_inc}/tensorrt_llm")
        set(OE_TRTLLM_FOUND TRUE PARENT_SCOPE)
        message(STATUS "[OmniEdge] TRT-LLM + tokenizers-cpp found")
        message(STATUS "[OmniEdge]   TRT-LLM headers  : ${_trtllm_inc}")
        message(STATUS "[OmniEdge]   tokenizers-cpp   : ${_tok_lib}")
    else()
        set(OE_TRTLLM_FOUND FALSE PARENT_SCOPE)
        message(STATUS "[OmniEdge] TRT-LLM or tokenizers-cpp not found (building with stub inferencer)")
    endif()

    # Cache for reuse by modules and tests
    set(TRTLLM_INCLUDE_DIRS "${_trtllm_inc}" CACHE PATH "TensorRT-LLM include directory" FORCE)
    set(TOKENIZERS_CPP_LIB "${_tok_lib}" CACHE FILEPATH "tokenizers-cpp library" FORCE)
    set(TOKENIZERS_CPP_INCLUDE_DIRS "${_tok_inc}" CACHE PATH "tokenizers-cpp include directory" FORCE)

    set(OE_TRTLLM_INCLUDE_DIRS "${_trtllm_inc}" PARENT_SCOPE)
    set(OE_TOKENIZERS_CPP_LIB "${_tok_lib}" PARENT_SCOPE)
    set(OE_TOKENIZERS_CPP_INCLUDE "${_tok_inc}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_find_espeak_ng()
#
# Find espeak-ng library and headers.
#
# Exports to parent scope:
#   OE_ESPEAK_FOUND        — TRUE if found
#   OE_ESPEAK_INCLUDE_DIR  — include path
#   OE_ESPEAK_LIB          — library path
#   OE_ESPEAK_RUNTIME_DIR  — directory with pcaudio.so.0 / sonic.so.0
# -----------------------------------------------------------------------------
function(oe_find_espeak_ng)
    set(ESPEAK_ROOT "$ENV{ESPEAK_NG_ROOT}")
    if(NOT ESPEAK_ROOT)
        set(ESPEAK_ROOT "/usr")
    endif()

    # Env-var override for dev builds (e.g. ESPEAK_NG_DEV=$HOME/espeak_ng_dev)
    set(_espeak_dev "$ENV{ESPEAK_NG_DEV}")
    if(NOT _espeak_dev)
        set(_espeak_dev "$ENV{HOME}/espeak_ng_dev")
    endif()

    find_path(OE_ESPEAK_INCLUDE_DIR espeak-ng/speak_lib.h
        HINTS
            ${ESPEAK_ROOT}/include
            ${_espeak_dev}/usr/include
    )
    find_library(OE_ESPEAK_LIB espeak-ng
        HINTS
            ${ESPEAK_ROOT}/lib
            ${ESPEAK_ROOT}/lib/x86_64-linux-gnu
            ${_espeak_dev}/usr/lib/x86_64-linux-gnu
    )

    set(OE_ESPEAK_RUNTIME_DIR "${_espeak_dev}/runtime/usr/lib/x86_64-linux-gnu")

    if(OE_ESPEAK_INCLUDE_DIR AND OE_ESPEAK_LIB)
        message(STATUS "[OmniEdge] Found espeak-ng: ${OE_ESPEAK_LIB}")
        set(OE_ESPEAK_FOUND TRUE PARENT_SCOPE)
    else()
        message(STATUS "[OmniEdge] espeak-ng not found")
        set(OE_ESPEAK_FOUND FALSE PARENT_SCOPE)
    endif()

    set(OE_ESPEAK_INCLUDE_DIR "${OE_ESPEAK_INCLUDE_DIR}" PARENT_SCOPE)
    set(OE_ESPEAK_LIB "${OE_ESPEAK_LIB}" PARENT_SCOPE)
    set(OE_ESPEAK_RUNTIME_DIR "${OE_ESPEAK_RUNTIME_DIR}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_find_python_site_packages()
#
# Detect the Python user site-packages directory (used for locating
# TRT-LLM, PyTorch, and NCCL shared libraries installed via pip).
#
# Exports to parent scope:
#   OE_PYTHON_SITE_PACKAGES — e.g. /home/user/.local/lib/python3.10/site-packages
# -----------------------------------------------------------------------------
function(oe_find_python_site_packages)
    if(DEFINED OE_PYTHON_SITE_PACKAGES AND OE_PYTHON_SITE_PACKAGES)
        return()
    endif()

    # Use system site-packages (works with both NGC and local pip installs).
    # getusersitepackages() misses packages installed system-wide in NGC images.
    execute_process(
        COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE _site_packages
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(OE_PYTHON_SITE_PACKAGES "${_site_packages}" CACHE PATH
        "Python system site-packages directory" FORCE)
    set(OE_PYTHON_SITE_PACKAGES "${_site_packages}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_setup_trtllm_link_flags(<target>)
#
# Configure RPATH, PyTorch CUDA 12 runtime, TRT-LLM, NCCL, tokenizers,
# CUDA driver, and CUPTI linking for a target that uses TRT-LLM.
#
# Requires: OE_PYTHON_SITE_PACKAGES, OE_TRTLLM_INCLUDE_DIRS,
#           OE_TOKENIZERS_CPP_LIB, OE_TOKENIZERS_CPP_INCLUDE to be set.
# -----------------------------------------------------------------------------
function(oe_setup_trtllm_link_flags TARGET)
    oe_find_python_site_packages()

    set(TORCH_DIR     "${OE_PYTHON_SITE_PACKAGES}/torch/lib")
    set(NCCL_DIR      "${OE_PYTHON_SITE_PACKAGES}/nvidia/nccl/lib")
    set(CUDART12_DIR  "${OE_PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib")
    set(TRTLLM_SO     "${OE_PYTHON_SITE_PACKAGES}/tensorrt_llm/libs/libtensorrt_llm.so")
    set(CUPTI_DIR     "${OE_PYTHON_SITE_PACKAGES}/nvidia/cuda_cupti/lib")

    # TRT-LLM and tokenizers-cpp include paths
    target_include_directories(${TARGET} PRIVATE
        ${TRTLLM_INCLUDE_DIRS}
        ${TOKENIZERS_CPP_INCLUDE_DIRS}
    )

    # PyTorch CUDA 12 runtime (for libc10_cuda.so compatibility)
    if(EXISTS "${CUDART12_DIR}/libcudart.so.12")
        target_link_directories(${TARGET} PRIVATE ${CUDART12_DIR})
        target_link_libraries(${TARGET} PRIVATE "${CUDART12_DIR}/libcudart.so.12")
        target_link_options(${TARGET} PRIVATE "LINKER:-rpath,${CUDART12_DIR}")
    else()
        target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
    endif()

    # RPATH for PyTorch, NCCL, and TRT-LLM shared libraries.
    # --disable-new-dtags sets RPATH (inherited by transitive deps)
    # instead of RUNPATH (not inherited — causes libnccl.so.2 load failure).
    target_link_options(${TARGET} PRIVATE
        "LINKER:--disable-new-dtags"
        "LINKER:-rpath,${TORCH_DIR}"
        "LINKER:-rpath,${NCCL_DIR}"
        "LINKER:-rpath,${OE_PYTHON_SITE_PACKAGES}/tensorrt_llm/libs"
        "LINKER:-rpath,${CUPTI_DIR}"
    )
    if(EXISTS "/usr/lib/wsl/lib")
        target_link_options(${TARGET} PRIVATE "LINKER:-rpath,/usr/lib/wsl/lib")
    endif()

    # PyTorch + NCCL shared libraries
    set(TORCH_LIBS
        ${TORCH_DIR}/libtorch.so
        ${TORCH_DIR}/libtorch_cpu.so
        ${TORCH_DIR}/libtorch_cuda.so
        ${TORCH_DIR}/libc10.so
        ${TORCH_DIR}/libc10_cuda.so
        ${TORCH_DIR}/libtorch_python.so
        ${NCCL_DIR}/libnccl.so.2
    )

    # TRT-LLM library
    target_link_libraries(${TARGET} PRIVATE ${TRTLLM_SO} ${TORCH_LIBS})

    # Tokenizers companion libraries (tokenizers_c, sentencepiece)
    target_link_libraries(${TARGET} PRIVATE ${TOKENIZERS_CPP_LIB})
    find_library(_tokenizers_c_lib NAMES tokenizers_c HINTS /usr/local/lib)
    find_library(_sentencepiece_lib NAMES sentencepiece HINTS /usr/local/lib)
    if(_tokenizers_c_lib AND EXISTS "${_tokenizers_c_lib}")
        target_link_libraries(${TARGET} PRIVATE ${_tokenizers_c_lib})
    endif()
    if(_sentencepiece_lib AND EXISTS "${_sentencepiece_lib}")
        target_link_libraries(${TARGET} PRIVATE ${_sentencepiece_lib})
    endif()

    # CUDA driver (WSL2 path) and CUPTI
    set(_cuda_driver "/usr/lib/wsl/lib/libcuda.so.1")
    set(_cupti_lib "${CUPTI_DIR}/libcupti.so.12")
    if(EXISTS "${_cuda_driver}")
        target_link_libraries(${TARGET} PRIVATE "${_cuda_driver}")
    elseif(TARGET CUDA::cuda_driver)
        target_link_libraries(${TARGET} PRIVATE CUDA::cuda_driver)
    endif()
    if(EXISTS "${_cupti_lib}")
        target_link_libraries(${TARGET} PRIVATE "${_cupti_lib}")
    elseif(TARGET CUDA::cupti)
        target_link_libraries(${TARGET} PRIVATE CUDA::cupti)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# oe_trtllm_ld_library_path()
#
# Returns a colon-separated LD_LIBRARY_PATH string needed at runtime
# for TRT-LLM binaries/tests. Use with set_tests_properties(ENVIRONMENT).
# -----------------------------------------------------------------------------
function(oe_trtllm_ld_library_path OUT_VAR)
    oe_find_python_site_packages()

    set(TORCH_DIR    "${OE_PYTHON_SITE_PACKAGES}/torch/lib")
    set(NCCL_DIR     "${OE_PYTHON_SITE_PACKAGES}/nvidia/nccl/lib")
    set(CUDART12_DIR "${OE_PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib")
    set(CUPTI_DIR    "${OE_PYTHON_SITE_PACKAGES}/nvidia/cuda_cupti/lib")
    set(TRTLLM_DIR   "${OE_PYTHON_SITE_PACKAGES}/tensorrt_llm/libs")

    set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    if(NOT ORT_ROOT)
        set(ORT_ROOT "/usr/local/onnxruntime")
    endif()

    set(LD_PATH "/usr/lib/x86_64-linux-gnu:${CUPTI_DIR}:${TORCH_DIR}:${NCCL_DIR}:${TRTLLM_DIR}:${CUDART12_DIR}:${ORT_ROOT}/lib")
    if(EXISTS "/usr/lib/wsl/lib")
        set(LD_PATH "/usr/lib/wsl/lib:${LD_PATH}")
    endif()
    set(_espeak_dev "$ENV{HOME}/espeak_ng_dev/runtime/usr/lib/x86_64-linux-gnu")
    if(EXISTS "${_espeak_dev}")
        set(LD_PATH "${LD_PATH}:${_espeak_dev}")
    endif()

    set(${OUT_VAR} "${LD_PATH}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# oe_find_tracy()
#
# Find or fetch Tracy profiler (https://github.com/wolfpld/tracy).
# Only active when OE_ENABLE_TRACY is ON.
#
# When enabled, creates the `TracyClient` target (static library) and
# propagates the TRACY_ENABLE compile definition so that Tracy.hpp macros
# expand to real instrumentation.  When disabled this function is a no-op.
#
# Usage: call from root CMakeLists.txt after the OE_ENABLE_TRACY option.
# Downstream targets link TracyClient (or inherit it via omniedge_common).
# -----------------------------------------------------------------------------
function(oe_find_tracy)
    if(NOT OE_ENABLE_TRACY)
        return()
    endif()

    # Try system-installed Tracy first
    find_package(Tracy QUIET)
    if(Tracy_FOUND AND TARGET TracyClient)
        message(STATUS "[OmniEdge] Found system Tracy")
        return()
    endif()

    # Fetch from GitHub
    message(STATUS "[OmniEdge] Tracy not found — fetching v0.11.1 from GitHub")
    include(FetchContent)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        v0.11.1
        GIT_SHALLOW    TRUE
    )
    set(TRACY_ENABLE  ON  CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "Connect only when profiler GUI attaches" FORCE)
    FetchContent_MakeAvailable(tracy)
endfunction()

# =============================================================================
# Node Target Helpers
#
# Shared 3-section pattern (common lib + executable + warnings/flags)
# for modules/nodes/*/CMakeLists.txt files.
# =============================================================================

# -----------------------------------------------------------------------------
# oe_add_node_common(<lib_name>
#     SOURCES <source_files...>
#     CORE_LIB <core_library>
#     [EXTRA_LIBS <libraries...>]
# )
#
# Creates a STATIC library for the node (shared with tests).
# Automatically adds include directory and links omniedge_common.
# -----------------------------------------------------------------------------
function(oe_add_node_common LIB_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 _NC
        ""
        "CORE_LIB"
        "SOURCES;EXTRA_LIBS"
    )

    add_library(${LIB_NAME} STATIC ${_NC_SOURCES})
    target_include_directories(${LIB_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(${LIB_NAME} PUBLIC
        ${_NC_CORE_LIB}
        omniedge_common
        ${_NC_EXTRA_LIBS}
    )
    oe_set_target_warnings(${LIB_NAME})
endfunction()

# -----------------------------------------------------------------------------
# oe_add_node_exe(<exe_name>
#     SOURCES <source_files...>
#     COMMON_LIB <common_library>
#     [EXTRA_LIBS <libraries...>]
# )
#
# Creates an executable target with standard warnings and release/debug flags.
# Links the common library and yaml-cpp automatically.
# -----------------------------------------------------------------------------
function(oe_add_node_exe EXE_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 _NE
        ""
        "COMMON_LIB"
        "SOURCES;EXTRA_LIBS"
    )

    add_executable(${EXE_NAME} ${_NE_SOURCES})
    target_compile_features(${EXE_NAME} PRIVATE cxx_std_20)
    target_link_libraries(${EXE_NAME} PRIVATE
        ${_NE_COMMON_LIB}
        yaml-cpp
        CLI11::CLI11
        ${_NE_EXTRA_LIBS}
    )
    oe_set_target_warnings(${EXE_NAME})
    oe_set_release_debug_flags(${EXE_NAME})
endfunction()


# =============================================================================
# Test Helpers
# =============================================================================

# -----------------------------------------------------------------------------
# oe_add_test(<test_name>
#     SOURCES <source_files...>
#     LINK_LIBRARIES <libraries...>
#     [INCLUDE_DIRS <directories...>]
#     [COMPILE_DEFS <definitions...>]
#     [RESOURCE_LOCK <lock_name>]
#     [LABELS <label1> <label2> ...]
#     [EXCLUDE_FROM_ALL]
#     [GPU_SPLIT <gpu_test_filter>]
# )
#
# Creates a GTest executable with standard OmniEdge compile settings.
# If GPU_SPLIT is provided, creates two test registrations:
#   - CPU tests: excluding the filter pattern
#   - GPU tests: matching the filter pattern (with LABELS gpu)
# If RESOURCE_LOCK is provided, uses add_test() instead of gtest_discover_tests.
# -----------------------------------------------------------------------------
function(oe_add_test TEST_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 OE_TEST
        "EXCLUDE_FROM_ALL"
        "RESOURCE_LOCK;GPU_SPLIT"
        "SOURCES;LINK_LIBRARIES;INCLUDE_DIRS;COMPILE_DEFS;LABELS"
    )

    if(OE_TEST_EXCLUDE_FROM_ALL)
        add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${OE_TEST_SOURCES})
    else()
        add_executable(${TEST_NAME} ${OE_TEST_SOURCES})
    endif()

    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    oe_set_target_warnings_no_error(${TEST_NAME})

    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${OE_TEST_INCLUDE_DIRS}
    )

    if(OE_TEST_COMPILE_DEFS)
        target_compile_definitions(${TEST_NAME} PRIVATE ${OE_TEST_COMPILE_DEFS})
    endif()

    target_link_libraries(${TEST_NAME} PRIVATE
        GTest::gtest_main
        ${OE_TEST_LINK_LIBRARIES}
    )

    include(GoogleTest)

    if(OE_TEST_EXCLUDE_FROM_ALL)
        # EXCLUDE_FROM_ALL targets: skip gtest_discover_tests
        return()
    endif()

    # WSL2 CUDA driver fix: propagate LD_LIBRARY_PATH to all tests
    set(_oe_env_props "")
    if(OE_WSL2_TEST_ENV)
        set(_oe_env_props ENVIRONMENT "${OE_WSL2_TEST_ENV}")
    endif()

    if(OE_TEST_GPU_SPLIT)
        # Split into CPU tests (always run) and GPU tests (tagged + serialized)
        gtest_discover_tests(${TEST_NAME}
            TEST_FILTER "-${OE_TEST_GPU_SPLIT}.*"
            PROPERTIES ${_oe_env_props}
        )
        gtest_discover_tests(${TEST_NAME}
            TEST_FILTER "${OE_TEST_GPU_SPLIT}.*"
            PROPERTIES LABELS gpu RESOURCE_LOCK gpu_device ${_oe_env_props}
        )
    elseif(OE_TEST_RESOURCE_LOCK)
        # Use add_test() for tests that need RESOURCE_LOCK
        add_test(NAME ${TEST_NAME}
            COMMAND ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set(PROPS RESOURCE_LOCK ${OE_TEST_RESOURCE_LOCK})
        if(OE_TEST_LABELS)
            list(APPEND PROPS LABELS "${OE_TEST_LABELS}")
        endif()
        if(_oe_env_props)
            list(APPEND PROPS ${_oe_env_props})
        endif()
        set_tests_properties(${TEST_NAME} PROPERTIES ${PROPS})
    elseif(OE_TEST_LABELS)
        gtest_discover_tests(${TEST_NAME}
            PROPERTIES LABELS "${OE_TEST_LABELS}" ${_oe_env_props}
        )
    else()
        gtest_discover_tests(${TEST_NAME}
            PROPERTIES ${_oe_env_props}
        )
    endif()
endfunction()
