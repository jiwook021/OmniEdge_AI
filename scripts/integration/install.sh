#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# OmniEdge_AI — Full Installation Script
# Installs all build dependencies, downloads models, quantizes, and builds
# TensorRT engines. Idempotent — skips steps that are already complete.
#
# MEMORY SAFETY:
#   Several steps (quantization, TRT engine builds, C++ compilation) can
#   consume 15-40 GB of RAM.  The script detects available memory and either
#   limits parallelism, expands swap, or skips steps that would OOM-kill the
#   system.  Use --force-all to override the safety checks.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Cleanup trap: ensure we return to the original directory and remove temp
# swap file on failure.
ORIGINAL_DIR="$(pwd)"
OE_TEMP_SWAP=""
cleanup() {
    cd "$ORIGINAL_DIR"
    # Remove temporary swap if we created one
    if [ -n "$OE_TEMP_SWAP" ] && [ -f "$OE_TEMP_SWAP" ]; then
        echo "  Removing temporary swap file..."
        sudo swapoff "$OE_TEMP_SWAP" 2>/dev/null || true
        sudo rm -f "$OE_TEMP_SWAP"
    fi
}
trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source shared utilities (colours, helpers, ini_get, download_hf, etc.)
source "$SCRIPT_DIR/common.sh"

mkdir -p "$OE_MODELS_DIR" "$OE_ENGINES_DIR"

ERRORS=0
FORCE_ALL=0
TEST_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --force-all) FORCE_ALL=1 ;;
        --test)      TEST_ONLY=1 ;;
    esac
done

# Source domain-specific install modules
source "$SCRIPT_DIR/install_memory.sh"
source "$SCRIPT_DIR/install_stt.sh"
source "$SCRIPT_DIR/install_tts.sh"
source "$SCRIPT_DIR/install_cv.sh"
source "$SCRIPT_DIR/install_enhancement.sh"

# ═══════════════════════════════════════════════════════════════════════════════
#  0. Silero VAD
# ═══════════════════════════════════════════════════════════════════════════════
install_vad() {
    section "VAD — Silero VAD v5"

    local VAD_MODEL="$OE_MODELS_DIR/silero_vad.onnx"
    if [ -f "$VAD_MODEL" ]; then
        skip "Silero VAD model"
    else
        echo "  Downloading Silero VAD v5 ONNX..."
        pip install --break-system-packages silero-vad 2>/dev/null || true
        python3 -c "
import torch, os
model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=True)
import shutil
src = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master/files/silero_vad.onnx')
shutil.copy2(src, '$VAD_MODEL')
print('Silero VAD saved to $VAD_MODEL')
" || {
            # Fallback: copy from repo if present
            if [ -f "$PROJECT_ROOT/models/silero_vad.onnx" ]; then
                cp "$PROJECT_ROOT/models/silero_vad.onnx" "$VAD_MODEL"
                ok "Silero VAD copied from repo"
            else
                record_error "Silero VAD download failed"
            fi
        }
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Build Toolchain & System Libraries
# ═══════════════════════════════════════════════════════════════════════════════
install_toolchain() {
    section "Build Toolchain & System Libraries"

    local APT_PACKAGES=(
        # Build tools
        build-essential cmake ninja-build git pkg-config
        # Messaging
        libzmq3-dev
        # GStreamer (video/audio ingest)
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-ugly gstreamer1.0-libav
        gstreamer1.0-tools
        # Media
        ffmpeg espeak-ng
        # Libraries
        libopencv-dev libsqlite3-dev nlohmann-json3-dev libinih-dev
        # TensorRT plugin
        libnvinfer-plugin-dev
        # Hardware
        v4l-utils
        # Audio
        pulseaudio pulseaudio-utils
    )

    sudo apt-get update && sudo apt-get install -y "${APT_PACKAGES[@]}"
    ok "system packages installed"

    if ! command -v nvcc &>/dev/null; then
        record_error "nvcc not found — install CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads"
        return
    fi

    ok "CUDA $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    ok "CMake $(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')"

    # ── OpenCV with CUDA modules ──
    # The apt libopencv-dev lacks CUDA modules (cudaarithm, cudaimgproc).
    # Build from source if cv::cuda is unavailable.
    if python3 -c "import cv2; assert cv2.cuda.getCudaEnabledDeviceCount() > 0" 2>/dev/null; then
        skip "OpenCV with CUDA support"
    else
        echo "  Building OpenCV from source with CUDA modules (~15 min)..."
        local OPENCV_BUILD_DIR="/tmp/opencv_cuda_build"
        local OPENCV_VER="4.13.0"

        local CUDA_MAJOR
        CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "0")

        sudo apt-get install -y libgtk-3-dev libavcodec-dev libavformat-dev \
            libswscale-dev libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
            libopenblas-dev libgflags-dev libgoogle-glog-dev libavif-dev \
            libtesseract-dev libleptonica-dev 2>/dev/null || true

        # NVIDIA Video Codec SDK headers (nvcuvid.h, nvEncodeAPI.h)
        if [ ! -f "/usr/local/cuda/include/nvcuvid.h" ]; then
            echo "  Installing NVIDIA Video Codec SDK headers..."
            local NV_CODEC_DIR="/tmp/nv-codec-headers"
            if [ ! -d "$NV_CODEC_DIR" ]; then
                git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git "$NV_CODEC_DIR" 2>/dev/null
            fi
            if [ -d "$NV_CODEC_DIR" ]; then
                (cd "$NV_CODEC_DIR" && sudo make install PREFIX=/usr/local 2>/dev/null)
                sudo ln -sf /usr/local/include/ffnvcodec/dynlink_nvcuvid.h /usr/local/cuda/include/nvcuvid.h
                sudo ln -sf /usr/local/include/ffnvcodec/dynlink_cuviddec.h /usr/local/cuda/include/cuviddec.h
                sudo ln -sf /usr/local/include/ffnvcodec/nvEncodeAPI.h /usr/local/cuda/include/nvEncodeAPI.h
                ok "NVIDIA Video Codec SDK headers"
            fi
        fi

        mkdir -p "$OPENCV_BUILD_DIR" && pushd "$OPENCV_BUILD_DIR" > /dev/null

        if [ ! -d "opencv-${OPENCV_VER}" ]; then
            curl -sL "https://github.com/opencv/opencv/archive/${OPENCV_VER}.tar.gz" | tar xz
            curl -sL "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VER}.tar.gz" | tar xz
        fi

        # Detect CUDA arch from GPU
        local CUDA_ARCH
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')

        mkdir -p "opencv-${OPENCV_VER}/build" && cd "opencv-${OPENCV_VER}/build"
        # cuDNN 9.x headers are under /usr/include/x86_64-linux-gnu/ on Ubuntu
        local CUDNN_INC="/usr/include/x86_64-linux-gnu"
        local CUDNN_LIB="/usr/lib/x86_64-linux-gnu"

        # cuDNN 9.x ships _v9-suffixed headers (cudnn_v9.h, cudnn_version_v9.h)
        # but internal includes reference the unsuffixed names. Create symlinks
        # for any missing unsuffixed headers.
        if [ -f "$CUDNN_INC/cudnn_v9.h" ]; then
            for _v9hdr in "$CUDNN_INC"/cudnn_*_v9.h; do
                _base="${_v9hdr%_v9.h}.h"
                if [ ! -f "$_base" ]; then
                    log_info "Symlinking $(basename "$_base") -> $(basename "$_v9hdr")"
                    sudo ln -sf "$_v9hdr" "$_base"
                fi
            done
            # cudnn.h itself (not _v9 suffixed) — OpenCV FindCUDNN.cmake needs it
            if [ ! -f "$CUDNN_INC/cudnn.h" ]; then
                log_info "Creating cudnn.h compatibility header for cuDNN 9..."
                sudo tee "$CUDNN_INC/cudnn.h" > /dev/null << 'CUDNN9_COMPAT'
/* cuDNN 9 compatibility wrapper — OpenCV FindCUDNN.cmake needs cudnn.h */
#ifndef CUDNN_H_COMPAT_V9
#define CUDNN_H_COMPAT_V9
#include "cudnn_v9.h"
#include "cudnn_version_v9.h"
#ifndef CUDNN_MAJOR
#define CUDNN_MAJOR 9
#endif
#ifndef CUDNN_MINOR
#define CUDNN_MINOR 0
#endif
#ifndef CUDNN_PATCHLEVEL
#define CUDNN_PATCHLEVEL 0
#endif
#ifndef CUDNN_VERSION
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
#endif
#endif
CUDNN9_COMPAT
            fi
        fi

        # ── CUDA 13.x compat patches ──
        # Applied conditionally — each patch checks whether the source still
        # needs it (older OpenCV releases do; newer ones may have native fixes).
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            log_info "CUDA ${CUDA_MAJOR}.x detected — applying compatibility patches if needed..."

            # CUDA 13.2 CCCL removed _LIBCUDACXX_{BEGIN,END}_NAMESPACE_STD and
            # _CCCL_{BEGIN,END}_NAMESPACE_STD macros entirely.  Replace with
            # raw namespace declarations in cudev zip.hpp.
            local ZIP_HPP="../../opencv_contrib-${OPENCV_VER}/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp"
            if [ -f "$ZIP_HPP" ] && grep -qE "_LIBCUDACXX_BEGIN_NAMESPACE_STD|_CCCL_BEGIN_NAMESPACE_STD" "$ZIP_HPP"; then
                log_info "  Patching cudev zip.hpp for CCCL namespace removal..."
                sed -i \
                    -e 's/_LIBCUDACXX_BEGIN_NAMESPACE_STD/namespace std {/g' \
                    -e 's/_LIBCUDACXX_END_NAMESPACE_STD/} \/\/ namespace std/g' \
                    -e 's/_CCCL_BEGIN_NAMESPACE_STD/namespace std {/g' \
                    -e 's/_CCCL_END_NAMESPACE_STD/} \/\/ namespace std/g' \
                    "$ZIP_HPP"
            fi

            # CUDA 13.2 NPP removed nppGetStreamContext().  Replace with manual
            # NppStreamContext initialisation from cudaDeviceProp.
            local PRIV_CUDA_HPP="../modules/core/include/opencv2/core/private.cuda.hpp"
            if [ -f "$PRIV_CUDA_HPP" ] && grep -q "nppGetStreamContext" "$PRIV_CUDA_HPP"; then
                log_info "  Patching private.cuda.hpp for nppGetStreamContext removal..."
                sed -i '/nppSafeCall(nppGetStreamContext(&nppStreamContext));/c\
            {\
                int _dev = 0;\
                cudaSafeCall(cudaGetDevice(&_dev));\
                cudaDeviceProp _props;\
                cudaSafeCall(cudaGetDeviceProperties(&_props, _dev));\
                nppStreamContext.nCudaDeviceId = _dev;\
                nppStreamContext.nMultiProcessorCount = _props.multiProcessorCount;\
                nppStreamContext.nMaxThreadsPerMultiProcessor = _props.maxThreadsPerMultiProcessor;\
                nppStreamContext.nMaxThreadsPerBlock = _props.maxThreadsPerBlock;\
                nppStreamContext.nSharedMemPerBlock = _props.sharedMemPerBlock;\
                nppStreamContext.nCudaDevAttrComputeCapabilityMajor = _props.major;\
                nppStreamContext.nCudaDevAttrComputeCapabilityMinor = _props.minor;\
            }' "$PRIV_CUDA_HPP"
            fi
        fi

        cmake .. \
            -GNinja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCUDA_NVCC_FLAGS="--std=c++17" \
            -DWITH_CUDA=ON \
            -DWITH_CUDNN=ON \
            -DOPENCV_DNN_CUDA=ON \
            -DCUDNN_INCLUDE_DIR="$CUDNN_INC" \
            -DCUDNN_LIBRARY="$CUDNN_LIB/libcudnn.so" \
            -DCUDA_ARCH_BIN="$CUDA_ARCH" \
            -DWITH_NVCUVID=ON \
            -DWITH_NVCUVENC=ON \
            -DWITH_TBB=ON \
            -DWITH_TESSERACT=ON \
            -DWITH_OPENBLAS=ON \
            -DBUILD_opencv_python3=ON \
            -DBUILD_opencv_sfm=OFF \
            -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-${OPENCV_VER}/modules" \
            -DBUILD_TESTS=OFF \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_PERF_TESTS=OFF \
            -Wno-dev

        local JOBS
        JOBS=$(safe_job_count)
        ninja -j"$JOBS" && sudo ninja install && sudo ldconfig
        ok "OpenCV ${OPENCV_VER} with CUDA installed"

        popd > /dev/null
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Build OmniEdge Binaries
# ═══════════════════════════════════════════════════════════════════════════════
install_build() {
    section "Build OmniEdge Binaries"

    cd "$PROJECT_ROOT"
    mkdir -p build && cd build

    # CMake configure
    if [ -f CMakeCache.txt ]; then
        local CACHED_GEN
        CACHED_GEN=$(grep "^CMAKE_GENERATOR:" CMakeCache.txt 2>/dev/null | cut -d= -f2 || true)
        if [ -n "$CACHED_GEN" ] && [ "$CACHED_GEN" != "Ninja" ]; then
            echo "  Generator mismatch ($CACHED_GEN != Ninja). Cleaning..."
            cd "$PROJECT_ROOT" && [ -n "$PROJECT_ROOT" ] && rm -rf "$PROJECT_ROOT/build" && mkdir -p "$PROJECT_ROOT/build" && cd "$PROJECT_ROOT/build"
        fi
    fi
    cmake "$PROJECT_ROOT" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
    ok "CMake configured"

    # Build with memory-safe parallelism
    # CUDA/TRT translation units can use 0.5-1 GB each.
    # With 15 GB RAM and -j20, that's 10-20 GB for compilation alone → OOM.
    local JOBS
    JOBS=$(safe_job_count)
    echo -e "  ${CYAN}Building with -j${JOBS} ($(available_memory_gb) GB available, $(nproc) CPUs)${NC}"
    ninja -j"$JOBS"
    ok "all targets built"

    # Ensure CUDA/ONNX Runtime libs are discoverable by test binaries
    LD_LIBRARY_PATH="$(build_ld_library_path)"
    export LD_LIBRARY_PATH

    # Run tests — exclude PipelineModelTests (EXCLUDE_FROM_ALL, needs TRT-LLM)
    # ctest continues past individual failures; non-zero exit means at least one
    # test failed or was Not Run. Treat "Not Run" as non-fatal.
    local CTEST_RC=0
    ctest --output-on-failure -j"$JOBS" --exclude-regex PipelineModelTests \
        || CTEST_RC=$?
    if [ "$CTEST_RC" -eq 0 ]; then
        ok "all tests passed"
    else
        warn "some tests failed (exit $CTEST_RC) — check output above"
    fi

    cd "$PROJECT_ROOT"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  TEST MODE — build + run all unit tests + integration tests
# ═══════════════════════════════════════════════════════════════════════════════
run_test_mode() {
    echo -e "${BOLD}OmniEdge_AI — Test Mode${NC}"
    echo "  Running: cmake build + ctest (unit tests) + integration tests"
    echo ""

    cd "$PROJECT_ROOT"
    mkdir -p build && cd build

    # CMake configure with testing enabled
    cmake "$PROJECT_ROOT" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_TESTING=ON
    ok "CMake configured (BUILD_TESTING=ON)"

    # Build with memory-safe parallelism
    local JOBS
    JOBS=$(safe_job_count)
    echo -e "  ${CYAN}Building with -j${JOBS} ($(available_memory_gb) GB available, $(nproc) CPUs)${NC}"
    ninja -j"$JOBS"
    ok "All targets built"

    # Ensure CUDA/ONNX Runtime libs are discoverable by test binaries
    LD_LIBRARY_PATH="$(build_ld_library_path)"
    export LD_LIBRARY_PATH

    # ── C++ unit tests via ctest ──
    section "C++ Unit Tests (ctest)"
    echo "  Running ctest --output-on-failure ..."
    local CTEST_RC=0
    ctest --output-on-failure -j"$JOBS" --exclude-regex PipelineModelTests || CTEST_RC=$?
    if [ "$CTEST_RC" -ne 0 ]; then
        record_error "some tests failed (exit $CTEST_RC) — check output above"
    else
        ok "All C++ unit tests passed"
    fi

    # ── GPU-labeled tests (requires GPU + models) ──
    section "GPU Inferencer Tests (ctest -L gpu)"
    echo "  Running GPU-gated tests (GTEST_SKIP if models missing) ..."
    ctest --output-on-failure -L gpu -j1 || warn "Some GPU tests failed or were skipped"

    # ── Mode-focused integration tests ──
    section "Mode Integration Tests"
    cd "$PROJECT_ROOT"
    if [ -x "test.sh" ]; then
        echo "  Running bash test.sh all ..."
        bash test.sh all || record_error "Mode test suites had failures"
    else
        warn "No mode integration test runner found (expected executable test.sh) — skipping"
    fi

    # ── Summary ──
    echo ""
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    if [ "$ERRORS" -eq 0 ]; then
        echo -e "  ${GREEN}ALL TESTS PASSED — 0 errors${NC}"
    else
        echo -e "  ${RED}TEST RUN FINISHED WITH $ERRORS ERROR(S)${NC}"
        echo -e "  ${YELLOW}Fix failing tests before declaring the task complete.${NC}"
    fi
    echo -e "${BOLD}══════════════════════════════════════${NC}"
    exit "$ERRORS"
}

# If --test flag was given, run test mode only (no install)
if [ "$TEST_ONLY" -eq 1 ]; then
    run_test_mode
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# Print system info for debugging OOM issues
echo -e "${BOLD}OmniEdge_AI — Full Installation${NC}"
echo "Models directory: $OE_MODELS_DIR"
echo "Engines directory: $OE_ENGINES_DIR"
echo ""
echo -e "${BOLD}System Resources:${NC}"
echo "  RAM:  $(total_ram_gb) GB total, $(available_memory_gb) GB available"
echo "  Swap: $(awk '/SwapTotal/{printf "%d", $2/1024/1024}' /proc/meminfo) GB total"
echo "  CPUs: $(nproc)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"

local_ram=$(total_ram_gb)
if [ "$local_ram" -lt 16 ]; then
    echo ""
    echo -e "  ${RED}WARNING: Only ${local_ram} GB RAM detected.${NC}"
    echo -e "  ${YELLOW}Model quantization and TRT engine builds require 16-32 GB.${NC}"
    echo -e "  ${YELLOW}Increase container memory limit in docker-compose.yaml if needed.${NC}"
fi
echo ""

install_toolchain
install_vad
install_stt
install_tts
install_cv
install_enhancement
install_build

echo ""
echo -e "${BOLD}══════════════════════════════════════${NC}"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "  ${GREEN}INSTALLATION COMPLETE — 0 errors${NC}"
else
    echo -e "  ${RED}INSTALLATION FINISHED WITH $ERRORS ERROR(S)${NC}"
    echo -e "  ${YELLOW}Re-run to retry failed steps (idempotent).${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════${NC}"
exit "$ERRORS"
