#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// OmniEdge_AI — CUDA Error Checking Macro (standalone header)
//
// This file is intentionally kept separate from all other headers.
// Include it in any .cpp or .cu file that calls CUDA runtime APIs.
//
// Usage:
//   OE_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream));
//   OE_CUDA_CHECK(cudaStreamSynchronize(stream));
//
// On error: logs the failing call, file, and line number, then aborts.
// ---------------------------------------------------------------------------

#define OE_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        const cudaError_t oe_cuda_err_ = (call);                              \
        if (oe_cuda_err_ != cudaSuccess) {                                    \
            std::fprintf(stderr,                                              \
                "[OE_CUDA_CHECK] %s:%d — %s returned %s (%s)\n",             \
                __FILE__, __LINE__, #call,                                     \
                cudaGetErrorName(oe_cuda_err_),                               \
                cudaGetErrorString(oe_cuda_err_));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
