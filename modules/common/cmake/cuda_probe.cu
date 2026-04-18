// Minimal CUDA runtime probe — exits 0 only if cudaGetDeviceCount succeeds.
// Used by CMakeLists.txt try_run() to detect CUDA toolkit / driver mismatch.
#include <cuda_runtime.h>
int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 0 : 1;
}
