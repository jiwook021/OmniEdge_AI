#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — GPU Face Recognition Preprocessing
//
// Fused CUDA kernels for zero-copy face recognition preprocessing.
// These replace CPU-side OpenCV operations (cv::resize, cv::dnn::blobFromImage)
// with single-pass GPU kernels that write directly into ONNX IO-bound buffers.
//
// Usage:
//   1. Upload BGR frame to GPU via PinnedStagingBuffer + cudaMemcpyAsync
//   2. Call launchLetterboxResizeNorm() to fill SCRFD input buffer
//   3. Run SCRFD inference via IO Binding
//   4. For each detected face, call launchAlignCropNorm() to fill AuraFace input
//   5. Run AuraFace inference via IO Binding
// ---------------------------------------------------------------------------

#include <cstdint>

#include <cuda_runtime.h>


namespace oe::cv {

/// Launch fused letterbox resize + BGR→RGB + normalize kernel.
///
/// @param d_bgrSrc    Device pointer to BGR24 source image (srcW × srcH × 3)
/// @param srcW        Source image width
/// @param srcH        Source image height
/// @param d_nchwDst   Device pointer to output [1, 3, dstSize, dstSize] float NCHW
/// @param dstSize     Square output dimension (e.g. 640 for SCRFD)
/// @param scale       Letterbox scale factor (min(dstSize/srcW, dstSize/srcH))
/// @param padX        Horizontal padding offset in pixels
/// @param padY        Vertical padding offset in pixels
/// @param mean        Normalization mean (e.g. 127.5 for SCRFD)
/// @param std         Normalization std (e.g. 128.0 for SCRFD)
/// @param stream      CUDA stream
void launchLetterboxResizeNorm(
    const uint8_t* d_bgrSrc,
    int            srcW,
    int            srcH,
    float*         d_nchwDst,
    int            dstSize,
    float          scale,
    float          padX,
    float          padY,
    float          mean,
    float          std,
    cudaStream_t   stream);

/// Launch fused affine alignment + BGR→RGB + normalize kernel.
///
/// @param d_bgrSrc      Device pointer to BGR24 source image (srcW × srcH × 3)
/// @param srcW          Source image width
/// @param srcH          Source image height
/// @param d_nchwDst     Device pointer to output [1, 3, dstSize, dstSize] float NCHW
/// @param dstSize       Square output dimension (e.g. 112 for AuraFace)
/// @param d_invAffine   Device pointer to inverse affine matrix [2][3] float row-major
/// @param mean          Normalization mean (e.g. 0.5 for AuraFace)
/// @param std           Normalization std (e.g. 0.5 for AuraFace)
/// @param stream        CUDA stream
void launchAlignCropNorm(
    const uint8_t* d_bgrSrc,
    int            srcW,
    int            srcH,
    float*         d_nchwDst,
    int            dstSize,
    const float*   d_invAffine,
    float          mean,
    float          std,
    cudaStream_t   stream);

} // namespace oe::cv
