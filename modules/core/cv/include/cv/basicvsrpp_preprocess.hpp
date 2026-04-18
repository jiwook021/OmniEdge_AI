#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — BasicVSR++ GPU Preprocessing
//
// Fused CUDA kernel: bilinear resize + BGR-to-RGB + uint8-to-float32 normalize
// Replaces CPU cv::dnn::blobFromImage (~3 ms/frame) with GPU (~0.05 ms/frame).
// ---------------------------------------------------------------------------

#include <cstdint>

#include <cuda_runtime.h>


namespace oe::cv {

// ---------------------------------------------------------------------------
// launchResizeNormFP32 — preprocess one BGR24 frame into a [1,N,3,H,W] tensor
//
// Performs in a single fused kernel:
//   1. Bilinear resize from (srcW, srcH) to (dstW, dstH)
//   2. BGR-to-RGB channel swap
//   3. uint8 [0,255] to float32 [0,1] normalization
//   4. Write to NCHW layout at the correct frame offset
//
// @param d_bgrSrc     Device pointer to BGR24 source frame (srcW * srcH * 3 bytes)
// @param srcW, srcH   Source frame dimensions
// @param d_nchwDst    Device pointer to [1, N, 3, dstH, dstW] float32 output tensor
// @param dstW, dstH   Destination (model input) dimensions
// @param frameIndex   Frame index within the temporal window [0, totalFrames)
// @param totalFrames  Total frames in the temporal window (N dimension)
// @param stream       CUDA stream for async execution
// ---------------------------------------------------------------------------
void launchResizeNormFP32(
    const uint8_t* d_bgrSrc, int srcW, int srcH,
    float*         d_nchwDst, int dstW, int dstH,
    int            frameIndex, int totalFrames,
    cudaStream_t   stream);

// ---------------------------------------------------------------------------
// launchChwToHwcBgr — convert float32 NCHW RGB [0,1] to uint8 HWC BGR [0,255]
//
// Used for nvJPEG encoding which expects interleaved BGR/RGB uint8 format.
//
// @param d_nchwSrc    Device pointer to NCHW float32 RGB data for one frame
// @param d_bgrDst     Device pointer to HWC BGR uint8 output
// @param width        Frame width
// @param height       Frame height
// @param stream       CUDA stream
// ---------------------------------------------------------------------------
void launchChwToHwcBgr(
    const float* d_nchwSrc,
    uint8_t*     d_bgrDst,
    int          width, int height,
    cudaStream_t stream);

} // namespace oe::cv
