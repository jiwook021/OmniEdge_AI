// test_gpu_compositor.cu — GPU pixel-level tests for ISP pipeline
//
// Tests for the CUDA ISP kernels in gpu_compositor.cu:
//   - applyIspBcs(): brightness, contrast, saturation
//   - applyIspSharpen(): 3×3 unsharp mask edge amplification
//   - composite(): foreground/background mask selection
//   - upscaleMask(): bilinear upscale preserves spatial proportions
//
// All tests create synthetic images on CPU, upload to GPU, run the kernel,
// download results, and verify pixel values against hand-computed expectations.
//
// Requires: CUDA device, OpenCV with CUDA support.
// Skipped automatically when no CUDA device is present.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "gpu/gpu_compositor.hpp"


// ---------------------------------------------------------------------------
// Fixture — skips all tests if no CUDA device is available
// ---------------------------------------------------------------------------

class GpuCompositorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available — skipping GPU compositor tests.";
        }
        cudaSetDevice(0);

        // Verify OpenCV was compiled with CUDA support.
        // GpuMat::create() throws if OpenCV lacks CUDA.
        try {
            cv::cuda::GpuMat probe(1, 1, CV_8UC1);
            (void)probe;
        } catch (const cv::Exception&) {
            GTEST_SKIP() << "OpenCV compiled without CUDA support — skipping GPU compositor tests.";
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// applyIspBcs() — Brightness
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, BrightnessPlusShiftsAllPixels)
{
    // Solid gray (128,128,128). Apply brightness +50, contrast 1.0, saturation 1.0.
    // Expected: pixel = 128*1.0 + 50 = 178.  Saturation is neutral → unchanged.
    constexpr int W = 64, H = 64;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);

    applyIspBcs(gpu, /*brightness=*/50.0f, /*contrast=*/1.0f, /*saturation=*/1.0f,
                /*stream=*/nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    // Check every pixel
    int wrongCount = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            if (px[0] != 178 || px[1] != 178 || px[2] != 178) {
                ++wrongCount;
            }
        }
    }
    EXPECT_EQ(wrongCount, 0) << "All pixels should be (178,178,178) after brightness +50";
}

TEST_F(GpuCompositorTest, BrightnessNegativeShiftsDown)
{
    // Solid (100,100,100). Apply brightness -60 → 100 - 60 = 40.
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(100, 100, 100));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, -60.0f, 1.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    auto px = result.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 40);
    EXPECT_EQ(px[1], 40);
    EXPECT_EQ(px[2], 40);
}

TEST_F(GpuCompositorTest, BrightnessClampToZero)
{
    // Pixel (30,30,30), brightness -100 → -70 → clamp to 0.
    constexpr int W = 16, H = 16;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, -100.0f, 1.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);
    auto px = result.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 0);
    EXPECT_EQ(px[1], 0);
    EXPECT_EQ(px[2], 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// applyIspBcs() — Contrast
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, ContrastDoublesPixelSpread)
{
    // Gradient image: row i has pixel value (50 + i, 50 + i, 50 + i) for i in [0,63].
    // With contrast=2.0, brightness=0, saturation=1.0:
    //   pixel(row 0)  = 50*2 = 100
    //   pixel(row 63) = 113*2 = 226
    // The spread doubles from 63 to ~126 (or up to 255 clamp).
    constexpr int W = 32, H = 64;
    cv::Mat cpu(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        uint8_t v = static_cast<uint8_t>(50 + y);
        cpu.row(y).setTo(cv::Scalar(v, v, v));
    }

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 2.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    // Row 0: 50*2 = 100
    auto pxTop = result.at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(pxTop[0], 100, 1);

    // Row 63: 113*2 = 226
    auto pxBot = result.at<cv::Vec3b>(63, 0);
    EXPECT_NEAR(pxBot[0], 226, 1);

    // Verify spread increased: range should be > original 63
    int rangeAfter = static_cast<int>(pxBot[0]) - static_cast<int>(pxTop[0]);
    EXPECT_GT(rangeAfter, 63) << "Contrast 2.0 should widen the pixel value range";
}

TEST_F(GpuCompositorTest, ContrastClampAt255)
{
    // Pixel (200,200,200), contrast 2.0 → 400 → clamp to 255.
    constexpr int W = 8, H = 8;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(200, 200, 200));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 2.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);
    auto px = result.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 255);
    EXPECT_EQ(px[1], 255);
    EXPECT_EQ(px[2], 255);
}

// ═══════════════════════════════════════════════════════════════════════════════
// applyIspBcs() — Saturation
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, SaturationZeroProducesGrayscale)
{
    // Colored pixel (200, 100, 50) in BGR. Saturation=0 → grayscale.
    // Kernel luma: 0.114*B + 0.587*G + 0.299*R = 0.114*200 + 0.587*100 + 0.299*50
    //            = 22.8 + 58.7 + 14.95 = 96.45 → 96
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(200, 100, 50));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            // All channels converge to luma ≈ 96
            EXPECT_NEAR(px[0], 96, 1) << "B channel should equal luma";
            EXPECT_NEAR(px[1], 96, 1) << "G channel should equal luma";
            EXPECT_NEAR(px[2], 96, 1) << "R channel should equal luma";
            // All three channels must be equal (grayscale)
            EXPECT_EQ(px[0], px[1]);
            EXPECT_EQ(px[1], px[2]);
        }
    }
}

TEST_F(GpuCompositorTest, SaturationTwoAmplifiesColorDistance)
{
    // Pixel (200, 100, 50) BGR. Saturation=2.0.
    // luma = 96.45
    // b = 96.45 + 2*(200-96.45) = 303.55 → clamp 255
    // g = 96.45 + 2*(100-96.45) = 103.55 → 104
    // r = 96.45 + 2*(50-96.45) = 3.55 → 4 (or 3 due to rounding)
    constexpr int W = 16, H = 16;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(200, 100, 50));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 1.0f, 2.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    auto px = result.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 255) << "B channel should clamp to 255 at saturation 2.0";
    EXPECT_NEAR(px[1], 104, 1) << "G channel at saturation 2.0";
    EXPECT_NEAR(px[2], 4, 1)   << "R channel should approach 0 at saturation 2.0";
}

TEST_F(GpuCompositorTest, SaturationOneIsIdentity)
{
    // Saturation=1.0 should leave image unchanged.
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(180, 90, 45));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 1.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            EXPECT_EQ(px[0], 180);
            EXPECT_EQ(px[1], 90);
            EXPECT_EQ(px[2], 45);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// applyIspBcs() — Combined brightness + contrast + saturation
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, BcsFullPipelineCombinedEffects)
{
    // Pixel (128, 64, 32) BGR. brightness=10, contrast=1.5, saturation=0.5.
    // Step 1: pixel * contrast + brightness
    //   B = 128*1.5 + 10 = 202
    //   G = 64*1.5 + 10  = 106
    //   R = 32*1.5 + 10  = 58
    // Step 2: luma = 0.114*202 + 0.587*106 + 0.299*58
    //              = 23.028 + 62.222 + 17.342 = 102.592
    // Step 3: interpolate toward luma with saturation=0.5
    //   B = 102.592 + 0.5*(202 - 102.592) = 102.592 + 49.704 = 152.296 → 152
    //   G = 102.592 + 0.5*(106 - 102.592) = 102.592 + 1.704  = 104.296 → 104
    //   R = 102.592 + 0.5*(58  - 102.592) = 102.592 - 22.296 = 80.296  → 80
    constexpr int W = 8, H = 8;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(128, 64, 32));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 10.0f, 1.5f, 0.5f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    auto px = result.at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 152, 1) << "B channel combined BCS";
    EXPECT_NEAR(px[1], 104, 1) << "G channel combined BCS";
    EXPECT_NEAR(px[2], 80,  1) << "R channel combined BCS";
}

// ═══════════════════════════════════════════════════════════════════════════════
// applyIspSharpen() — Edge amplification
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, SharpenAmplifiesEdgePixels)
{
    // Image with sharp vertical edge at x=50 in a 100×10 image.
    // Left half = (50,50,50), right half = (200,200,200).
    // Sharpen amount=5.0.
    //
    // At edge pixel x=49 (interior, y>0 and y<9):
    //   center = 50. In the 3×3 neighborhood:
    //   - 6 pixels at 50 + 3 pixels at 200 → avg = (300+600)/9 = 100
    //   - dst = 50 + 5*(50-100) = -200 → clamp to 0
    //
    // At edge pixel x=50:
    //   center = 200. In the 3×3 neighborhood:
    //   - 3 pixels at 50 + 6 pixels at 200 → avg = (150+1200)/9 = 150
    //   - dst = 200 + 5*(200-150) = 450 → clamp to 255
    //
    // Flat area x=25:
    //   center = 50, all 9 neighbors = 50, avg = 50
    //   dst = 50 + 5*(50-50) = 50 → unchanged

    constexpr int W = 100, H = 10;
    cv::Mat cpu(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = (x < 50) ? 50 : 200;
            cpu.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
        }
    }

    cv::cuda::GpuMat gpuSrc, gpuDst;
    gpuSrc.upload(cpu);

    applyIspSharpen(gpuSrc, gpuDst, 5.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuDst.download(result);

    // Check flat area is unchanged (interior pixel far from edge)
    auto flatPx = result.at<cv::Vec3b>(5, 25);
    EXPECT_EQ(flatPx[0], 50) << "Flat area pixel should be unchanged after sharpen";

    // Check edge pixel x=49 (interior row, not border) → should be 0
    auto edgeLeft = result.at<cv::Vec3b>(5, 49);
    EXPECT_EQ(edgeLeft[0], 0) << "Edge pixel (dark side) should clamp to 0 with strong sharpen";

    // Check edge pixel x=50 → should be 255
    auto edgeRight = result.at<cv::Vec3b>(5, 50);
    EXPECT_EQ(edgeRight[0], 255) << "Edge pixel (bright side) should clamp to 255 with strong sharpen";
}

TEST_F(GpuCompositorTest, SharpenZeroIsIdentity)
{
    // Sharpen amount=0.0 should be identity (dst = src + 0*(src-avg) = src).
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3, cv::Scalar(120, 80, 40));

    cv::cuda::GpuMat gpuSrc, gpuDst;
    gpuSrc.upload(cpu);

    applyIspSharpen(gpuSrc, gpuDst, 0.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuDst.download(result);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            EXPECT_EQ(px[0], 120);
            EXPECT_EQ(px[1], 80);
            EXPECT_EQ(px[2], 40);
        }
    }
}

TEST_F(GpuCompositorTest, SharpenPreservesBorderPixels)
{
    // Border pixels (row 0, last row, col 0, last col) should be copied unchanged.
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3);
    // Fill with random-ish pattern
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = static_cast<uint8_t>((y * 13 + x * 7) % 256);
            cpu.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
        }
    }

    cv::cuda::GpuMat gpuSrc, gpuDst;
    gpuSrc.upload(cpu);
    applyIspSharpen(gpuSrc, gpuDst, 3.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuDst.download(result);

    // Top row
    for (int x = 0; x < W; ++x) {
        EXPECT_EQ(result.at<cv::Vec3b>(0, x), cpu.at<cv::Vec3b>(0, x))
            << "Border pixel (0," << x << ") should be unchanged";
    }
    // Bottom row
    for (int x = 0; x < W; ++x) {
        EXPECT_EQ(result.at<cv::Vec3b>(H - 1, x), cpu.at<cv::Vec3b>(H - 1, x))
            << "Border pixel (" << H-1 << "," << x << ") should be unchanged";
    }
    // Left column
    for (int y = 0; y < H; ++y) {
        EXPECT_EQ(result.at<cv::Vec3b>(y, 0), cpu.at<cv::Vec3b>(y, 0))
            << "Border pixel (" << y << ",0) should be unchanged";
    }
    // Right column
    for (int y = 0; y < H; ++y) {
        EXPECT_EQ(result.at<cv::Vec3b>(y, W - 1), cpu.at<cv::Vec3b>(y, W - 1))
            << "Border pixel (" << y << "," << W-1 << ") should be unchanged";
    }
}

TEST_F(GpuCompositorTest, SharpenDetectsPixelDeltaNearEdges)
{
    // Build a image with a horizontal edge at y=16 in a 32×32 image.
    // Top half = (40,40,40), bottom half = (210,210,210).
    // Count total pixel changes after sharpen.
    constexpr int W = 32, H = 32;
    cv::Mat cpu(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        uint8_t v = (y < 16) ? 40 : 210;
        cpu.row(y).setTo(cv::Scalar(v, v, v));
    }

    cv::cuda::GpuMat gpuSrc, gpuDst;
    gpuSrc.upload(cpu);
    applyIspSharpen(gpuSrc, gpuDst, 2.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuDst.download(result);

    // Count pixels that changed
    int changedPixels = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (result.at<cv::Vec3b>(y, x) != cpu.at<cv::Vec3b>(y, x)) {
                ++changedPixels;
            }
        }
    }

    // Only pixels adjacent to the edge (rows 15-16, excluding borders) should change.
    // Interior pixels in rows 15 and 16, columns 1..30 = 2 rows × 30 cols = 60 pixels.
    EXPECT_GT(changedPixels, 0) << "Sharpening should change pixels near edges";
    EXPECT_LE(changedPixels, 4 * W) << "Only pixels near the edge should change";
}

// ═══════════════════════════════════════════════════════════════════════════════
// composite() — Foreground/background mask selection
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, CompositeBinaryMaskSelectsCorrectSource)
{
    // Original = (100,100,100), Blurred = (200,200,200).
    // Mask: left half = 1.0 (foreground), right half = 0.0 (background).
    // Output: left = original, right = blurred.
    constexpr int W = 64, H = 32;
    cv::Mat cpuOrig(H, W, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat cpuBlur(H, W, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat cpuMask(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cpuMask.at<float>(y, x) = (x < W / 2) ? 1.0f : 0.0f;
        }
    }

    cv::cuda::GpuMat gpuOrig, gpuBlur, gpuMask, gpuOut;
    gpuOrig.upload(cpuOrig);
    gpuBlur.upload(cpuBlur);
    gpuMask.upload(cpuMask);

    composite(gpuOrig, gpuBlur, gpuMask, gpuOut, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuOut.download(result);

    // Left half (foreground, mask > 0.5) → original (100,100,100)
    auto pxLeft = result.at<cv::Vec3b>(16, 10);
    EXPECT_EQ(pxLeft[0], 100);
    EXPECT_EQ(pxLeft[1], 100);
    EXPECT_EQ(pxLeft[2], 100);

    // Right half (background, mask ≤ 0.5) → blurred (200,200,200)
    auto pxRight = result.at<cv::Vec3b>(16, 50);
    EXPECT_EQ(pxRight[0], 200);
    EXPECT_EQ(pxRight[1], 200);
    EXPECT_EQ(pxRight[2], 200);
}

TEST_F(GpuCompositorTest, CompositeAllForeground)
{
    // Mask all 1.0 → output == original everywhere.
    constexpr int W = 16, H = 16;
    cv::Mat cpuOrig(H, W, CV_8UC3, cv::Scalar(50, 100, 150));
    cv::Mat cpuBlur(H, W, CV_8UC3, cv::Scalar(250, 250, 250));
    cv::Mat cpuMask(H, W, CV_32FC1, cv::Scalar(1.0f));

    cv::cuda::GpuMat gpuOrig, gpuBlur, gpuMask, gpuOut;
    gpuOrig.upload(cpuOrig);
    gpuBlur.upload(cpuBlur);
    gpuMask.upload(cpuMask);

    composite(gpuOrig, gpuBlur, gpuMask, gpuOut, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuOut.download(result);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            EXPECT_EQ(result.at<cv::Vec3b>(y, x), cpuOrig.at<cv::Vec3b>(y, x));
        }
    }
}

TEST_F(GpuCompositorTest, CompositeAllBackground)
{
    // Mask all 0.0 → output == blurred everywhere.
    constexpr int W = 16, H = 16;
    cv::Mat cpuOrig(H, W, CV_8UC3, cv::Scalar(50, 100, 150));
    cv::Mat cpuBlur(H, W, CV_8UC3, cv::Scalar(250, 250, 250));
    cv::Mat cpuMask(H, W, CV_32FC1, cv::Scalar(0.0f));

    cv::cuda::GpuMat gpuOrig, gpuBlur, gpuMask, gpuOut;
    gpuOrig.upload(cpuOrig);
    gpuBlur.upload(cpuBlur);
    gpuMask.upload(cpuMask);

    composite(gpuOrig, gpuBlur, gpuMask, gpuOut, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpuOut.download(result);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            EXPECT_EQ(result.at<cv::Vec3b>(y, x), cpuBlur.at<cv::Vec3b>(y, x));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// upscaleMask() — Bilinear upscale spatial proportions
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, UpscaleMaskPreservesSpatialProportions)
{
    // 4×4 mask: top 2 rows = 1.0, bottom 2 rows = 0.0.
    // Upscale to 16×16.
    // Top 8 rows should be mostly ≈ 1.0 (except bilinear transition zone).
    // Bottom 8 rows should be mostly ≈ 0.0.
    constexpr int srcW = 4, srcH = 4;
    constexpr int dstW = 16, dstH = 16;

    std::vector<float> srcHost(srcW * srcH);
    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            srcHost[y * srcW + x] = (y < 2) ? 1.0f : 0.0f;
        }
    }

    float *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_src, srcW * srcH * sizeof(float));
    cudaMalloc(&d_dst, dstW * dstH * sizeof(float));
    cudaMemcpy(d_src, srcHost.data(), srcW * srcH * sizeof(float),
               cudaMemcpyHostToDevice);

    upscaleMask(d_src, srcW, srcH, d_dst, dstW, dstH, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> dstHost(dstW * dstH);
    cudaMemcpy(dstHost.data(), d_dst, dstW * dstH * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Top quarter (rows 0-3) should be solidly ≈ 1.0
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < dstW; ++x) {
            EXPECT_NEAR(dstHost[y * dstW + x], 1.0f, 0.05f)
                << "Top region should be ~1.0 at (" << y << "," << x << ")";
        }
    }

    // Bottom quarter (rows 12-15) should be solidly ≈ 0.0
    for (int y = 12; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            EXPECT_NEAR(dstHost[y * dstW + x], 0.0f, 0.05f)
                << "Bottom region should be ~0.0 at (" << y << "," << x << ")";
        }
    }

    // Count pixels > 0.5 — should be roughly half (top half of the image)
    int aboveHalf = 0;
    for (float v : dstHost) {
        if (v > 0.5f) { ++aboveHalf; }
    }
    int totalPixels = dstW * dstH;
    EXPECT_GT(aboveHalf, totalPixels / 4) << "At least 25% should be foreground";
    EXPECT_LT(aboveHalf, 3 * totalPixels / 4) << "At most 75% should be foreground";

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_F(GpuCompositorTest, UpscaleAllOnesStaysOnes)
{
    constexpr int srcW = 4, srcH = 4;
    constexpr int dstW = 32, dstH = 32;

    std::vector<float> srcHost(srcW * srcH, 1.0f);

    float *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_src, srcW * srcH * sizeof(float));
    cudaMalloc(&d_dst, dstW * dstH * sizeof(float));
    cudaMemcpy(d_src, srcHost.data(), srcW * srcH * sizeof(float),
               cudaMemcpyHostToDevice);

    upscaleMask(d_src, srcW, srcH, d_dst, dstW, dstH, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> dstHost(dstW * dstH);
    cudaMemcpy(dstHost.data(), d_dst, dstW * dstH * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < dstW * dstH; ++i) {
        EXPECT_NEAR(dstHost[i], 1.0f, 0.01f);
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pixel-change counting: total changed pixels for saturation color shift
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(GpuCompositorTest, SaturationChangeCountMatchesExpectation)
{
    // Multi-color image: each quadrant has a different color.
    // After saturation=0 (grayscale), every non-gray pixel should change.
    constexpr int W = 64, H = 64;
    cv::Mat cpu(H, W, CV_8UC3);

    // TL = red, TR = green, BL = blue, BR = yellow
    cv::Mat tl(cpu, cv::Rect(0,  0,  32, 32)); tl.setTo(cv::Scalar(0,   0,   255));
    cv::Mat tr(cpu, cv::Rect(32, 0,  32, 32)); tr.setTo(cv::Scalar(0,   255, 0));
    cv::Mat bl(cpu, cv::Rect(0,  32, 32, 32)); bl.setTo(cv::Scalar(255, 0,   0));
    cv::Mat br(cpu, cv::Rect(32, 32, 32, 32)); br.setTo(cv::Scalar(0,   255, 255));

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 1.0f, 0.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    // Every pixel should have changed (all original pixels had unequal channels)
    int changedPixels = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (result.at<cv::Vec3b>(y, x) != cpu.at<cv::Vec3b>(y, x)) {
                ++changedPixels;
            }
        }
    }
    int totalPixels = W * H;
    EXPECT_EQ(changedPixels, totalPixels)
        << "Every colored pixel should change when saturation goes to 0";

    // After saturation=0, all pixels MUST be grayscale (R==G==B)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto px = result.at<cv::Vec3b>(y, x);
            EXPECT_EQ(px[0], px[1]) << "Pixel (" << y << "," << x << ") B!=G after desaturation";
            EXPECT_EQ(px[1], px[2]) << "Pixel (" << y << "," << x << ") G!=R after desaturation";
        }
    }
}

TEST_F(GpuCompositorTest, ContrastChangeCountForEdgeDetection)
{
    // Create gradient image, apply high contrast, count pixels that changed.
    // With contrast=3.0, most pixels should change (saturate toward 0 or 255).
    constexpr int W = 64, H = 64;
    cv::Mat cpu(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t v = static_cast<uint8_t>((x * 4) % 256);
            cpu.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
        }
    }

    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    applyIspBcs(gpu, 0.0f, 3.0f, 1.0f, nullptr);
    cudaDeviceSynchronize();

    cv::Mat result;
    gpu.download(result);

    int changedPixels = 0;
    int edgePixels = 0;  // pixels that hit 0 or 255
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            auto orig = cpu.at<cv::Vec3b>(y, x);
            auto out  = result.at<cv::Vec3b>(y, x);
            if (out != orig) ++changedPixels;
            if (out[0] == 0 || out[0] == 255) ++edgePixels;
        }
    }

    int total = W * H;
    // Most pixels should change with contrast=3.0 (all except pixel value 0)
    EXPECT_GT(changedPixels, total * 3 / 4)
        << "High contrast should change most pixels";

    // Many pixels should saturate to 0 or 255
    EXPECT_GT(edgePixels, total / 4)
        << "High contrast should push many pixels to extremes (0 or 255)";
}

