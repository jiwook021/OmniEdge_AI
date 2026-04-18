#include <gtest/gtest.h>

#include "cv/onnx_face_recog_inferencer.hpp"
#include "cv/face_recog_node.hpp"
#include "cv_test_helpers.hpp"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>

// ---------------------------------------------------------------------------
// ONNX Face Recognition Inferencer Integration Tests
//
// Purpose: Verify the OnnxFaceRecogInferencer (SCRFD + AuraFace v1) produces
// correct detections with real ONNX models. These tests require:
//   1. GPU with ONNX Runtime + CUDA EP
//   2. Model files in $OE_MODELS_DIR/face_models/scrfd_auraface/
//
// Tests GTEST_SKIP() if models are not found.
//
// What these tests catch:
//   - ONNX model loading failures (wrong paths, incompatible models)
//   - SCRFD post-processing bugs (NMS, anchor decoding, coordinate rescaling)
//   - Face alignment errors (affine warp, landmark transform)
//   - Embedding dimension mismatches (expected 512-d)
//   - L2-normalization errors (embeddings must have unit norm)
//   - Landmark population (SCRFD provides real landmarks, not zero-init)
//   - VRAM budget reporting per variant
// ---------------------------------------------------------------------------


namespace {

/// Resolve the model directory for a given variant.
std::filesystem::path modelDir(const std::string& variant)
{
    const char* modelsRoot = std::getenv("OE_MODELS_DIR");
    std::filesystem::path base = modelsRoot
        ? std::filesystem::path(modelsRoot)
        : std::filesystem::path(std::getenv("HOME")) / "omniedge_models";
    return base / "face_models" / variant;
}

/// Check if both detector.onnx and recognizer.onnx exist for a variant.
bool modelsAvailable(const std::string& variant)
{
    const auto dir = modelDir(variant);
    return std::filesystem::exists(dir / "detector.onnx")
        && std::filesystem::exists(dir / "recognizer.onnx");
}

/// Compute L2 norm of a vector.
float l2Norm(const std::vector<float>& v)
{
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    return std::sqrt(sum);
}

/// Load model, returning false on CUDA/TensorRT EP registration failure.
/// The EP may be unavailable when cuBLAS version doesn't match ONNX Runtime's
/// build-time CUDA version (e.g., CUDA 13 container with ONNX RT built for CUDA 12).
bool tryLoadModel(OnnxFaceRecogInferencer& inferencer, const std::string& modelDir)
{
    try {
        inferencer.loadModel(modelDir);
        return true;
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("EP registration failed") != std::string::npos) {
            return false;
        }
        throw;
    }
}

/// Macro: load model or GTEST_SKIP + return from the calling TEST function.
#define LOAD_MODEL_OR_SKIP(inferencer, dir) \
    do { \
        if (!tryLoadModel((inferencer), (dir))) { \
            GTEST_SKIP() << "CUDA/TensorRT EP unavailable in this container"; \
            return; \
        } \
    } while (0)

} // anonymous namespace

// ===========================================================================
// Model Loading
// ===========================================================================

TEST(OnnxFaceRecogInferencer, LoadModel_DefaultVariant_Succeeds)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found at "
                     << modelDir("scrfd_auraface");
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    SPDLOG_DEBUG("Model loaded successfully, VRAM reported: {} bytes",
                 inferencer.currentVramUsageBytes());

    EXPECT_GT(inferencer.currentVramUsageBytes(), 0u)
        << "VRAM usage must be reported after loading";
}

TEST(OnnxFaceRecogInferencer, LoadModel_MissingDirectory_Throws)
{
    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    EXPECT_THROW(
        inferencer.loadModel("/nonexistent/path/to/models"),
        std::runtime_error);
}

// ===========================================================================
// Face Detection + Embedding
// ===========================================================================

TEST(OnnxFaceRecogInferencer, DetectFace_SingleFace_ReturnsBboxLandmarksEmbedding)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found";
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    auto faceImage = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(faceImage);

    auto result = inferencer.detect(
        faceImage.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(result.has_value()) << "Detection must not fail: " << result.error();
    ASSERT_GE(result.value().size(), 1u)
        << "Expected at least one face in a single-face image";

    const auto& face = result.value().front();

    // Bounding box must be positive and within image bounds
    EXPECT_GT(face.bbox.w, 0) << "Bounding box width must be positive";
    EXPECT_GT(face.bbox.h, 0) << "Bounding box height must be positive";
    EXPECT_GE(face.bbox.x, 0) << "Bounding box x must be >= 0";
    EXPECT_GE(face.bbox.y, 0) << "Bounding box y must be >= 0";

    SPDLOG_DEBUG("Detected face: bbox=({},{},{},{})",
                 face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h);

    // Embedding must be 512-d
    EXPECT_EQ(face.embedding.size(), 512u)
        << "Face embedding must be 512-dimensional";

    // Embedding must be L2-normalized (unit norm)
    const float norm = l2Norm(face.embedding);
    EXPECT_NEAR(norm, 1.0f, 1e-3f)
        << "Embedding must be L2-normalized (unit norm)";

    SPDLOG_DEBUG("Embedding L2 norm: {:.6f}", norm);
}

TEST(OnnxFaceRecogInferencer, DetectFace_EmptyRoom_ReturnsEmpty)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found";
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    auto emptyRoom = loadBgr24Fixture(kEmptyRoomFile);
    SKIP_IF_NO_FIXTURE(emptyRoom);

    auto result = inferencer.detect(
        emptyRoom.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().empty())
        << "Empty room must return no face detections";
}

// ===========================================================================
// Landmark Quality
// ===========================================================================

TEST(OnnxFaceRecogInferencer, LandmarksArePopulated)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found";
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    auto faceImage = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(faceImage);

    auto result = inferencer.detect(
        faceImage.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(result.has_value());
    ASSERT_GE(result.value().size(), 1u);

    const auto& landmarks = result.value().front().landmarks;

    // All 5 landmark points must have non-zero coordinates
    // All points must be populated with real coordinates
    for (int p = 0; p < 5; ++p) {
        const bool isNonZero = (landmarks.pts[p][0] != 0.0f
                             || landmarks.pts[p][1] != 0.0f);
        EXPECT_TRUE(isNonZero)
            << "Landmark point " << p << " must have non-zero coordinates";

        SPDLOG_DEBUG("Landmark[{}]: ({:.1f}, {:.1f})",
                     p, landmarks.pts[p][0], landmarks.pts[p][1]);
    }
}

// ===========================================================================
// Embedding Similarity — same face vs different faces
// ===========================================================================

TEST(OnnxFaceRecogInferencer, SameFaceTwice_HighSimilarity)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found";
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    auto faceImage = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(faceImage);

    auto result1 = inferencer.detect(
        faceImage.data(), kFixtureImageWidth, kFixtureImageHeight);
    auto result2 = inferencer.detect(
        faceImage.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(result1.has_value() && result2.has_value());
    ASSERT_GE(result1.value().size(), 1u);
    ASSERT_GE(result2.value().size(), 1u);

    const float similarity = FaceRecognitionNode::cosineSimilarity(
        result1.value().front().embedding,
        result2.value().front().embedding);

    SPDLOG_DEBUG("Same face similarity: {:.6f}", similarity);

    EXPECT_GT(similarity, 0.9f)
        << "Same face detected twice must have cosine similarity > 0.9";
}

TEST(OnnxFaceRecogInferencer, DifferentFaces_LowSimilarity)
{
    if (!modelsAvailable("scrfd_auraface")) {
        GTEST_SKIP() << "SCRFD+AuraFace models not found";
    }

    OnnxFaceRecogInferencer inferencer(FaceRecogVariant::kScrfdAuraFaceV1);
    LOAD_MODEL_OR_SKIP(inferencer, modelDir("scrfd_auraface").string());

    auto aliceImage = loadBgr24Fixture(kFaceAliceFile);
    auto bobImage   = loadBgr24Fixture(kFaceBobFile);
    SKIP_IF_NO_FIXTURE(aliceImage);
    SKIP_IF_NO_FIXTURE(bobImage);

    auto aliceResult = inferencer.detect(
        aliceImage.data(), kFixtureImageWidth, kFixtureImageHeight);
    auto bobResult = inferencer.detect(
        bobImage.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(aliceResult.has_value() && bobResult.has_value());
    ASSERT_GE(aliceResult.value().size(), 1u);
    ASSERT_GE(bobResult.value().size(), 1u);

    const float similarity = FaceRecognitionNode::cosineSimilarity(
        aliceResult.value().front().embedding,
        bobResult.value().front().embedding);

    SPDLOG_DEBUG("Different faces similarity: {:.6f}", similarity);

    EXPECT_LT(similarity, 0.5f)
        << "Different faces must have cosine similarity < 0.5";
}

// ===========================================================================
// VRAM Reporting per Variant
// ===========================================================================

TEST(OnnxFaceRecogInferencer, VramUsage_ReportsAuraFaceBudget)
{
    // No models needed — test the unloaded and factory behavior
    {
        OnnxFaceRecogInferencer unloaded(FaceRecogVariant::kScrfdAuraFaceV1);
        EXPECT_EQ(unloaded.currentVramUsageBytes(), 0u)
            << "Unloaded inferencer must report 0 VRAM";
    }

    const auto bytes = faceRecogVariantVramBytes(FaceRecogVariant::kScrfdAuraFaceV1);
    EXPECT_GT(bytes, 0u) << "AuraFace budget must be non-zero";
    SPDLOG_DEBUG("VRAM budget: AuraFace={}MiB", bytes / (1024*1024));
}

// ===========================================================================
// Variant Parsing
// ===========================================================================

TEST(OnnxFaceRecogInferencer, ParseVariant_ValidName)
{
    EXPECT_EQ(parseFaceRecogVariant("scrfd_auraface_v1"),
              FaceRecogVariant::kScrfdAuraFaceV1);
}

TEST(OnnxFaceRecogInferencer, ParseVariant_UnknownDefaultsToAuraFace)
{
    EXPECT_EQ(parseFaceRecogVariant("unknown_model"),
              FaceRecogVariant::kScrfdAuraFaceV1);
    EXPECT_EQ(parseFaceRecogVariant(""),
              FaceRecogVariant::kScrfdAuraFaceV1);
}

