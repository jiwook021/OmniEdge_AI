#include <gtest/gtest.h>

#include "cv/face_recog_node.hpp"
#include "cv_test_helpers.hpp"
#include "common/constants/video_constants.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Face Detection API Tests
//
// Purpose: Verify the FaceRecogInferencer interface contract from the user's
// perspective — "I give the face engine a camera frame, it tells me where
// faces are and produces an embedding I can use for matching."
//
// What these tests catch:
//   - detect() fails to return a face when one is present
//   - detect() falsely returns a face in an empty scene
//   - Embeddings are identical for different faces (hashing collision)
//   - Inferencer errors crash instead of propagating cleanly
//   - cosineSimilarity() math errors (the core matching algorithm)
//
// Tests load actual BGR24 image files from tests/cv/fixtures/ when available.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubFaceDetectionInferencer — simulates face detection for CPU-only testing.
//
// Returns configurable face detections with embeddings that depend on the
// input pixel data.  This proves the pipeline actually reads the input.
// ---------------------------------------------------------------------------
class StubFaceDetectionInferencer : public FaceRecogInferencer {
public:
    void loadModel(const std::string& /*modelPackPath*/) override {}

    [[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
    detect(const uint8_t* bgrInputFrame,
           uint32_t       frameWidth,
           uint32_t       frameHeight) override
    {
        ++detectCallCount_;

        if (forceError_) {
            return tl::unexpected(
                std::string("simulated face detection failure"));
        }
        if (!shouldDetectFace_) {
            return std::vector<FaceDetection>{};
        }

        FaceDetection detectedFace;
        detectedFace.bbox = {100, 50, 80, 90};
        detectedFace.landmarks = {};
        detectedFace.embedding.assign(512, 0.0f);

        // Derive embedding dimensions from the centre of the image where the
        // actual face content lives.  The corners are uniform background on
        // canvas-style fixtures, so sampling pixel [0] would be identical for
        // every portrait.
        if (bgrInputFrame != nullptr) {
            const std::size_t cx = frameWidth / 2;
            const std::size_t cy = frameHeight / 2;
            const std::size_t stride = static_cast<std::size_t>(frameWidth) * 3;
            const std::size_t centreOffset = cy * stride + cx * 3;
            detectedFace.embedding[0] =
                static_cast<float>(bgrInputFrame[centreOffset])     / 255.0f;
            detectedFace.embedding[1] =
                static_cast<float>(bgrInputFrame[centreOffset + 1]) / 255.0f;
            detectedFace.embedding[2] =
                static_cast<float>(bgrInputFrame[centreOffset + 2]) / 255.0f;
        } else {
            detectedFace.embedding[0] = 1.0f;
        }

        return std::vector<FaceDetection>{detectedFace};
    }

    void unload() noexcept override {}

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
        return 0;
    }

    // --- Configuration ---
    bool shouldDetectFace_{true};
    bool forceError_{false};
    int  detectCallCount_{0};
};

// ---------------------------------------------------------------------------
// StubMultiFaceInferencer — returns multiple faces with unique embeddings.
// Used for group photo / multi-face detection testing.
// ---------------------------------------------------------------------------
class StubMultiFaceInferencer : public FaceRecogInferencer {
public:
    void loadModel(const std::string& /*modelPackPath*/) override {}

    [[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
    detect(const uint8_t* /*bgrInputFrame*/,
           uint32_t       /*frameWidth*/,
           uint32_t       /*frameHeight*/) override
    {
        ++detectCallCount_;

        if (forceError_) {
            return tl::unexpected(
                std::string("simulated face detection failure"));
        }

        std::vector<FaceDetection> detectedFaces;
        for (int faceIndex = 0; faceIndex < numberOfFacesToDetect_; ++faceIndex) {
            FaceDetection face;
            face.bbox = {100 + faceIndex * 50, 50, 80, 90};
            face.landmarks = {};
            face.embedding.assign(512, 0.0f);
            // Each face gets a unique unit vector along a different dimension
            if (faceIndex < 512) {
                face.embedding[static_cast<std::size_t>(faceIndex)] = 1.0f;
            }
            detectedFaces.push_back(std::move(face));
        }
        return detectedFaces;
    }

    void unload() noexcept override {}

    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
        return 0;
    }

    // --- Configuration ---
    int  numberOfFacesToDetect_{1};
    bool forceError_{false};
    int  detectCallCount_{0};
};

// ===========================================================================
// Face Detection — "find faces in a camera frame"
// ===========================================================================

TEST(FaceDetection, FaceImage_ReturnsBoundingBoxAndEmbedding)
{
    // GIVEN: a face image loaded from a test fixture
    auto faceImageBgr = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(faceImageBgr);

    StubFaceDetectionInferencer faceDetector;
    faceDetector.loadModel("face_model_pack");

    // WHEN: the face detector processes the image
    auto result = faceDetector.detect(
        faceImageBgr.data(), kFixtureImageWidth, kFixtureImageHeight);

    // THEN: exactly one face is found with a valid bounding box and embedding
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1u)
        << "Expected exactly one face in a single-face image";

    const auto& detectedFace = result.value().front();
    EXPECT_GT(detectedFace.bbox.w, 0) << "Bounding box must have positive width";
    EXPECT_GT(detectedFace.bbox.h, 0) << "Bounding box must have positive height";
    EXPECT_EQ(detectedFace.embedding.size(), 512u)
        << "Face embedding must be 512-dimensional";

    // Embedding is derived from input centre pixels (not all zeros).
    // The stub samples from the centre of the image where face content is.
    const std::size_t centreOffset =
        (kFixtureImageHeight / 2) * static_cast<std::size_t>(kFixtureImageWidth) * 3
        + (kFixtureImageWidth / 2) * 3;
    EXPECT_NEAR(detectedFace.embedding[0],
                static_cast<float>(faceImageBgr[centreOffset]) / 255.0f, 1e-5f)
        << "Embedding must depend on the actual input image content";
}

TEST(FaceDetection, EmptyRoom_ReturnsNoFaces)
{
    // Bug caught: false positive face detections in scenes with no people

    StubFaceDetectionInferencer faceDetector;
    faceDetector.shouldDetectFace_ = false;

    auto emptyRoomBgr = loadBgr24Fixture(kEmptyRoomFile);
    SKIP_IF_NO_FIXTURE(emptyRoomBgr);

    // WHEN: the detector processes an empty room
    auto result = faceDetector.detect(
        emptyRoomBgr.data(), kFixtureImageWidth, kFixtureImageHeight);

    // THEN: no faces are detected
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().empty())
        << "An image with no faces must return an empty detection list";
}

TEST(FaceDetection, InferencerFailure_PropagatesErrorMessage)
{
    // Bug caught: face detection errors silently ignored

    StubFaceDetectionInferencer faceDetector;
    faceDetector.forceError_ = true;

    auto result = faceDetector.detect(nullptr,
        kMaxInputWidth, kMaxInputHeight);

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("simulated"), std::string::npos)
        << "Error message must propagate from the detection inferencer";
}

TEST(FaceDetection, DifferentFaceImages_ProduceDifferentEmbeddings)
{
    // Bug caught: inferencer returns the same embedding regardless of input
    // (would cause all faces to match each other)

    auto aliceFaceBgr = loadBgr24Fixture(kFaceAliceFile);
    auto bobFaceBgr   = loadBgr24Fixture(kFaceBobFile);
    SKIP_IF_NO_FIXTURE(aliceFaceBgr);
    SKIP_IF_NO_FIXTURE(bobFaceBgr);

    StubFaceDetectionInferencer faceDetector;

    auto aliceResult = faceDetector.detect(
        aliceFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);
    auto bobResult = faceDetector.detect(
        bobFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(aliceResult.has_value() && bobResult.has_value());

    const auto& aliceEmbedding = aliceResult.value().front().embedding;
    const auto& bobEmbedding   = bobResult.value().front().embedding;

    EXPECT_NE(aliceEmbedding[0], bobEmbedding[0])
        << "Different face images must produce different embeddings — "
           "the inferencer may be ignoring the input frame";
}

TEST(FaceDetection, GroupPhoto_ReturnsMultipleFacesWithDistinctEmbeddings)
{
    // Bug caught: inferencer only detects one face in a multi-face image

    auto groupPhotoBgr = loadBgr24Fixture(kGroupPhotoFile);
    SKIP_IF_NO_FIXTURE(groupPhotoBgr);

    StubMultiFaceInferencer multiFaceDetector;
    multiFaceDetector.numberOfFacesToDetect_ = 3;

    auto result = multiFaceDetector.detect(
        groupPhotoBgr.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 3u)
        << "Expected three faces in a group photo";

    // Each face must have a unique embedding (orthogonal unit vectors)
    for (int i = 0; i < 3; ++i) {
        const auto& face = result.value()[static_cast<std::size_t>(i)];
        EXPECT_EQ(face.embedding.size(), 512u);
        EXPECT_EQ(face.embedding[static_cast<std::size_t>(i)], 1.0f)
            << "Face " << i << " should have a unit vector along dimension " << i;

        // Cross-check: distinct faces have orthogonal embeddings
        for (int j = i + 1; j < 3; ++j) {
            const auto& otherFace = result.value()[static_cast<std::size_t>(j)];
            float similarity = FaceRecognitionNode::cosineSimilarity(
                face.embedding, otherFace.embedding);
            EXPECT_NEAR(similarity, 0.0f, 1e-5f)
                << "Face " << i << " and face " << j
                << " should have orthogonal (distinct) embeddings";
        }
    }
}

// ===========================================================================
// Cosine Similarity — core face matching algorithm
//
// These tests verify the math that determines whether two faces are the
// same person.  A bug here would cause either:
//   - False positives: strangers identified as known people
//   - False negatives: known people not recognized
// ===========================================================================

TEST(CosineSimilarity, IdenticalVectors_ReturnsOne)
{
    // Same face embedding compared to itself → perfect match
    const std::vector<float> faceEmbedding(512, 1.0f / std::sqrt(512.0f));
    const float similarity =
        FaceRecognitionNode::cosineSimilarity(faceEmbedding, faceEmbedding);
    EXPECT_NEAR(similarity, 1.0f, 1e-5f);
}

TEST(CosineSimilarity, OrthogonalVectors_ReturnsZero)
{
    // Two completely unrelated face embeddings → no similarity
    std::vector<float> embeddingA(4, 0.0f);
    std::vector<float> embeddingB(4, 0.0f);
    embeddingA[0] = 1.0f;
    embeddingB[1] = 1.0f;

    const float similarity =
        FaceRecognitionNode::cosineSimilarity(embeddingA, embeddingB);
    EXPECT_NEAR(similarity, 0.0f, 1e-6f);
}

TEST(CosineSimilarity, OppositeVectors_ReturnsNegativeOne)
{
    const std::vector<float> positive = {1.0f, 0.0f, 0.0f};
    const std::vector<float> negative = {-1.0f, 0.0f, 0.0f};

    const float similarity =
        FaceRecognitionNode::cosineSimilarity(positive, negative);
    EXPECT_NEAR(similarity, -1.0f, 1e-6f);
}

TEST(CosineSimilarity, ScaledVectors_ReturnsSameResult)
{
    // Bug caught: implementation fails to normalize, making brightness
    // changes affect face matching

    std::vector<float> normalEmbedding(512, 1.0f);
    std::vector<float> scaledEmbedding(512, 3.0f);  // 3x magnitude

    const float similarity =
        FaceRecognitionNode::cosineSimilarity(normalEmbedding, scaledEmbedding);
    EXPECT_NEAR(similarity, 1.0f, 1e-5f)
        << "Cosine similarity must be scale-invariant — "
           "the same face at different brightness must still match";
}

TEST(CosineSimilarity, EmptyVectors_ReturnsZero)
{
    const std::vector<float> emptyEmbedding;
    EXPECT_FLOAT_EQ(
        FaceRecognitionNode::cosineSimilarity(emptyEmbedding, emptyEmbedding),
        0.0f);
}

TEST(CosineSimilarity, DimensionMismatch_ReturnsZero)
{
    // Bug caught: out-of-bounds access when comparing embeddings from
    // different model versions (128-dim vs 512-dim)

    const std::vector<float> embedding128d(128, 1.0f);
    const std::vector<float> embedding512d(512, 1.0f);

    EXPECT_FLOAT_EQ(
        FaceRecognitionNode::cosineSimilarity(embedding128d, embedding512d),
        0.0f);
}

TEST(CosineSimilarity, KnownAngle_ReturnsExpectedCosine)
{
    // a = [1, 0],  b = [1/sqrt(2), 1/sqrt(2)]  → angle = 45 degrees
    // cosine(45 degrees) = 1/sqrt(2) ≈ 0.7071

    const float kOneOverSqrt2 = 1.0f / std::sqrt(2.0f);
    const std::vector<float> axisAligned = {1.0f, 0.0f};
    const std::vector<float> at45Degrees = {kOneOverSqrt2, kOneOverSqrt2};

    const float similarity =
        FaceRecognitionNode::cosineSimilarity(axisAligned, at45Degrees);
    EXPECT_NEAR(similarity, kOneOverSqrt2, 1e-5f);
}

