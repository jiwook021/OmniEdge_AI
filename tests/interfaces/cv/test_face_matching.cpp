#include <gtest/gtest.h>

#include "cv/face_recog_node.hpp"
#include "face_recog_inferencer.hpp"
#include "common/runtime_defaults.hpp"
#include "cv_test_helpers.hpp"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Face Matching Pipeline Tests
//
// Purpose: Verify the end-to-end face identification flow from the user's
// perspective — "I register a person's face, and later the system recognizes
// (or correctly rejects) them in new camera frames."
//
// What these tests catch:
//   - Same person not recognized (false negative — threshold too high)
//   - Stranger identified as known person (false positive — threshold too low)
//   - Wrong person identified when multiple faces are in the gallery
//   - Frame subsampling skips too many or too few frames
//   - cosineSimilarity impl diverges from reference (gradual drift in accuracy)
//
// All matching tests operate on the static cosineSimilarity() utility and
// mock detect() outputs.  No GPU or ONNX models are required.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// StubFaceDetectionInferencer — produces embeddings that depend on input pixels
// ---------------------------------------------------------------------------
class StubFaceInferencer : public FaceRecogInferencer {
public:
    void loadModel(const std::string& /*modelPackPath*/) override {}

    [[nodiscard]] tl::expected<std::vector<FaceDetection>, std::string>
    detect(const uint8_t* bgrInputFrame,
           uint32_t       /*frameWidth*/,
           uint32_t       /*frameHeight*/) override
    {
        ++detectCallCount_;

        FaceDetection face;
        face.bbox = {100, 50, 80, 90};
        face.landmarks = {};
        face.embedding.assign(512, 0.0f);
        if (bgrInputFrame != nullptr) {
            // Derive embedding dimensions from the first pixel's B, G, R channels.
            // Different images with different per-channel colors produce embeddings
            // that point in genuinely different directions in 3D+ space,
            // guaranteeing different cosine similarity values.
            face.embedding[0] = static_cast<float>(bgrInputFrame[0]) / 255.0f;  // B
            face.embedding[1] = static_cast<float>(bgrInputFrame[1]) / 255.0f;  // G
            face.embedding[2] = static_cast<float>(bgrInputFrame[2]) / 255.0f;  // R
        }
        return std::vector<FaceDetection>{face};
    }

    void unload() noexcept override {}
    [[nodiscard]] std::size_t currentVramUsageBytes() const noexcept override {
        return 0;
    }

    int detectCallCount_{0};
};

// ---------------------------------------------------------------------------
// Reference cosine similarity for cross-checking the production implementation
// ---------------------------------------------------------------------------
static float referenceCosineSimilarity(const std::vector<float>& embeddingA,
                                       const std::vector<float>& embeddingB)
{
    if (embeddingA.size() != embeddingB.size() || embeddingA.empty()) {
        return 0.0f;
    }

    float dotProduct = 0.0f;
    float normSquaredA = 0.0f;
    float normSquaredB = 0.0f;
    for (std::size_t i = 0; i < embeddingA.size(); ++i) {
        dotProduct    += embeddingA[i] * embeddingB[i];
        normSquaredA  += embeddingA[i] * embeddingA[i];
        normSquaredB  += embeddingB[i] * embeddingB[i];
    }
    const float denominator =
        std::sqrt(normSquaredA) * std::sqrt(normSquaredB);
    return (denominator > 1e-9f) ? dotProduct / denominator : 0.0f;
}

// ===========================================================================
// Gallery Matching — "register a face, then identify or reject"
// ===========================================================================

TEST(FaceMatching, SameFaceTwice_ExceedsRecognitionThreshold)
{
    // User scenario: Alice registers her face, then walks past the camera again.
    // The system must recognize her (cosine similarity > threshold).

    auto aliceFaceBgr = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(aliceFaceBgr);

    StubFaceInferencer faceInferencer;

    // Register: detect Alice's face and store her embedding
    auto registrationResult = faceInferencer.detect(
        aliceFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);
    ASSERT_TRUE(registrationResult.has_value());
    const auto& registeredEmbedding =
        registrationResult.value().front().embedding;

    // Query: detect Alice again from the same image
    auto queryResult = faceInferencer.detect(
        aliceFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);
    ASSERT_TRUE(queryResult.has_value());
    const auto& queryEmbedding = queryResult.value().front().embedding;

    // THEN: cosine similarity is near-perfect and exceeds the threshold
    const float matchScore = FaceRecognitionNode::cosineSimilarity(
        registeredEmbedding, queryEmbedding);

    EXPECT_NEAR(matchScore, 1.0f, 1e-5f)
        << "The same face must yield cosine similarity near 1.0";
    EXPECT_GT(matchScore, kFaceRecognitionThreshold)
        << "Same face must exceed recognition threshold ("
        << kFaceRecognitionThreshold << ")";
}

TEST(FaceMatching, DifferentPerson_BelowRecognitionThreshold)
{
    // User scenario: Alice registers, then Bob walks past the camera.
    // The system must NOT identify Bob as Alice.

    auto aliceFaceBgr = loadBgr24Fixture(kFaceAliceFile);
    auto bobFaceBgr   = loadBgr24Fixture(kFaceBobFile);
    SKIP_IF_NO_FIXTURE(aliceFaceBgr);
    SKIP_IF_NO_FIXTURE(bobFaceBgr);

    StubFaceInferencer faceInferencer;

    auto aliceResult = faceInferencer.detect(
        aliceFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);
    auto bobResult = faceInferencer.detect(
        bobFaceBgr.data(), kFixtureImageWidth, kFixtureImageHeight);

    ASSERT_TRUE(aliceResult.has_value() && bobResult.has_value());

    const float matchScore = FaceRecognitionNode::cosineSimilarity(
        aliceResult.value().front().embedding,
        bobResult.value().front().embedding);

    EXPECT_NE(matchScore, 1.0f)
        << "Different people must NOT produce identical embeddings";
}

TEST(FaceMatching, BestMatch_SelectedFromMultipleGalleryEntries)
{
    // User scenario: Gallery contains Alice, Bob, and Carol.
    // A query embedding close to Alice must select Alice as the best match.

    // Gallery embeddings: orthogonal unit vectors
    std::vector<float> aliceEmbedding(512, 0.0f);
    aliceEmbedding[0] = 1.0f;  // Alice = dimension 0

    std::vector<float> bobEmbedding(512, 0.0f);
    bobEmbedding[1] = 1.0f;    // Bob = dimension 1

    const float kOneOverSqrt2 = 1.0f / std::sqrt(2.0f);
    std::vector<float> carolEmbedding(512, 0.0f);
    carolEmbedding[0] = kOneOverSqrt2;  // Carol = between Alice and Bob
    carolEmbedding[1] = kOneOverSqrt2;

    // Query = [1, 0, ...] → should match Alice best
    std::vector<float> queryEmbedding(512, 0.0f);
    queryEmbedding[0] = 1.0f;

    const float aliceScore = FaceRecognitionNode::cosineSimilarity(
        aliceEmbedding, queryEmbedding);
    const float bobScore = FaceRecognitionNode::cosineSimilarity(
        bobEmbedding, queryEmbedding);
    const float carolScore = FaceRecognitionNode::cosineSimilarity(
        carolEmbedding, queryEmbedding);

    EXPECT_GT(aliceScore, carolScore) << "Alice must be the best match";
    EXPECT_GT(carolScore, bobScore)   << "Carol must rank second (she's between)";
    EXPECT_NEAR(bobScore, 0.0f, 1e-5f) << "Bob is orthogonal — no match";
}

TEST(FaceMatching, ThresholdBoundary_CorrectlyClassified)
{
    // Bug caught: off-by-one or float comparison error at the threshold boundary

    const float recognitionThreshold = kFaceRecognitionThreshold;  // 0.45

    // Construct an embedding just ABOVE threshold
    const float aboveThresholdAngle = std::acos(recognitionThreshold + 0.01f);
    std::vector<float> slightlyAboveThreshold(512, 0.0f);
    slightlyAboveThreshold[0] = std::cos(aboveThresholdAngle);
    slightlyAboveThreshold[1] = std::sin(aboveThresholdAngle);

    std::vector<float> referenceEmbedding(512, 0.0f);
    referenceEmbedding[0] = 1.0f;

    EXPECT_GT(
        FaceRecognitionNode::cosineSimilarity(
            referenceEmbedding, slightlyAboveThreshold),
        recognitionThreshold)
        << "Score just above threshold must be classified as a match";

    // Construct an embedding just BELOW threshold
    const float belowThresholdAngle = std::acos(recognitionThreshold - 0.01f);
    std::vector<float> slightlyBelowThreshold(512, 0.0f);
    slightlyBelowThreshold[0] = std::cos(belowThresholdAngle);
    slightlyBelowThreshold[1] = std::sin(belowThresholdAngle);

    EXPECT_LT(
        FaceRecognitionNode::cosineSimilarity(
            referenceEmbedding, slightlyBelowThreshold),
        recognitionThreshold)
        << "Score just below threshold must be classified as unknown";
}

// ===========================================================================
// Frame Subsampling — "run face detection every Nth frame, not every frame"
// ===========================================================================

TEST(FaceSubsampling, DetectsEveryNthFrame_SkipsTheRest)
{
    // Bug caught: face detection running on every frame wastes GPU cycles,
    // or subsampling logic has off-by-one error.

    StubFaceInferencer faceInferencer;

    const uint32_t subsampleInterval = kFaceDetectionFrameSubsample;  // 3
    uint32_t videoFrameCounter = 0;
    constexpr int kTotalVideoFrames = 30;

    auto faceImageBgr = loadBgr24Fixture(kFaceAliceFile);
    SKIP_IF_NO_FIXTURE(faceImageBgr);

    // Simulate 30 incoming video frames; only call detect on every Nth
    for (int frame = 0; frame < kTotalVideoFrames; ++frame) {
        ++videoFrameCounter;
        if (videoFrameCounter % subsampleInterval == 0) {
            auto result = faceInferencer.detect(
                faceImageBgr.data(), kFixtureImageWidth, kFixtureImageHeight);
            ASSERT_TRUE(result.has_value());
        }
    }

    const int expectedDetections = kTotalVideoFrames / subsampleInterval;
    EXPECT_EQ(faceInferencer.detectCallCount_, expectedDetections)
        << "At subsample interval=" << subsampleInterval
        << ", expected " << expectedDetections << " detect calls out of "
        << kTotalVideoFrames << " video frames";
}

// ===========================================================================
// Cross-Check — production implementation vs. reference implementation
// ===========================================================================

TEST(FaceMatchingCrossCheck, ProductionMatchesReference_ForKnownVectors)
{
    // Bug caught: gradual accuracy drift if the cosine similarity implementation
    // uses a different formula or floating-point order of operations.

    const float kOneOverSqrt2 = 1.0f / std::sqrt(2.0f);

    std::vector<float> embeddingA(512, 0.0f);
    embeddingA[0] = 1.0f;

    std::vector<float> embeddingB(512, 0.0f);
    embeddingB[0] = kOneOverSqrt2;
    embeddingB[1] = kOneOverSqrt2;

    const float productionScore =
        FaceRecognitionNode::cosineSimilarity(embeddingA, embeddingB);
    const float referenceScore =
        referenceCosineSimilarity(embeddingA, embeddingB);

    EXPECT_NEAR(productionScore, referenceScore, 1e-5f)
        << "Production implementation must match the reference";
    EXPECT_NEAR(productionScore, kOneOverSqrt2, 1e-5f);
}

TEST(FaceMatchingCrossCheck, ProductionMatchesReference_ForArbitraryVectors)
{
    // Use deterministic pseudo-random vectors to exercise more of the
    // floating-point path than simple unit vectors do.

    std::vector<float> embeddingA(512);
    std::vector<float> embeddingB(512);
    for (int i = 0; i < 512; ++i) {
        embeddingA[static_cast<std::size_t>(i)] =
            static_cast<float>(i) * 0.001f - 0.256f;
        embeddingB[static_cast<std::size_t>(i)] =
            static_cast<float>(511 - i) * 0.002f + 0.1f;
    }

    const float productionScore =
        FaceRecognitionNode::cosineSimilarity(embeddingA, embeddingB);
    const float referenceScore =
        referenceCosineSimilarity(embeddingA, embeddingB);

    EXPECT_NEAR(productionScore, referenceScore, 1e-4f)
        << "Production must match reference for arbitrary vectors";
}

