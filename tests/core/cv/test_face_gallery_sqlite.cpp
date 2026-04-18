#include <gtest/gtest.h>

#include "cv/face_gallery.hpp"

#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Face Gallery SQLite Tests
//
// Purpose: Verify the face embedding database (SQLite) that stores known
// people for face recognition.  From the user's perspective:
//   "I register Alice's face → the system remembers her."
//   "I remove Bob → the system forgets him."
//   "I update Carol's photo → her embedding is replaced, not duplicated."
//
// What these tests catch:
//   - Schema creation failure on first open
//   - INSERT creates duplicates instead of UPSERT overwriting
//   - DELETE leaves orphaned rows
//   - Embedding bytes corrupted during SQLite BLOB round-trip
//   - Size count out of sync with actual table rows
//
// All tests use SQLite ":memory:" databases — no files created on disk.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Create an in-memory face gallery (no disk files).
static FaceGallery createEmptyGallery()
{
    return FaceGallery(":memory:");
}

/// Create a synthetic 512-dimensional face embedding filled with a single value.
static std::vector<float> createFakeEmbedding(float fillValue)
{
    return std::vector<float>(kGalleryEmbeddingDim, fillValue);
}

// ===========================================================================
// Schema and Initialization
// ===========================================================================

TEST(FaceGallery, OpenInMemory_Succeeds)
{
    EXPECT_NO_THROW({ auto gallery = createEmptyGallery(); });
}

TEST(FaceGallery, EmptyGallery_HasZeroEntries)
{
    auto gallery = createEmptyGallery();
    EXPECT_EQ(gallery.size(), 0u);
}

// ===========================================================================
// Register (Upsert) — "add a person to the gallery"
// ===========================================================================

TEST(FaceGallery, RegisterOnePerson_GallerySizeIsOne)
{
    auto gallery = createEmptyGallery();
    auto aliceEmbedding = createFakeEmbedding(0.1f);

    gallery.upsert("Alice", aliceEmbedding.data());

    EXPECT_EQ(gallery.size(), 1u);
}

TEST(FaceGallery, RegisterTwoPeople_GallerySizeIsTwo)
{
    auto gallery = createEmptyGallery();
    auto aliceEmbedding = createFakeEmbedding(0.1f);
    auto bobEmbedding   = createFakeEmbedding(0.2f);

    gallery.upsert("Alice", aliceEmbedding.data());
    gallery.upsert("Bob",   bobEmbedding.data());

    EXPECT_EQ(gallery.size(), 2u);
}

TEST(FaceGallery, RegisterSamePersonTwice_OverwritesInsteadOfDuplicating)
{
    // Bug caught: INSERT without ON CONFLICT creates duplicate rows

    auto gallery = createEmptyGallery();
    auto originalEmbedding = createFakeEmbedding(0.1f);
    auto updatedEmbedding  = createFakeEmbedding(0.9f);

    gallery.upsert("Alice", originalEmbedding.data());
    gallery.upsert("Alice", updatedEmbedding.data());  // should OVERWRITE

    EXPECT_EQ(gallery.size(), 1u) << "UPSERT must not create a duplicate entry";

    const auto allEntries = gallery.queryAll();
    ASSERT_EQ(allEntries.size(), 1u);
    EXPECT_FLOAT_EQ(allEntries[0].embedding[0], 0.9f)
        << "The embedding must be the updated value, not the original";
}

// ===========================================================================
// Remove — "forget a person"
// ===========================================================================

TEST(FaceGallery, RemoveExistingPerson_GalleryBecomesEmpty)
{
    auto gallery = createEmptyGallery();
    auto aliceEmbedding = createFakeEmbedding(0.1f);

    gallery.upsert("Alice", aliceEmbedding.data());
    EXPECT_EQ(gallery.size(), 1u);

    gallery.remove("Alice");
    EXPECT_EQ(gallery.size(), 0u);
}

TEST(FaceGallery, RemoveNonexistentPerson_IsNoOp)
{
    auto gallery = createEmptyGallery();
    auto aliceEmbedding = createFakeEmbedding(0.1f);
    gallery.upsert("Alice", aliceEmbedding.data());

    EXPECT_NO_THROW(gallery.remove("UnknownPerson"));
    EXPECT_EQ(gallery.size(), 1u)
        << "Removing a name not in the gallery must not affect existing entries";
}

TEST(FaceGallery, RemoveOneOfTwo_OtherRemains)
{
    auto gallery = createEmptyGallery();
    gallery.upsert("Alice", createFakeEmbedding(0.1f).data());
    gallery.upsert("Bob",   createFakeEmbedding(0.2f).data());

    gallery.remove("Alice");

    const auto remainingEntries = gallery.queryAll();
    ASSERT_EQ(remainingEntries.size(), 1u);
    EXPECT_EQ(remainingEntries[0].name, "Bob")
        << "Removing Alice must leave Bob intact";
}

// ===========================================================================
// Embedding Round-Trip — "stored bytes match original floats"
// ===========================================================================

TEST(FaceGallery, EmbeddingRoundTrip_StoredBytesMatchOriginal)
{
    // Bug caught: BLOB serialization truncates or corrupts float values
    // (endianness, sizeof mismatch, missing bytes)

    auto gallery = createEmptyGallery();

    // Create a non-trivial embedding with a range of float values
    std::vector<float> originalEmbedding(kGalleryEmbeddingDim);
    for (int i = 0; i < kGalleryEmbeddingDim; ++i) {
        originalEmbedding[i] = static_cast<float>(i) * 0.001f - 0.256f;
    }

    gallery.upsert("Carol", originalEmbedding.data());

    const auto allEntries = gallery.queryAll();
    ASSERT_EQ(allEntries.size(), 1u);
    ASSERT_EQ(allEntries[0].name, "Carol");
    ASSERT_EQ(static_cast<int>(allEntries[0].embedding.size()),
              kGalleryEmbeddingDim);

    for (int i = 0; i < kGalleryEmbeddingDim; ++i) {
        EXPECT_FLOAT_EQ(allEntries[0].embedding[i], originalEmbedding[i])
            << "Embedding value mismatch at index " << i
            << " — BLOB serialization may be corrupted";
    }
}

TEST(FaceGallery, QueryAll_ReturnsBothEntriesInInsertionOrder)
{
    auto gallery = createEmptyGallery();
    gallery.upsert("Alice", createFakeEmbedding(1.0f).data());
    gallery.upsert("Bob",   createFakeEmbedding(2.0f).data());

    const auto allEntries = gallery.queryAll();
    ASSERT_EQ(allEntries.size(), 2u);

    EXPECT_EQ(allEntries[0].name, "Alice");
    EXPECT_EQ(allEntries[1].name, "Bob");

    EXPECT_FLOAT_EQ(allEntries[0].embedding[0], 1.0f);
    EXPECT_FLOAT_EQ(allEntries[1].embedding[0], 2.0f);
}

