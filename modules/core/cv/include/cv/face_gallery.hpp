#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <sqlite3.h>

// ---------------------------------------------------------------------------
// OmniEdge_AI — SQLite Face Embedding Gallery
//
// Persists name → 512-dim float embedding mappings in a SQLite database.
//
// Schema:
//   CREATE TABLE IF NOT EXISTS known_faces (
//       id        INTEGER PRIMARY KEY AUTOINCREMENT,
//       name      TEXT NOT NULL UNIQUE,
//       embedding BLOB NOT NULL          -- 512 × sizeof(float) = 2048 bytes
//   );
//
// Thread safety: none. One FaceGallery per module instance.
// ---------------------------------------------------------------------------


/// Dimensionality of stored face embeddings.
inline constexpr int kGalleryEmbeddingDim = 512;

// ---------------------------------------------------------------------------
// GalleryEntry — one row returned from queryAll()
// ---------------------------------------------------------------------------

struct GalleryEntry {
    std::string        name;
    std::vector<float> embedding;  ///< kGalleryEmbeddingDim floats
};

// ---------------------------------------------------------------------------
// FaceGallery
// ---------------------------------------------------------------------------

class FaceGallery {
public:
    /**
     * @brief Open (or create) the SQLite database at dbPath.
     * @throws std::runtime_error on sqlite3_open or schema creation failure.
     */
    explicit FaceGallery(const std::string& dbPath);
    ~FaceGallery();

    FaceGallery(const FaceGallery&)            = delete;
    FaceGallery& operator=(const FaceGallery&) = delete;

    /**
     * @brief Insert or replace a face embedding.
     *
     * If a row with the given name already exists it is overwritten (UPSERT).
     *
     * @param name       Identity label (e.g. "Admin").
     * @param embedding  Float array of exactly kGalleryEmbeddingDim elements.
     * @throws std::runtime_error on SQL execution failure.
     */
    void upsert(const std::string& name,
                const float*       embedding);

    /**
     * @brief Remove a face entry by name.  No-op if the name does not exist.
     * @throws std::runtime_error on SQL execution failure.
     */
    void remove(const std::string& name);

    /**
     * @brief Return all stored name–embedding pairs.
     * @throws std::runtime_error on SQL query failure.
     */
    [[nodiscard]] std::vector<GalleryEntry> queryAll() const;

    /**
     * @brief Number of entries currently stored.
     */
    [[nodiscard]] std::size_t size() const;

private:
    void createSchema();
    void prepareStatements();
    void finalizeStatements();

    sqlite3* db_ = nullptr;

    // Prepared statement cache — prepared once, reused via sqlite3_reset().
    // Mutable because queryAll() and size() are const but sqlite3_reset/step
    // mutate the statement's internal cursor state (not the database).
    sqlite3_stmt*         stmtUpsert_   = nullptr;
    sqlite3_stmt*         stmtRemove_   = nullptr;
    mutable sqlite3_stmt* stmtQueryAll_ = nullptr;
    mutable sqlite3_stmt* stmtSize_     = nullptr;
};

