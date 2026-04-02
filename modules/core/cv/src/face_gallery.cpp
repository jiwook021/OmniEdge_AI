#include "cv/face_gallery.hpp"

#include <cstring>
#include <format>
#include <stdexcept>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


namespace {

// Execute a SQL statement, throw on error
void sqlCheck(int rc, sqlite3* db, const char* context)
{
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
        throw std::runtime_error(
            std::format("[FaceGallery] SQLite error in {}: {} (code {})",
                        context, sqlite3_errmsg(db), rc));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

FaceGallery::FaceGallery(const std::string& dbPath)
{
    const int rc = sqlite3_open(dbPath.c_str(), &db_);
    if (rc != SQLITE_OK) {
        const std::string msg = db_ ? sqlite3_errmsg(db_) : "unknown";
        sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error(
            std::format("[FaceGallery] sqlite3_open('{}') failed: {}", dbPath, msg));
    }
    createSchema();
    OE_LOG_INFO("face_gallery_opened: db={}", dbPath);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

FaceGallery::~FaceGallery()
{
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// createSchema()
// ---------------------------------------------------------------------------

void FaceGallery::createSchema()
{
    const char* sql = R"sql(
        CREATE TABLE IF NOT EXISTS known_faces (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL UNIQUE,
            embedding BLOB    NOT NULL
        );
    )sql";

    char* errmsg = nullptr;
    const int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error(
            std::format("[FaceGallery] Schema creation failed: {}", msg));
    }
}

// ---------------------------------------------------------------------------
// upsert()
// ---------------------------------------------------------------------------

void FaceGallery::upsert(const std::string& name, const float* embedding)
{
    const char* sql =
        "INSERT INTO known_faces (name, embedding) VALUES (?1, ?2) "
        "ON CONFLICT(name) DO UPDATE SET embedding = excluded.embedding;";

    sqlite3_stmt* stmt = nullptr;
    sqlCheck(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr),
             db_, "upsert prepare");

    sqlCheck(sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC),
             db_, "upsert bind name");

    const int blobBytes = kGalleryEmbeddingDim * static_cast<int>(sizeof(float));
    sqlCheck(sqlite3_bind_blob(stmt, 2, embedding, blobBytes, SQLITE_STATIC),
             db_, "upsert bind embedding");

    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(
            std::format("[FaceGallery] upsert step failed: {} (code {})",
                        sqlite3_errmsg(db_), rc));
    }
}

// ---------------------------------------------------------------------------
// remove()
// ---------------------------------------------------------------------------

void FaceGallery::remove(const std::string& name)
{
    const char* sql = "DELETE FROM known_faces WHERE name = ?1;";

    sqlite3_stmt* stmt = nullptr;
    sqlCheck(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr),
             db_, "remove prepare");
    sqlCheck(sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC),
             db_, "remove bind name");

    const int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        throw std::runtime_error(
            std::format("[FaceGallery] remove step failed: {} (code {})",
                        sqlite3_errmsg(db_), rc));
    }
}

// ---------------------------------------------------------------------------
// queryAll()
// ---------------------------------------------------------------------------

std::vector<GalleryEntry> FaceGallery::queryAll() const
{
    const char* sql = "SELECT name, embedding FROM known_faces ORDER BY id;";

    sqlite3_stmt* stmt = nullptr;
    sqlCheck(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr),
             db_, "queryAll prepare");

    std::vector<GalleryEntry> entries;

    while (true) {
        const int rc = sqlite3_step(stmt);
        if (rc == SQLITE_DONE) { break; }
        if (rc != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            throw std::runtime_error(
                std::format("[FaceGallery] queryAll step failed: {} (code {})",
                            sqlite3_errmsg(db_), rc));
        }

        GalleryEntry entry;
        const auto* namePtr = sqlite3_column_text(stmt, 0);
        if (!namePtr) { continue; }  // NULL name — skip corrupt row
        entry.name = reinterpret_cast<const char*>(namePtr);

        const int blobBytes = sqlite3_column_bytes(stmt, 1);
        const int expectedBytes =
            kGalleryEmbeddingDim * static_cast<int>(sizeof(float));

        if (blobBytes != expectedBytes) {
            OE_LOG_WARN("face_gallery_bad_embedding_size: name={}, bytes={}",
                    entry.name, blobBytes);
            continue;
        }

        entry.embedding.resize(kGalleryEmbeddingDim);
        std::memcpy(entry.embedding.data(),
                    sqlite3_column_blob(stmt, 1),
                    static_cast<std::size_t>(blobBytes));

        entries.push_back(std::move(entry));
    }

    sqlite3_finalize(stmt);
    return entries;
}

// ---------------------------------------------------------------------------
// size()
// ---------------------------------------------------------------------------

std::size_t FaceGallery::size() const
{
    const char* sql = "SELECT COUNT(*) FROM known_faces;";

    sqlite3_stmt* stmt = nullptr;
    sqlCheck(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr),
             db_, "size prepare");

    const int rc = sqlite3_step(stmt);
    std::size_t count = 0;
    if (rc == SQLITE_ROW) {
        count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
    }
    sqlite3_finalize(stmt);
    return count;
}

