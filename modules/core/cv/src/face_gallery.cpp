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
    prepareStatements();
    OE_LOG_INFO("face_gallery_opened: db={}", dbPath);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

FaceGallery::~FaceGallery()
{
    finalizeStatements();
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
// prepareStatements() / finalizeStatements()
// ---------------------------------------------------------------------------

void FaceGallery::prepareStatements()
{
    auto prep = [&](const char* sql, sqlite3_stmt*& out, const char* label) {
        sqlCheck(sqlite3_prepare_v2(db_, sql, -1, &out, nullptr), db_, label);
    };

    prep("INSERT INTO known_faces (name, embedding) VALUES (?1, ?2) "
         "ON CONFLICT(name) DO UPDATE SET embedding = excluded.embedding;",
         stmtUpsert_, "prepare upsert");

    prep("DELETE FROM known_faces WHERE name = ?1;",
         stmtRemove_, "prepare remove");

    prep("SELECT name, embedding FROM known_faces ORDER BY id;",
         stmtQueryAll_, "prepare queryAll");

    prep("SELECT COUNT(*) FROM known_faces;",
         stmtSize_, "prepare size");
}

void FaceGallery::finalizeStatements()
{
    auto fin = [](sqlite3_stmt*& s) {
        if (s) { sqlite3_finalize(s); s = nullptr; }
    };
    fin(stmtUpsert_);
    fin(stmtRemove_);
    fin(stmtQueryAll_);
    fin(stmtSize_);
}

// ---------------------------------------------------------------------------
// upsert()
// ---------------------------------------------------------------------------

void FaceGallery::upsert(const std::string& name, const float* embedding)
{
    sqlite3_reset(stmtUpsert_);
    sqlite3_clear_bindings(stmtUpsert_);

    sqlCheck(sqlite3_bind_text(stmtUpsert_, 1, name.c_str(), -1, SQLITE_TRANSIENT),
             db_, "upsert bind name");

    const int blobBytes = kGalleryEmbeddingDim * static_cast<int>(sizeof(float));
    sqlCheck(sqlite3_bind_blob(stmtUpsert_, 2, embedding, blobBytes, SQLITE_TRANSIENT),
             db_, "upsert bind embedding");

    const int rc = sqlite3_step(stmtUpsert_);

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
    sqlite3_reset(stmtRemove_);
    sqlite3_clear_bindings(stmtRemove_);

    sqlCheck(sqlite3_bind_text(stmtRemove_, 1, name.c_str(), -1, SQLITE_TRANSIENT),
             db_, "remove bind name");

    const int rc = sqlite3_step(stmtRemove_);

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
    sqlite3_reset(stmtQueryAll_);

    std::vector<GalleryEntry> entries;

    while (true) {
        const int rc = sqlite3_step(stmtQueryAll_);
        if (rc == SQLITE_DONE) { break; }
        if (rc != SQLITE_ROW) {
            throw std::runtime_error(
                std::format("[FaceGallery] queryAll step failed: {} (code {})",
                            sqlite3_errmsg(db_), rc));
        }

        GalleryEntry entry;
        const auto* namePtr = sqlite3_column_text(stmtQueryAll_, 0);
        if (!namePtr) { continue; }  // NULL name — skip corrupt row
        entry.name = reinterpret_cast<const char*>(namePtr);

        const int blobBytes = sqlite3_column_bytes(stmtQueryAll_, 1);
        const int expectedBytes =
            kGalleryEmbeddingDim * static_cast<int>(sizeof(float));

        if (blobBytes != expectedBytes) {
            OE_LOG_WARN("face_gallery_bad_embedding_size: name={}, bytes={}",
                    entry.name, blobBytes);
            continue;
        }

        entry.embedding.resize(kGalleryEmbeddingDim);
        std::memcpy(entry.embedding.data(),
                    sqlite3_column_blob(stmtQueryAll_, 1),
                    static_cast<std::size_t>(blobBytes));

        entries.push_back(std::move(entry));
    }

    return entries;
}

// ---------------------------------------------------------------------------
// size()
// ---------------------------------------------------------------------------

std::size_t FaceGallery::size() const
{
    sqlite3_reset(stmtSize_);

    const int rc = sqlite3_step(stmtSize_);
    if (rc != SQLITE_ROW) {
        throw std::runtime_error(
            std::format("[FaceGallery] size() query failed: {} (code {})",
                        sqlite3_errmsg(db_), rc));
    }
    return static_cast<std::size_t>(sqlite3_column_int64(stmtSize_, 0));
}

