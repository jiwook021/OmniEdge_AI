#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — CV Test Helpers
//
// Shared utilities for all CV unit tests:
//   - loadBgr24File()   — reads a raw BGR24 fixture into a vector
//   - loadBinaryFile()  — reads any binary file into a vector
//   - fixtureDir()      — resolves the path to tests/cv/fixtures/
//
// These helpers let tests use actual image data from disk instead of
// synthetic buffers, making the tests realistic and easier to understand.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// Test fixture image dimensions (must match generate_test_images.py)
// ---------------------------------------------------------------------------

inline constexpr uint32_t kFixtureImageWidth  = 640;
inline constexpr uint32_t kFixtureImageHeight = 480;
inline constexpr std::size_t kFixtureBgrBytes =
    static_cast<std::size_t>(kFixtureImageWidth) * kFixtureImageHeight * 3;

// ---------------------------------------------------------------------------
// Fixture file names — these map to files in tests/cv/fixtures/
// ---------------------------------------------------------------------------

inline constexpr const char* kPersonSceneFile    = "person_scene_640x480.bgr24";
inline constexpr const char* kEmptyRoomFile      = "empty_room_640x480.bgr24";
inline constexpr const char* kFaceAliceFile      = "face_alice_640x480.bgr24";
inline constexpr const char* kFaceBobFile        = "face_bob_640x480.bgr24";
inline constexpr const char* kGroupPhotoFile     = "group_three_faces_640x480.bgr24";

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Returns the absolute path to the test fixtures directory for CV tests.
/// Searches common locations — the canonical path is tests/fixtures/cv/.
inline std::filesystem::path fixtureDir()
{
    // CMake sets the working directory; try common locations
    for (const auto& candidate : {
        std::filesystem::path(OE_PROJECT_ROOT) / "tests" / "fixtures" / "cv",
        std::filesystem::path(OE_PROJECT_ROOT) / "tests" / "cv" / "fixtures",
        std::filesystem::current_path() / "tests" / "fixtures" / "cv",
        std::filesystem::current_path() / "tests" / "cv" / "fixtures",
        std::filesystem::current_path().parent_path() / "tests" / "fixtures" / "cv",
    }) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    // Fallback: return the canonical path (tests will GTEST_SKIP if missing)
    return std::filesystem::path(OE_PROJECT_ROOT) / "tests" / "fixtures" / "cv";
}

// ---------------------------------------------------------------------------
// File loaders
// ---------------------------------------------------------------------------

/// Load a binary file into a byte vector.  Returns an empty vector if the
/// file does not exist (caller should GTEST_SKIP in that case).
inline std::vector<uint8_t> loadBinaryFile(const std::filesystem::path& filePath)
{
    if (!std::filesystem::exists(filePath)) {
        return {};
    }

    const auto fileSize = std::filesystem::file_size(filePath);
    std::vector<uint8_t> buffer(static_cast<std::size_t>(fileSize));

    std::ifstream stream(filePath, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(buffer.data()),
                     static_cast<std::streamsize>(fileSize))) {
        return {};
    }
    return buffer;
}

/// Load a BGR24 fixture image.  Returns the raw pixel bytes.
/// If the file is missing, returns an empty vector — the caller
/// should call GTEST_SKIP() << "fixture not found".
inline std::vector<uint8_t> loadBgr24Fixture(const std::string& fileName)
{
    const auto path = fixtureDir() / fileName;
    auto data = loadBinaryFile(path);
    if (!data.empty() && data.size() != kFixtureBgrBytes) {
        throw std::runtime_error(
            "Fixture size mismatch: " + path.string()
            + " is " + std::to_string(data.size())
            + " bytes, expected " + std::to_string(kFixtureBgrBytes));
    }
    return data;
}

/// Macro to skip a test if the fixture file is not available.
/// Usage:  SKIP_IF_NO_FIXTURE(imageData);
#define SKIP_IF_NO_FIXTURE(vec)                        \
    do {                                               \
        if ((vec).empty()) {                           \
            GTEST_SKIP() << "Test fixture not found. " \
                "Run: python3 tests/fixtures/cv/generate_test_images.py"; \
        }                                              \
    } while (false)

