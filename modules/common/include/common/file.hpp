#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Consolidated File Utilities
//
// Common file I/O operations used across all modules. Every function returns
// tl::expected for fallible operations — no exceptions escape to the caller.
//
// All implementations live in file.cpp (not here). This header is
// declaration-only per OE coding standards.
//
// Usage:
//   #include "common/file.hpp"
//
//   auto text = readText("config.yaml");
//   auto blob = readBinary("model.onnx");
//   auto json = readJson("vocab.json");
//   auto ok   = writeText("out.txt", contents);
//   auto ok2  = writeBinary("frame.bin", data);
//   auto ok3  = atomicWrite("session.json", payload);
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <tl/expected.hpp>


// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

/// Read an entire file into a std::string (text mode).
/// Returns the file contents or an error message on failure.
[[nodiscard]] tl::expected<std::string, std::string>
readText(const std::filesystem::path& path);

/// Read an entire file into a byte vector (binary mode).
/// Returns the raw bytes or an error message on failure.
[[nodiscard]] tl::expected<std::vector<uint8_t>, std::string>
readBinary(const std::filesystem::path& path);

/// Read and parse a JSON file via nlohmann::json.
/// Forward-declared to avoid pulling nlohmann/json.hpp into every includer —
/// the caller must #include <nlohmann/json.hpp> before calling this.
/// Returns the parsed JSON or an error message on failure (open, parse).
///
/// Note: implemented in file.cpp which includes nlohmann/json.hpp.
/// The return type uses a forward-friendly wrapper; see file.cpp.
[[nodiscard]] tl::expected<std::string, std::string>
readJsonString(const std::filesystem::path& path);

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

/// Write a string to a file (text mode, truncate).
/// Creates parent directories if they do not exist.
/// Returns void on success or an error message on failure.
[[nodiscard]] tl::expected<void, std::string>
writeText(const std::filesystem::path& path, std::string_view contents);

/// Write raw bytes to a file (binary mode, truncate).
/// Creates parent directories if they do not exist.
/// Returns void on success or an error message on failure.
[[nodiscard]] tl::expected<void, std::string>
writeBinary(const std::filesystem::path& path,
            std::span<const uint8_t> data);

/// Atomic write: writes contents to a temporary file in the same directory,
/// then renames to the final path. Guarantees the target is never partially
/// written. Creates parent directories if they do not exist.
/// Returns void on success or an error message on failure.
[[nodiscard]] tl::expected<void, std::string>
atomicWrite(const std::filesystem::path& path, std::string_view contents);

// ---------------------------------------------------------------------------
// Filesystem queries
// ---------------------------------------------------------------------------

/// Check whether a path exists (file or directory). Swallows filesystem
/// errors and returns false on any error condition.
[[nodiscard]] bool exists(const std::filesystem::path& path) noexcept;

/// Check whether a path is a regular file.
[[nodiscard]] bool isFile(const std::filesystem::path& path) noexcept;

/// Check whether a path is a directory.
[[nodiscard]] bool isDirectory(const std::filesystem::path& path) noexcept;

/// Get the size of a file in bytes. Returns 0 on error or if not a file.
[[nodiscard]] std::size_t fileSize(const std::filesystem::path& path) noexcept;

/// Ensure that all parent directories for the given path exist.
/// Creates them recursively if needed.
/// Returns void on success or an error message on failure.
[[nodiscard]] tl::expected<void, std::string>
ensureParentDirs(const std::filesystem::path& path);

