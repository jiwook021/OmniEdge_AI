// file.cpp — Consolidated file utility implementations
//
// All file I/O helpers live here. Headers declare only; this file defines.
// Every fallible function returns tl::expected — no exceptions escape.

#include "common/file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "common/oe_tracy.hpp"


namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

tl::expected<std::string, std::string>
readText(const fs::path& path)
{
	OE_ZONE_SCOPED;

	std::ifstream file(path);
	if (!file.is_open()) {
		return tl::unexpected(
			"cannot open file for reading: " + path.string());
	}

	std::string contents(
		(std::istreambuf_iterator<char>(file)),
		 std::istreambuf_iterator<char>());

	if (file.bad()) {
		return tl::unexpected(
			"I/O error while reading: " + path.string());
	}

	return contents;
}

tl::expected<std::vector<uint8_t>, std::string>
readBinary(const fs::path& path)
{
	OE_ZONE_SCOPED;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		return tl::unexpected(
			"cannot open file for binary reading: " + path.string());
	}

	const auto size = file.tellg();
	if (size < 0) {
		return tl::unexpected(
			"cannot determine file size: " + path.string());
	}

	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> data(static_cast<std::size_t>(size));
	file.read(reinterpret_cast<char*>(data.data()),
	          static_cast<std::streamsize>(size));

	if (file.bad()) {
		return tl::unexpected(
			"I/O error while reading binary: " + path.string());
	}

	return data;
}

tl::expected<std::string, std::string>
readJsonString(const fs::path& path)
{
	// Read the raw JSON text — callers parse it with nlohmann::json themselves.
	// This avoids pulling nlohmann/json.hpp into the common header.
	return readText(path);
}

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

tl::expected<void, std::string>
writeText(const fs::path& path, std::string_view contents)
{
	OE_ZONE_SCOPED;

	// Ensure parent directory exists.
	if (auto result = ensureParentDirs(path); !result) {
		return tl::unexpected(result.error());
	}

	std::ofstream file(path, std::ios::out | std::ios::trunc);
	if (!file.is_open()) {
		return tl::unexpected(
			"cannot open file for writing: " + path.string());
	}

	file.write(contents.data(),
	           static_cast<std::streamsize>(contents.size()));

	if (file.bad()) {
		return tl::unexpected(
			"I/O error while writing: " + path.string());
	}

	return {};
}

tl::expected<void, std::string>
writeBinary(const fs::path& path, std::span<const uint8_t> data)
{
	OE_ZONE_SCOPED;

	// Ensure parent directory exists.
	if (auto result = ensureParentDirs(path); !result) {
		return tl::unexpected(result.error());
	}

	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (!file.is_open()) {
		return tl::unexpected(
			"cannot open file for binary writing: " + path.string());
	}

	file.write(reinterpret_cast<const char*>(data.data()),
	           static_cast<std::streamsize>(data.size()));

	if (file.bad()) {
		return tl::unexpected(
			"I/O error while writing binary: " + path.string());
	}

	return {};
}

tl::expected<void, std::string>
atomicWrite(const fs::path& path, std::string_view contents)
{
	OE_ZONE_SCOPED;

	// Ensure parent directory exists.
	if (auto result = ensureParentDirs(path); !result) {
		return tl::unexpected(result.error());
	}

	// Write to a temporary file in the same directory, then rename.
	// Same-directory rename is atomic on POSIX filesystems.
	const fs::path tmpPath = path.string() + ".tmp";

	{
		std::ofstream file(tmpPath, std::ios::out | std::ios::trunc);
		if (!file.is_open()) {
			return tl::unexpected(
				"cannot open temp file for atomic write: " + tmpPath.string());
		}

		file.write(contents.data(),
		           static_cast<std::streamsize>(contents.size()));

		if (file.bad()) {
			// Clean up partial temp file.
			std::error_code ec;
			fs::remove(tmpPath, ec);
			return tl::unexpected(
				"I/O error writing temp file: " + tmpPath.string());
		}
	} // close the file before rename

	// Atomic rename (POSIX guarantees atomicity for same-filesystem rename).
	if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
		const int savedErrno = errno;
		// Clean up the temp file on rename failure.
		std::error_code ec;
		fs::remove(tmpPath, ec);
		return tl::unexpected(
			"rename failed (" + std::string(std::strerror(savedErrno)) +
			"): " + tmpPath.string() + " -> " + path.string());
	}

	return {};
}

// ---------------------------------------------------------------------------
// Filesystem queries
// ---------------------------------------------------------------------------

bool exists(const fs::path& path) noexcept
{
	std::error_code ec;
	return fs::exists(path, ec);
}

bool isFile(const fs::path& path) noexcept
{
	std::error_code ec;
	return fs::is_regular_file(path, ec);
}

bool isDirectory(const fs::path& path) noexcept
{
	std::error_code ec;
	return fs::is_directory(path, ec);
}

std::size_t fileSize(const fs::path& path) noexcept
{
	std::error_code ec;
	const auto size = fs::file_size(path, ec);
	if (ec) return 0;
	return static_cast<std::size_t>(size);
}

tl::expected<void, std::string>
ensureParentDirs(const fs::path& path)
{
	const auto parentDir = path.parent_path();
	if (parentDir.empty()) return {};

	std::error_code ec;
	fs::create_directories(parentDir, ec);
	if (ec) {
		return tl::unexpected(
			"cannot create parent directories for " + path.string() +
			": " + ec.message());
	}

	return {};
}

