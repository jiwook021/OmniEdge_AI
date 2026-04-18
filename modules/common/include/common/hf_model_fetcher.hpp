#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Hugging Face model fetcher
//
// Single entry point for runtime model auto-download. Invokes the shared
// scripts/helpers/hf_fetch.py via SubprocessManager::runOnce and returns
// the local path the file was written to.
//
// Idempotent: if the file already exists under cacheDir it is returned
// immediately without contacting the network.
//
// Usage:
//     auto path = fetchHfModel("fal/AuraFace-v1", "glintr100.onnx",
//                              modelsRoot / "face_recog");
//     if (!path) {
//         OE_LOG_ERROR("hf_fetch_failed: {}", path.error());
//         return;
//     }
//     session = loadOnnx(path.value());
// ---------------------------------------------------------------------------

#include <chrono>
#include <filesystem>
#include <string>

#include <tl/expected.hpp>


/// Download (or locate cached) a single file from the Hugging Face Hub.
///
/// @param repo      HF repo id, e.g. "fal/AuraFace-v1".
/// @param file      Filename inside the repo, e.g. "glintr100.onnx".
/// @param cacheDir  Local directory to place the file under.
/// @param timeout   Max time to spend waiting for the download
///                  (ignored if the file is already cached).
///
/// @returns The absolute local path on success, or a diagnostic string
///          on failure. The file is guaranteed to exist at
///          (cacheDir / file) when the expected is populated.
[[nodiscard]] tl::expected<std::filesystem::path, std::string>
fetchHfModel(const std::string&             repo,
             const std::string&             file,
             const std::filesystem::path&   cacheDir,
             std::chrono::seconds           timeout = std::chrono::minutes(10));
