#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Shared SHM Utilities
//
// Eliminates duplicated SHM consumer retry loops and producer setup
// across all node configureTransport() implementations.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include <tl/expected.hpp>

#include "common/oe_logger.hpp"
#include "common/runtime_defaults.hpp"
#include "shm/shm_mapping.hpp"


namespace oe::shm {

// ---------------------------------------------------------------------------
// openConsumerWithRetry — open an existing SHM segment, retrying up to
// maxAttempts times with kInferencerRetryDelayMs between attempts.
//
// Previously duplicated identically in:
//   background_blur_node.cpp, face_filter_node.cpp, face_recog_node.cpp,
//   sam2_node.cpp, stt_node.cpp
//
// @param shmName      POSIX SHM name (e.g. "/oe.vid.ingest")
// @param shmSize      Expected segment size in bytes
// @param logPrefix    Module prefix for warning logs (e.g. "bg_blur")
// @param maxAttempts  Maximum retry count (default: 10)
// @return Unique pointer to opened ShmMapping, or error string
// ---------------------------------------------------------------------------
[[nodiscard]] inline tl::expected<std::unique_ptr<ShmMapping>, std::string>
openConsumerWithRetry(std::string_view shmName,
                      std::size_t shmSize,
                      std::string_view logPrefix,
                      int maxAttempts = 10)
{
	for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
		try {
			return std::make_unique<ShmMapping>(
				shmName, shmSize, /*create=*/false);
		} catch (const std::runtime_error& e) {
			if (attempt == maxAttempts) {
				return tl::unexpected(std::format(
					"SHM open failed after {} retries: {}", maxAttempts, e.what()));
			}
			OE_LOG_WARN("{}_shm_retry: attempt={}/{}, waiting for {}: {}",
				logPrefix, attempt, maxAttempts, shmName, e.what());
			std::this_thread::sleep_for(
				std::chrono::milliseconds{kInferencerRetryDelayMs});
		}
	}
	// Unreachable, but satisfies compiler return path analysis
	return tl::unexpected(std::string("SHM open failed"));
}

// ---------------------------------------------------------------------------
// createProducer — create a new SHM segment and zero-initialize it.
//
// Previously duplicated in:
//   background_blur_node.cpp, face_filter_node.cpp, video_denoise_node.cpp,
//   sam2_node.cpp, tts_node.cpp, audio_denoise_node.cpp
//
// @param shmName  POSIX SHM name (e.g. "/oe.cv.blur.jpeg")
// @param shmSize  Segment size in bytes
// @return Unique pointer to the created ShmMapping
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::unique_ptr<ShmMapping>
createProducer(std::string_view shmName, std::size_t shmSize)
{
	auto mapping = std::make_unique<ShmMapping>(shmName, shmSize, /*create=*/true);
	std::memset(mapping->data(), 0, shmSize);
	return mapping;
}

} // namespace oe::shm
