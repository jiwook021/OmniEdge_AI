#pragma once

#include <cstddef>
#include <format>
#include <string>

#include <cuda_runtime.h>

#include <tl/expected.hpp>
#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"
#include "common/constants/memory_constants.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — VRAM Pre-Flight Guard
//
// Queries free VRAM via cudaMemGetInfo and rejects an inference request
// before the inferencer ever attempts an allocation.  This prevents CUDA from
// entering an unrecoverable error state after OOM — the caller receives
// a clean tl::expected failure and can skip the frame or trigger eviction.
//
// TOCTOU caveat: another process may allocate between the check and the
// actual inference.  The guard eliminates the *common* case (steady-state
// VRAM pressure) but is not a hard guarantee.
//
// Usage:
//   if (auto ok = ensureVramAvailable(peakInferenceBytes_); !ok) {
//       OE_LOG_WARN("preflight_failed: {}", ok.error());
//       return tl::unexpected(ok.error());
//   }
// ---------------------------------------------------------------------------


/// Check that at least `requiredBytes` of VRAM is free on the current device.
/// Returns void on success or a descriptive error string on failure.
[[nodiscard]] inline tl::expected<void, std::string>
ensureVramAvailable(std::size_t requiredBytes)
{
	OE_ZONE_SCOPED;

	std::size_t freeBytes  = 0;
	std::size_t totalBytes = 0;
	const cudaError_t err  = cudaMemGetInfo(&freeBytes, &totalBytes);
	if (err != cudaSuccess) {
		return tl::unexpected(
			std::format("cudaMemGetInfo failed: {} ({})",
			            cudaGetErrorName(err), cudaGetErrorString(err)));
	}

	const std::size_t freeMiB     = freeBytes  / kBytesPerMebibyte;
	const std::size_t totalMiB    = totalBytes / kBytesPerMebibyte;
	const std::size_t requiredMiB = requiredBytes / kBytesPerMebibyte;

	OE_LOG_DEBUG("vram_preflight: required={} MiB, free={} MiB, total={} MiB",
	             requiredMiB, freeMiB, totalMiB);

	if (freeBytes < requiredBytes) {
		return tl::unexpected(
			std::format("insufficient VRAM: need {} MiB, free {} MiB (total {} MiB)",
			            requiredMiB, freeMiB, totalMiB));
	}

	return {};
}

/// Convenience overload accepting mebibytes (matches VRAM budget constants).
[[nodiscard]] inline tl::expected<void, std::string>
ensureVramAvailableMiB(std::size_t requiredMiB)
{
	return ensureVramAvailable(requiredMiB * kBytesPerMebibyte);
}

