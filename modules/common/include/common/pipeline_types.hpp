#pragma once

#include <cstdint>
#include <string_view>

// ---------------------------------------------------------------------------
// OmniEdge_AI — Pipeline Output Format
//
// Every CV node writes processed frames to a POSIX shared-memory segment.
// This enum selects the format and SHM container type:
//
//   kJpeg  — Terminal output for browser display.  The node encodes the
//            frame to JPEG (nvJPEG on GPU or cv::imencode fallback) and
//            writes it to a ShmDoubleBuffer<ShmJpegControl>.  The
//            WebSocketBridge reads the JPEG and relays it over WebSocket.
//
//   kBgr24 — Intermediate output for pipeline chaining.  The node writes
//            raw BGR24 pixels to a ShmCircularBuffer<ShmVideoHeader>.
//            A downstream CV node opens the same segment as its input,
//            avoiding an encode/decode round-trip (~3 ms at 1080p).
//
// BGR24 is the native pixel format throughout the pipeline — see
// constants/video_constants.hpp for the rationale.
//
// Both modes share the same OutputFormat enum so a single YAML field
// ("output_format: bgr24" or "output_format: jpeg") toggles the behavior
// at deploy time without recompilation.
// ---------------------------------------------------------------------------

enum class OutputFormat : uint8_t {
	kJpeg  = 0,   ///< JPEG via ShmDoubleBuffer — for WebSocketBridge display
	kBgr24 = 1,   ///< Raw BGR24 via ShmCircularBuffer — for downstream chaining
};

[[nodiscard]] inline constexpr OutputFormat parseOutputFormat(std::string_view s) noexcept
{
	if (s == "bgr24" || s == "BGR24") return OutputFormat::kBgr24;
	return OutputFormat::kJpeg;  // default
}

[[nodiscard]] inline constexpr std::string_view outputFormatName(OutputFormat f) noexcept
{
	switch (f) {
	case OutputFormat::kBgr24: return "bgr24";
	case OutputFormat::kJpeg:  return "jpeg";
	}
	return "jpeg";
}
