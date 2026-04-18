#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Typed ZMQ Message Structs
//
// Typed structs with auto-generated to_json/from_json via
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE.
//
// Usage:
//   VideoFrameMsg msg{.shm = "/oe.vid.ingest", .seq = seq, .ts = tsNs};
//   messageRouter_.publish("video_frame", nlohmann::json(msg));
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "common/constants/conversation_constants.hpp"
#include "common/runtime_defaults.hpp"


// ---------------------------------------------------------------------------
// Video Ingest — "video_frame" topic
// ---------------------------------------------------------------------------
struct VideoFrameMsg {
	int         v    = kSchemaVersion;
	std::string type = "video_frame";
	std::string shm  = "/oe.vid.ingest";
	uint64_t    seq  = 0;
	uint64_t    ts   = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VideoFrameMsg, v, type, shm, seq, ts)

// ---------------------------------------------------------------------------
// Screen Ingest — "screen_frame" topic
// ---------------------------------------------------------------------------
struct ScreenFrameMsg {
	int         v    = kSchemaVersion;
	std::string type = "screen_frame";
	std::string shm  = "/oe.screen.ingest";
	uint64_t    seq  = 0;
	uint64_t    ts   = 0;
	uint32_t    w    = 0;
	uint32_t    h    = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ScreenFrameMsg, v, type, shm, seq, ts, w, h)

// ---------------------------------------------------------------------------
// Audio Ingest — "audio_chunk" topic
// ---------------------------------------------------------------------------
struct AudioChunkMsg {
	int         v       = kSchemaVersion;
	std::string type    = "audio_chunk";
	std::string shm     = "/oe.aud.ingest";
	uint32_t    slot    = 0;
	uint32_t    samples = 0;
	uint64_t    seq     = 0;
	std::string vad     = "speech";
	int64_t     ts      = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioChunkMsg, v, type, shm, slot, samples, seq, vad, ts)

// ---------------------------------------------------------------------------
// Audio Denoise — "denoised_audio" topic
// ---------------------------------------------------------------------------
struct DenoisedAudioMsg {
	int         v       = kSchemaVersion;
	std::string type    = "denoised_audio";
	std::string shm;
	int64_t     samples = 0;
	uint64_t    seq     = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenoisedAudioMsg, v, type, shm, samples, seq)

// ---------------------------------------------------------------------------
// Background Blur — "blurred_frame" topic
// ---------------------------------------------------------------------------
struct BlurredFrameMsg {
	int         v       = kSchemaVersion;
	std::string type    = "blurred_frame";
	std::string shm;
	int64_t     size    = 0;
	uint64_t    seq     = 0;
	bool        blurred = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BlurredFrameMsg, v, type, shm, size, seq, blurred)

// ---------------------------------------------------------------------------
// BGR24 Intermediate Frame — "*_bgr_frame" topics (pipeline chaining)
// ---------------------------------------------------------------------------
struct BgrFrameMsg {
	int         v    = kSchemaVersion;
	std::string type;           ///< e.g. "blur_bgr_frame", "beauty_bgr_frame"
	std::string shm;            ///< e.g. "/oe.cv.blur.bgr"
	uint64_t    seq  = 0;
	uint32_t    w    = 0;
	uint32_t    h    = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BgrFrameMsg, v, type, shm, seq, w, h)

// ---------------------------------------------------------------------------
// Face Filter — "filtered_frame" topic
// ---------------------------------------------------------------------------
struct FilteredFrameMsg {
	int         v         = kSchemaVersion;
	std::string type      = "filtered_frame";
	std::string shm;
	int64_t     size      = 0;
	uint64_t    seq       = 0;
	bool        filtered  = false;
	std::string filter_id;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FilteredFrameMsg, v, type, shm, size, seq, filtered, filter_id)

// ---------------------------------------------------------------------------
// Video Denoise — "denoised_frame" topic
// ---------------------------------------------------------------------------
struct DenoisedFrameMsg {
	int         v        = kSchemaVersion;
	std::string type     = "denoised_frame";
	std::string shm;
	int64_t     size     = 0;
	uint64_t    seq      = 0;
	bool        enhanced = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenoisedFrameMsg, v, type, shm, size, seq, enhanced)

// ---------------------------------------------------------------------------
// Conversation — "llm_response" topic (schema v2)
// ---------------------------------------------------------------------------
struct ConversationResponseMsg {
	int         v                 = 2;
	std::string type              = std::string(kZmqTopicConversationResponse);
	std::string token;
	bool        finished          = false;
	bool        sentence_boundary = false;
	int         sequence_id       = 0;
	bool        has_audio         = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ConversationResponseMsg,
	v, type, token, finished, sentence_boundary, sequence_id, has_audio)

// ---------------------------------------------------------------------------
// Beauty — "beauty_frame" topic
// ---------------------------------------------------------------------------
struct BeautyFrameMsg {
	int         v       = kSchemaVersion;
	std::string type    = "beauty_frame";
	std::string shm;
	int64_t     size    = 0;
	uint64_t    seq     = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BeautyFrameMsg, v, type, shm, size, seq)
