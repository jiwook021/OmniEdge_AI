#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "zmq/message_router.hpp"
#include "common/module_node_base.hpp"
#include "common/validated_config.hpp"
#include "common/runtime_defaults.hpp"
#include <tl/expected.hpp>
#include "zmq/port_settings.hpp"
#include "gpu/cuda_priority.hpp"
#include "vram/vram_thresholds.hpp"
#include "zmq/audio_constants.hpp"
#include "zmq/zmq_constants.hpp"
#include "shm/shm_mapping.hpp"
#include "common/shm_tts_header.hpp"
#include "tts_inferencer.hpp"
#include "tts/sentence_splitter.hpp"

// ---------------------------------------------------------------------------
// OmniEdge_AI — TTSNode
//
// Consumes llm_response token stream from ConversationNode, accumulates tokens
// until a sentence boundary is detected, synthesises each sentence via
// OnnxKokoroInferencer, writes the resulting PCM to /oe.aud.tts SHM, and
// publishes a tts_audio ZMQ notification for WebSocketBridge.
//
// IPC contracts:
//   ZMQ SUB (port 5572) — "llm_response" from ConversationNode
//   ZMQ SUB (port 5571) — "module_status" from OmniEdgeDaemon
//                         Handles flush_tts when daemon interrupts TTS.
//   ZMQ PUB (port 5565) — "tts_audio"
//   SHM producer        — /oe.aud.tts (PCM F32 24 kHz mono)
//
// Hot path:
//   llm_response → SentenceSplitter → TTSInferencer::synthesize()
//   → SHM write (ShmTtsHeader + PCM) → ZMQ "tts_audio"
//
// Sentence streaming:
//   Synthesis begins sentence-by-sentence before the LLM finishes, enabling
//   audio playback to start ~100 ms after the first sentence is complete.
//   The LLM also sets sentence_boundary:true on the first token of each new
//   sentence; this is used as a secondary flush trigger.
//


// ShmTtsHeader is defined in common/shm_tts_header.hpp (included above).
// WebSocketBridge reads numSamples from this header (or from the ZMQ
// tts_audio message which carries the same value).

/// Maximum float32 samples in one TTS SHM segment.
/// 10 seconds × 24,000 Hz = 240,000 samples.
/// Covers the longest expected single sentence with wide margin.
inline constexpr std::size_t kTtsShmMaxSamples = 240'000;

/// Total /oe.aud.tts SHM segment size in bytes.
inline constexpr std::size_t kTtsShmBytes =
	ShmTtsHeader::segmentSize(kTtsShmMaxSamples);

// ---------------------------------------------------------------------------
// TTSNode
// ---------------------------------------------------------------------------

/**
 * @brief TTS pipeline node: token stream → sentence synthesis → SHM + ZMQ.
 *
 * Instantiated once per process.  Inject a custom TTSInferencer for unit tests
 * (pass via the two-argument constructor below).
 */
class TTSNode : public ModuleNodeBase<TTSNode> {
public:
	friend class ModuleNodeBase<TTSNode>;

	/**
	 * @brief Runtime configuration — all fields have YAML-friendly defaults.
	 *
	 * Numeric defaults are drawn from common/oe_defaults.hpp and
	 * the focused constant headers so that no bare literals appear in .cpp files.
	 */
	struct Config {
		// Model paths — Kokoro v1.0 TTS: 82M-param StyleTTS2-derived model.
		// INT8 ONNX variant uses <100 MB VRAM via ONNX Runtime CUDA EP;
		// quality is competitive with models 10× larger. See Architecture § TTS.
		std::string onnxModelPath  = "./models/onnx/kokoro-v1_0-int8.onnx";
		// Pre-computed NumPy voice style embeddings for speaker selection.
		std::string voiceDir       = "./models/kokoro/voices";
		// "af_heart" is the default female English voice style.
		std::string defaultVoice   = "af_heart";

		// Synthesis
		float       speed          = 1.0f;

		// SHM
		std::string shmOutput      = "/oe.aud.tts";

		// ZMQ ports
		int         subLlmPort     = kConversationModel;  // 5572
		int         subDaemonPort  = kDaemon;  // 5571
		int         pubPort        = kTts;     // 5565

		// CUDA
		int         cudaStreamPriority = kCudaPriorityTts;  // -3

		// Polling
		std::chrono::milliseconds pollTimeout{kZmqPollTimeout};

		// Module identity (for logging)
		std::string moduleName     = "tts";

		[[nodiscard]] static tl::expected<Config, std::string> validate(const Config& raw);
	};

	/**
	 * @brief Construct with an injected inferencer.
	 *
	 * Always inject the inferencer explicitly — either a production
	 * OnnxKokoroInferencer (created in main.cpp) or a MockTTSInferencer for tests.
	 * This keeps tts_node.cpp free of ONNX Runtime headers so that the
	 * omniedge_tts_common static library compiles without a GPU toolchain.
	 *
	 * @param config   Runtime configuration.
	 * @param inferencer  Concrete TTS inferencer.  Must not be nullptr.
	 * @throws  Nothing — construction is infallible.
	 */
	TTSNode(const Config& config,
	        std::unique_ptr<TTSInferencer> inferencer);

	~TTSNode();

	TTSNode(const TTSNode&)            = delete;
	TTSNode& operator=(const TTSNode&) = delete;

	[[nodiscard]] tl::expected<void, std::string> configureTransport();
	[[nodiscard]] tl::expected<void, std::string> loadInferencer();
	[[nodiscard]] MessageRouter& router() noexcept { return messageRouter_; }
	[[nodiscard]] std::string_view moduleName() const noexcept { return config_.moduleName; }

private:
	// -----------------------------------------------------------------------
	// Message handlers
	// -----------------------------------------------------------------------

	/** Process one llm_response JSON message from ConversationNode. */
	void handleLlmResponse(const nlohmann::json& msg);

	/** Process one module_status / flush_tts message from the daemon. */
	void handleDaemonMessage(const nlohmann::json& msg);

	/**
	 * @brief Synthesise `text`, write PCM to SHM, publish tts_audio.
	 *
	 * Called for each completed sentence from the SentenceSplitter.
	 *
	 * @param text  One complete sentence.
	 */
	void synthesizeAndPublish(const std::string& text);

	// -----------------------------------------------------------------------
	// Members
	// -----------------------------------------------------------------------

	Config                       config_;

	// Inference inferencer
	std::unique_ptr<TTSInferencer> inferencer_;

	// Sentence accumulator
	SentenceSplitter             splitter_;

	// SHM producer — /oe.aud.tts
	std::unique_ptr<ShmMapping>  shmOutput_;

	// Networking — single MessageRouter replaces all ZMQ boilerplate
	MessageRouter                messageRouter_;

	// Sequence counter for SHM header and tts_audio ZMQ messages
	uint64_t seqNumber_{0};
};

