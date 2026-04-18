#include "tts/onnx_kokoro_inferencer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include <nlohmann/json.hpp>

#include "common/file.hpp"
#include "common/oe_logger.hpp"
#include "vram/vram_thresholds.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Built-in English IPA → Kokoro token ID vocabulary
//
// Based on the Kokoro v1.0 phoneme vocabulary (hexgrad/Kokoro-82M).
// Each entry maps a UTF-32 codepoint string to its token ID.
// ID 0 is the padding token; 1 is the word-boundary marker.
// IDs 2+ cover IPA vowels, consonants, stress markers, and length marks.
//
// The built-in table covers American English IPA as produced by
// espeak-ng --ipa with voice "en-us". Developers targeting other languages
// should supply a companion phoneme_vocab.json in the model directory.
// ---------------------------------------------------------------------------

// clang-format off
static const std::pair<const char*, int64_t> kBuiltinVocab[] = {
	// Word boundary and silence
	{ " ",   1 },   // inter-word space

	// Vowels — monophthongs
	{ "i",   2 },   // FLEECE   — iː in British
	{ "ɪ",   3 },   // KIT
	{ "e",   4 },   // FACE (monophthong variant)
	{ "ɛ",   5 },   // DRESS
	{ "æ",   6 },   // TRAP
	{ "ɑ",   7 },   // START / LOT (American)
	{ "ɒ",   8 },   // LOT (British)
	{ "ɔ",   9 },   // THOUGHT
	{ "o",  10 },   // GOAT (monophthong)
	{ "ʊ",  11 },   // FOOT
	{ "u",  12 },   // GOOSE
	{ "ʌ",  13 },   // STRUT
	{ "ə",  14 },   // schwa
	{ "ɐ",  15 },   // near-open central (reduced)
	{ "ɜ",  16 },   // NURSE
	{ "ɝ",  17 },   // NURSE (rhotic)
	{ "ɚ",  18 },   // schwa (rhotic)

	// Vowels — diphthongs (treated as two-codepoint keys where possible)
	{ "eɪ", 19 },   // FACE
	{ "aɪ", 20 },   // PRICE
	{ "ɔɪ", 21 },   // CHOICE
	{ "oʊ", 22 },   // GOAT
	{ "aʊ", 23 },   // MOUTH
	{ "ɪə", 24 },   // NEAR
	{ "ɛə", 25 },   // SQUARE
	{ "ʊə", 26 },   // CURE

	// Consonants — plosives
	{ "p",  27 },
	{ "b",  28 },
	{ "t",  29 },
	{ "d",  30 },
	{ "k",  31 },
	{ "ɡ",  32 },   // voiced velar plosive (espeak outputs ɡ U+0261)
	{ "g",  32 },   // ASCII g alias

	// Consonants — affricates
	{ "tʃ", 33 },   // CHURCH
	{ "dʒ", 34 },   // JUDGE

	// Consonants — fricatives
	{ "f",  35 },
	{ "v",  36 },
	{ "θ",  37 },   // THIN
	{ "ð",  38 },   // THIS
	{ "s",  39 },
	{ "z",  40 },
	{ "ʃ",  41 },   // SHIP
	{ "ʒ",  42 },   // MEASURE
	{ "h",  43 },

	// Consonants — nasals
	{ "m",  44 },
	{ "n",  45 },
	{ "ŋ",  46 },   // SING

	// Consonants — approximants
	{ "l",  47 },
	{ "r",  48 },   // ASCII r (General American)
	{ "ɹ",  49 },   // alveolar approximant
	{ "j",  50 },   // YET
	{ "w",  51 },

	// Consonants — lateral approximant
	{ "ɫ",  52 },   // dark l

	// Syllabic consonants
	{ "l̩",  53 },   // syllabic l
	{ "n̩",  54 },   // syllabic n
	{ "m̩",  55 },   // syllabic m

	// Prosodic markers
	{ "ˈ",  56 },   // primary stress (IPA U+02C8)
	{ "ˌ",  57 },   // secondary stress (IPA U+02CC)
	{ "ː",  58 },   // length mark (IPA U+02D0)
	{ ".",  59 },   // syllable boundary
	{ "|",  60 },   // minor phrase boundary (espeak uses | )
	{ "‖",  61 },   // major phrase boundary

	// Glottal
	{ "ʔ",  62 },   // glottal stop
};
// clang-format on

// ---------------------------------------------------------------------------
// espeak-ng global singleton reference count
// ---------------------------------------------------------------------------
static std::atomic<int> espeakRefCount{0};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

OnnxKokoroInferencer::OnnxKokoroInferencer()
	: ortEnv_(ORT_LOGGING_LEVEL_WARNING, "omniedge_tts")
{
}

OnnxKokoroInferencer::~OnnxKokoroInferencer()
{
	unloadModel();
}

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------

tl::expected<void, std::string> OnnxKokoroInferencer::loadModel(
	const std::string& onnxModelPath,
	const std::string& voiceDir)
{
	if (modelLoaded_) {
		unloadModel();
	}

	// Reset session options so CUDA EPs do not accumulate across reloads.
	sessionOpts_ = Ort::SessionOptions{};

	// ── 1. Validate paths ────────────────────────────────────────────────
	if (!isFile(onnxModelPath)) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] ONNX model not found: {}", onnxModelPath));
	}
	if (!isDirectory(voiceDir)) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] Voice directory not found: {}", voiceDir));
	}
	voiceDir_ = voiceDir;

	// ── 2. Build ORT session options with CUDA EP ─────────────────────────
	sessionOpts_.SetIntraOpNumThreads(1);
	sessionOpts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

	// Cache the optimised graph so CUDA EP kernel compilation is a one-time
	// cost.  Subsequent loads read the .ort file instead of re-compiling.
	const auto optimizedPath =
		std::filesystem::path(onnxModelPath).parent_path() / "kokoro-v1_0-int8.optimized.ort";
	sessionOpts_.SetOptimizedModelFilePath(optimizedPath.c_str());

	// CUDA EP required — no CPU fallback
	OrtCUDAProviderOptions cudaOpts{};
	cudaOpts.device_id                 = 0;
	cudaOpts.arena_extend_strategy     = 0;     // kNextPowerOfTwo — larger coherent chunks so Kokoro's per-sentence allocs don't fragment the arena
	cudaOpts.gpu_mem_limit             = static_cast<size_t>(
		kTtsMiB) * 1024 * 1024;
	cudaOpts.do_copy_in_default_stream = 0;     // keep H2D/D2H on the CUDA EP's non-default stream
	cudaOpts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchHeuristic;

	try {
		sessionOpts_.AppendExecutionProvider_CUDA(cudaOpts);
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] CUDA EP registration failed (is CUDA available?): {}",
			ex.what()));
	}
	cudaEpActive_ = true;
	OE_LOG_INFO("tts_cuda_ep_enabled: device_id=0, vram_limit_mb={}, "
		"cudnn_algo=heuristic",
		kTtsMiB);

	// ── 3. Create inference session on CUDA ────────────────────────────────
	try {
		session_ = std::make_unique<Ort::Session>(
			ortEnv_,
			onnxModelPath.c_str(),
			sessionOpts_);
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] CUDA session creation failed: {}", ex.what()));
	}

	// ── 4. Load phoneme vocabulary ─────────────────────────────────────────
	const std::string modelDir =
		std::filesystem::path(onnxModelPath).parent_path().string();

	if (!loadVocabJson(modelDir)) {
		// Companion JSON absent — use built-in English IPA table.
		loadBuiltinVocab();
		OE_LOG_INFO("tts_phoneme_vocab_builtin: entries={}",
			phonemeVocab_.size());
	}

	// ── 5. Initialise espeak-ng ────────────────────────────────────────────
	// espeak-ng is a global singleton — reference-count so multiple
	// OnnxKokoroInferencer instances (or reload cycles) don't double-init
	// or early-terminate.  (See file-scope espeakRefCount above.)
	if (espeakRefCount.fetch_add(1) == 0) {
		// Honour ESPEAK_DATA_PATH env var so extracted-deb installs work
		// without requiring a system-wide espeak-ng-data package.
		const char* espeakDataPath = std::getenv("ESPEAK_DATA_PATH");
		const int espeakVersion = espeak_Initialize(
			AUDIO_OUTPUT_RETRIEVAL,
			/*buflength=*/0,
			/*path=*/espeakDataPath,
			/*options=*/0);
		if (espeakVersion < 0) {
			espeakRefCount.fetch_sub(1);
			return tl::unexpected("[OnnxKokoro] espeak_Initialize failed — "
				"is libespeak-ng installed?");
		}
	}

	// Default voice: American English.
	espeak_ERROR esErr = espeak_SetVoiceByName("en-us");
	if (esErr != EE_OK) {
		// Non-fatal — espeak will use its built-in default.
		OE_LOG_WARN("tts_espeak_voice_fallback: requested=en-us, error_code={}",
			static_cast<int>(esErr));
	}

	modelLoaded_ = true;
	vramUsageBytes_ = kTtsMiB * 1024 * 1024;

	OE_LOG_INFO("tts_model_loaded: path={}, voice_dir={}, cuda_ep=true, vram_mb={}",
		onnxModelPath, voiceDir, kTtsMiB);

	return {};
}

// ---------------------------------------------------------------------------
// unloadModel
// ---------------------------------------------------------------------------

void OnnxKokoroInferencer::unloadModel() noexcept
{
	// Terminate espeak-ng before releasing other resources to avoid
	// dangling internal state (espeak allocates global buffers).
	// Reference-counted: only call espeak_Terminate when this is the last
	// user of the global singleton.
	if (modelLoaded_) {
		if (espeakRefCount.fetch_sub(1) == 1) {
			espeak_Terminate();
		}
	}

	session_.reset();
	voiceCache_.clear();
	phonemeVocab_.clear();
	modelLoaded_ = false;
	vramUsageBytes_   = 0;
	cudaEpActive_ = false;
}

// ---------------------------------------------------------------------------
// currentVramUsageBytes
// ---------------------------------------------------------------------------

std::size_t OnnxKokoroInferencer::currentVramUsageBytes() const
{
	return vramUsageBytes_;
}

// ---------------------------------------------------------------------------
// synthesize
// ---------------------------------------------------------------------------

tl::expected<std::vector<float>, std::string> OnnxKokoroInferencer::synthesize(
	std::string_view text,
	std::string_view voiceName,
	float            speed)
{
	if (!modelLoaded_) {
		return tl::unexpected("[OnnxKokoro] Model not loaded — call loadModel() first.");
	}

	if (text.empty()) {
		return tl::unexpected("[OnnxKokoro] synthesize() called with empty text.");
	}

	// ── 1. G2P: text → phoneme token IDs ────────────────────────────────
	OE_LOG_INFO("tts_synthesize_start: text_len={}", text.size());
	auto tokensResult = textToPhonemeTokens(text);
	if (!tokensResult) {
		return tl::unexpected(tokensResult.error());
	}
	const std::vector<int64_t>& tokens = *tokensResult;

	if (tokens.empty()) {
		return tl::unexpected("[OnnxKokoro] G2P produced zero tokens for input text.");
	}

	OE_LOG_INFO("tts_g2p_done: tokens={}, voice={}, speed={}",
	           tokens.size(), voiceName, speed);

	// ── 2. Retrieve voice style tensor ───────────────────────────────────
	auto styleResult = getVoiceStyle(voiceName, tokens.size());
	if (!styleResult) {
		return tl::unexpected(styleResult.error());
	}
	const std::span<const float> voiceStyle = *styleResult;

	// ── 3. Build ONNX input tensors ───────────────────────────────────────

	// Input 0 — phoneme tokens: int64_t [1, token_len]
	const std::array<int64_t, 2> tokShape{1, static_cast<int64_t>(tokens.size())};
	Ort::Value tokTensor = Ort::Value::CreateTensor<int64_t>(
		memInfo_,
		const_cast<int64_t*>(tokens.data()),
		tokens.size(),
		tokShape.data(),
		tokShape.size());

	// Input 1 — voice style: float32 [1, 256]
	const std::array<int64_t, 2> styleShape{1, 256};
	Ort::Value styleTensor = Ort::Value::CreateTensor<float>(
		memInfo_,
		const_cast<float*>(voiceStyle.data()),
		voiceStyle.size(),
		styleShape.data(),
		styleShape.size());

	// Input 2 — speed: float32 [1]
	const std::array<int64_t, 1> speedShape{1};
	float speedVal = speed;
	Ort::Value speedTensor = Ort::Value::CreateTensor<float>(
		memInfo_,
		&speedVal,
		1,
		speedShape.data(),
		speedShape.size());

	// ── 4. Run inference ──────────────────────────────────────────────────
	constexpr std::array<const char*, 3> kInputNames{
		"input_ids", "style", "speed"};
	constexpr std::array<const char*, 1> kOutputNames{
		"waveform"};

	std::vector<Ort::Value> inputs;
	inputs.reserve(3);
	inputs.push_back(std::move(tokTensor));
	inputs.push_back(std::move(styleTensor));
	inputs.push_back(std::move(speedTensor));

	OE_LOG_INFO("tts_ort_run_start: inputs=3, cuda_ep={}",
		cudaEpActive_ ? "true" : "false");

	std::vector<Ort::Value> outputs;
	try {
		outputs = session_->Run(
			Ort::RunOptions{nullptr},
			kInputNames.data(),
			inputs.data(),
			inputs.size(),
			kOutputNames.data(),
			kOutputNames.size());
	} catch (const Ort::Exception& ex) {
		return tl::unexpected(
			std::format("[OnnxKokoro] Run() failed: {}", ex.what()));
	}

	// ── 5. Extract PCM from output tensor: float32 [1, num_samples] ──────
	const auto& audioTensor = outputs[0];
	const auto* audioData   = audioTensor.GetTensorData<float>();
	const auto  audioShape  = audioTensor.GetTensorTypeAndShapeInfo().GetShape();

	if (audioShape.size() < 2 || audioShape[1] <= 0) {
		return tl::unexpected("[OnnxKokoro] Unexpected audio tensor shape.");
	}

	const std::size_t numSamples = static_cast<std::size_t>(audioShape[1]);
	OE_LOG_INFO("tts_ort_run_done: output_samples={}", numSamples);
	std::vector<float> pcm(audioData, audioData + numSamples);

	// Clamp to valid PCM range — ONNX model may produce out-of-range values
	// that cause clipping artifacts or undefined behavior in downstream DACs.
	for (auto& sample : pcm) sample = std::clamp(sample, -1.0f, 1.0f);

	return pcm;
}

// ---------------------------------------------------------------------------
// textToPhonemeTokens (private)
// ---------------------------------------------------------------------------

tl::expected<std::vector<int64_t>, std::string>
OnnxKokoroInferencer::textToPhonemeTokens(std::string_view text) const
{
	// espeak_TextToPhonemes() is stateful and iterates word-by-word.
	// It expects a null-terminated C string, so copy the string_view.
	const std::string textCopy(text);
	const void* textPtr = textCopy.c_str();
	const auto* textEnd = textCopy.c_str() + textCopy.size();

	std::string ipaBuffer;
	ipaBuffer.reserve(text.size() * 2);  // IPA is often 2× the Latin length

	while (textPtr != nullptr
		&& static_cast<const char*>(textPtr) < textEnd) {
		// phonememode = espeakPHONEMES_IPA (= 0x0100 in espeak-ng headers)
		// Returns IPA for the next word and advances textPtr.
		const char* phonemes = espeak_TextToPhonemes(
			&textPtr,
			espeakCHARS_UTF8,
			espeakPHONEMES_IPA);

		if (phonemes == nullptr || *phonemes == '\0') {
			continue;
		}

		// Append word phonemes + inter-word space separator.
		ipaBuffer += phonemes;
		ipaBuffer += ' ';
	}

	// ── Map IPA UTF-8 → token IDs via phonemeVocab_ ───────────────────────
	// Convert the whole IPA string to UTF-32 for stable codepoint iteration.
	const std::u32string ipaU32 = utf8ToUtf32(ipaBuffer);

	std::vector<int64_t> tokenIds;
	tokenIds.reserve(ipaU32.size());

	std::size_t pos = 0;
	while (pos < ipaU32.size()) {
		// Try two-codepoint digraph first (common for diphthongs and affricates).
		if (pos + 1 < ipaU32.size()) {
			const std::u32string digraph = ipaU32.substr(pos, 2);
			auto it = phonemeVocab_.find(digraph);
			if (it != phonemeVocab_.end()) {
				tokenIds.push_back(it->second);
				pos += 2;
				continue;
			}
		}

		// Single codepoint lookup.
		const std::u32string single = ipaU32.substr(pos, 1);
		auto it = phonemeVocab_.find(single);
		if (it != phonemeVocab_.end()) {
			tokenIds.push_back(it->second);
		} else {
			// Unknown codepoint — skip silently.
		}
		++pos;
	}

	if (tokenIds.empty()) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] G2P produced no known tokens for: \"{}\"",
			std::string(text)));
	}

	return tokenIds;
}

// ---------------------------------------------------------------------------
// getVoiceStyle (private)
// ---------------------------------------------------------------------------

tl::expected<std::span<const float>, std::string>
OnnxKokoroInferencer::getVoiceStyle(std::string_view voiceName,
	std::size_t tokenCount)
{
	const std::string name(voiceName);

	// ── Load voice pack into cache if not present ───────────────────────
	if (voiceCache_.find(name) == voiceCache_.end()) {
		// Try .bin first (raw float32), then .npy (NumPy format).
		const auto binPath =
			std::filesystem::path(voiceDir_) / (name + ".bin");
		const auto npyPath =
			std::filesystem::path(voiceDir_) / (name + ".npy");

		std::vector<float> pack;
		if (isFile(binPath)) {
			auto result = loadBinFloat32(binPath.string());
			if (!result) return tl::unexpected(result.error());
			pack = std::move(*result);
		} else {
			auto result = loadNpy(npyPath.string());
			if (!result) return tl::unexpected(result.error());
			pack = std::move(*result);
		}

		if (pack.size() % kStyleDim != 0 || pack.empty()) {
			return tl::unexpected(std::format(
				"[OnnxKokoro] Voice pack '{}' has {} floats; "
				"expected a multiple of {}.",
				name, pack.size(), kStyleDim));
		}

		OE_LOG_INFO("tts_voice_loaded: voice={}, rows={}, dim={}",
			name, pack.size() / kStyleDim, kStyleDim);

		voiceCache_.emplace(name, std::move(pack));
	}

	// ── Select the style row for this token count ───────────────────────
	const auto& pack = voiceCache_.at(name);
	const std::size_t numRows = pack.size() / kStyleDim;
	const std::size_t row = std::min(tokenCount, numRows - 1);

	return std::span<const float>(pack.data() + row * kStyleDim, kStyleDim);
}

// ---------------------------------------------------------------------------
// loadNpy (private, static)
//
// Supports NumPy format v1.0 and v2.0 with C-order float32 ('float32'/'<f4')
// flat or [1, 256] shaped arrays.
// ---------------------------------------------------------------------------

tl::expected<std::vector<float>, std::string>
OnnxKokoroInferencer::loadNpy(const std::string& path)
{
	auto rawResult = readBinary(path);
	if (!rawResult) {
		return tl::unexpected(
			std::format("[OnnxKokoro] Cannot open .npy file: {}", rawResult.error()));
	}
	const auto& raw = *rawResult;

	// ── Magic ───────────────────────────────────────────────────────────────
	// \x93NUMPY (6 bytes) + major + minor = 8 bytes minimum header
	constexpr std::size_t kMagicLen = 6;
	if (raw.size() < kMagicLen + 2) {
		return tl::unexpected(
			std::format("[OnnxKokoro] File too small for .npy header: {}", path));
	}
	if (raw[0] != 0x93 ||
		raw[1] != 'N' || raw[2] != 'U' ||
		raw[3] != 'M' || raw[4] != 'P' || raw[5] != 'Y') {
		return tl::unexpected(
			std::format("[OnnxKokoro] Not a .npy file: {}", path));
	}

	// ── Version ─────────────────────────────────────────────────────────────
	const uint8_t majorVer = raw[6];
	const uint8_t minorVer = raw[7];
	(void)minorVer;

	// ── Header length ────────────────────────────────────────────────────────
	std::size_t headerLen{0};
	std::size_t offset = 8;
	if (majorVer == 1) {
		if (raw.size() < offset + 2)
			return tl::unexpected(std::format("[OnnxKokoro] Truncated v1 header: {}", path));
		// NumPy stores headerLen in little-endian; WSL2 / x86 are LE.
		headerLen = static_cast<std::size_t>(raw[offset]) |
		            (static_cast<std::size_t>(raw[offset + 1]) << 8);
		offset += 2;
	} else if (majorVer == 2) {
		if (raw.size() < offset + 4)
			return tl::unexpected(std::format("[OnnxKokoro] Truncated v2 header: {}", path));
		headerLen = static_cast<std::size_t>(raw[offset]) |
		            (static_cast<std::size_t>(raw[offset + 1]) << 8) |
		            (static_cast<std::size_t>(raw[offset + 2]) << 16) |
		            (static_cast<std::size_t>(raw[offset + 3]) << 24);
		offset += 4;
	} else {
		return tl::unexpected(
			std::format("[OnnxKokoro] Unsupported .npy major version {} in: {}",
						static_cast<int>(majorVer), path));
	}

	// ── Header dict ─────────────────────────────────────────────────────────
	if (raw.size() < offset + headerLen)
		return tl::unexpected(std::format("[OnnxKokoro] Truncated header dict: {}", path));

	std::string header(reinterpret_cast<const char*>(raw.data() + offset), headerLen);
	offset += headerLen;

	// Quick dtype check — must contain 'float32' or '<f4'.
	if (header.find("float32") == std::string::npos &&
		header.find("<f4")     == std::string::npos) {
		return tl::unexpected(
			std::format("[OnnxKokoro] .npy file '{}' is not float32 dtype.", path));
	}

	// ── Data payload ─────────────────────────────────────────────────────────
	const std::size_t dataBytes = raw.size() - offset;

	if (dataBytes % sizeof(float) != 0) {
		return tl::unexpected(
			std::format("[OnnxKokoro] .npy data byte count {} not divisible by 4: {}",
						dataBytes, path));
	}

	const std::size_t numFloats = dataBytes / sizeof(float);
	std::vector<float> values(numFloats);
	std::memcpy(values.data(), raw.data() + offset, dataBytes);

	return values;
}

// ---------------------------------------------------------------------------
// loadBuiltinVocab (private)
// ---------------------------------------------------------------------------

void OnnxKokoroInferencer::loadBuiltinVocab()
{
	phonemeVocab_.clear();
	for (const auto& [utf8, id] : kBuiltinVocab) {
		phonemeVocab_.emplace(utf8ToUtf32(utf8), id);
	}
}

// ---------------------------------------------------------------------------
// loadVocabJson (private)
// ---------------------------------------------------------------------------

bool OnnxKokoroInferencer::loadVocabJson(const std::string& modelDir)
{
	const std::string vocabPath =
		(std::filesystem::path(modelDir) / "phoneme_vocab.json").string();

	auto vocabText = readText(vocabPath);
	if (!vocabText) {
		return false;
	}

	try {
		nlohmann::json j = nlohmann::json::parse(*vocabText);

		phonemeVocab_.clear();
		for (auto it = j.begin(); it != j.end(); ++it) {
			phonemeVocab_.emplace(
				utf8ToUtf32(it.key()),
				it.value().get<int64_t>());
		}

		OE_LOG_INFO("tts_phoneme_vocab_json: path={}, entries={}",
			vocabPath, phonemeVocab_.size());
		return true;
	} catch (const nlohmann::json::exception& ex) {
		OE_LOG_WARN("tts_vocab_json_parse_fail: path={}, error={}",
			vocabPath, std::string(ex.what()));
		return false;
	}
}

// ---------------------------------------------------------------------------
// loadBinFloat32 (private, static)
// ---------------------------------------------------------------------------

tl::expected<std::vector<float>, std::string>
OnnxKokoroInferencer::loadBinFloat32(const std::string& path)
{
	auto rawResult = readBinary(path);
	if (!rawResult) {
		return tl::unexpected(
			std::format("[OnnxKokoro] Cannot open .bin file: {}", rawResult.error()));
	}
	const auto& raw = *rawResult;

	if (raw.empty() || raw.size() % sizeof(float) != 0) {
		return tl::unexpected(std::format(
			"[OnnxKokoro] Invalid .bin file size ({}): {}",
			raw.size(), path));
	}

	const std::size_t count = raw.size() / sizeof(float);
	std::vector<float> data(count);
	std::memcpy(data.data(), raw.data(), raw.size());

	return data;
}

// ---------------------------------------------------------------------------
// utf8ToUtf32 (private, static)
//
// Minimal UTF-8 → UTF-32 decoder.  Supports all 4-byte sequences.
// Invalid byte sequences are replaced with U+FFFD.
// ---------------------------------------------------------------------------

std::u32string OnnxKokoroInferencer::utf8ToUtf32(std::string_view utf8)
{
	std::u32string result;
	result.reserve(utf8.size());

	std::size_t i = 0;
	while (i < utf8.size()) {
		const unsigned char c = static_cast<unsigned char>(utf8[i]);
		char32_t codepoint{};
		std::size_t extraBytes{};

		if (c < 0x80) {
			codepoint  = c;
			extraBytes = 0;
		} else if ((c & 0xE0) == 0xC0) {
			codepoint  = c & 0x1F;
			extraBytes = 1;
		} else if ((c & 0xF0) == 0xE0) {
			codepoint  = c & 0x0F;
			extraBytes = 2;
		} else if ((c & 0xF8) == 0xF0) {
			codepoint  = c & 0x07;
			extraBytes = 3;
		} else {
			// Invalid lead byte — replace with U+FFFD.
			result += U'\uFFFD';
			++i;
			continue;
		}

		++i;
		for (std::size_t j = 0; j < extraBytes; ++j) {
			if (i >= utf8.size() ||
				(static_cast<unsigned char>(utf8[i]) & 0xC0) != 0x80) {
				codepoint = U'\uFFFD';
				break;
			}
			codepoint = (codepoint << 6) |
				(static_cast<unsigned char>(utf8[i]) & 0x3F);
			++i;
		}

		result += codepoint;
	}

	return result;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::unique_ptr<TTSInferencer> createOnnxKokoroInferencer()
{
	return std::make_unique<OnnxKokoroInferencer>();
}

// MockTTSInferencer is defined in tests/tts/test_onnx_kokoro_inferencer.cpp
// to avoid linking the production binary against test code.

