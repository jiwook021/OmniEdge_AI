#include "cv/face_recog_node.hpp"

#include "gpu/cuda_guard.hpp"
#include "shm/shm_frame_reader.hpp"
#include "common/oe_shm_helpers.hpp"
#include "common/oe_tracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <thread>

#include <nlohmann/json.hpp>
#include <sqlite3.h>


// --- Construction / Destruction ---

FaceRecognitionNode::FaceRecognitionNode(const Config& config)
	: config_(config),
	  messageRouter_(MessageRouter::Config{
		  config.moduleName,
		  config.pubPort,
		  config.zmqSendHighWaterMark,
		  config.pollTimeout})
{
}

FaceRecognitionNode::~FaceRecognitionNode()
{
	stop();
}

// --- Config::validate ---

tl::expected<FaceRecognitionNode::Config, std::string>
FaceRecognitionNode::Config::validate(const Config& raw)
{
	ConfigValidator v;
	v.requirePort("pubPort", raw.pubPort);
	v.requirePort("videoSubPort", raw.videoSubPort);
	v.requirePort("wsBridgeSubPort", raw.wsBridgeSubPort);
	v.requireNonEmpty("modelPackPath", raw.modelPackPath);
	v.requireRangeF("recognitionThreshold", raw.recognitionThreshold, 0.0f, 1.0f);
	v.requirePositive("frameSubsample", static_cast<int>(raw.frameSubsample));

	if (auto err = v.finish(); !err.empty()) {
		return tl::unexpected(err);
	}
	return raw;
}

// --- configureTransport() ---

tl::expected<void, std::string> FaceRecognitionNode::configureTransport()
{
	OE_ZONE_SCOPED;
	OeLogger::instance().setModule(config_.moduleName);

	// Input SHM: /oe.vid.ingest (consumer — retry up to 10 times waiting for producer)
	const std::size_t inShmSize =
		ShmCircularBuffer<ShmVideoHeader>::segmentSize(
			kCircularBufferSlotCount, kMaxBgr24FrameBytes);
	auto shmResult = oe::shm::openConsumerWithRetry(
		config_.inputShmName, inShmSize, "face_recog");
	if (!shmResult) return tl::unexpected(shmResult.error());
	shmIn_ = std::move(*shmResult);

	// SQLite — load known faces into memory cache
	loadFacesFromDb();

	// ZMQ subscriptions via MessageRouter
	messageRouter_.subscribe(config_.videoSubPort, "video_frame", /*conflate=*/true,
		[this](const nlohmann::json& msg) {
			++frameCounter_;
			if (frameCounter_ % config_.frameSubsample == 0) {
				processVideoFrame(msg);
			}
		});
	messageRouter_.subscribe(config_.wsBridgeSubPort, "ui_command", /*conflate=*/false,
		[this](const nlohmann::json& msg) { handleUiCommand(msg); });

	OE_LOG_INFO("face_recog_configured: pub={}, model={}, faces={}, threshold={}, subsample={}",
		config_.pubPort, config_.modelPackPath,
		static_cast<int>(knownFaces_.size()),
		config_.recognitionThreshold, config_.frameSubsample);

	return {};
}

// --- loadInferencer() ---

tl::expected<void, std::string> FaceRecognitionNode::loadInferencer()
{
	OE_ZONE_SCOPED;

	if (!inferencer_) {
		return tl::unexpected(std::string(
			"No inferencer — call setInferencer() before initialize()"));
	}

	if (!config_.modelPackPath.empty()
	    && !std::filesystem::exists(config_.modelPackPath)) {
		return tl::unexpected(std::format(
			"Model pack not found: {}", config_.modelPackPath));
	}

	try {
		inferencer_->loadModel(config_.modelPackPath);
	} catch (const std::exception& e) {
		return tl::unexpected(std::format(
			"Face recognition inferencer load failed: {}", e.what()));
	}

	OE_LOG_INFO("face_recog_inferencer_loaded: model={}", config_.modelPackPath);

	return {};
}

// --- registerFace() ---
//
// Detect → extract embedding → store in SQLite + memory cache → publish result.

bool FaceRecognitionNode::registerFace(const std::string& name,
                                        const uint8_t*     bgrData,
                                        uint32_t           width,
                                        uint32_t           height)
{
	if (auto check = ensureVramAvailableMiB(kFaceRecogInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("fr_vram_preflight_failed: {}", check.error());
		publishFaceRegistered(name, false, "insufficient VRAM");
		return false;
	}
	auto result = inferencer_->detect(bgrData, width, height);
	if (!result || result.value().empty()) {
		OE_LOG_WARN("register_face_no_detection: name={}", name);
		publishFaceRegistered(name, false, "no face detected");
		return false;
	}

	const std::vector<float>& embedding = result.value().front().embedding;
	if (embedding.empty()) {
		publishFaceRegistered(name, false, "empty embedding");
		return false;
	}

	insertFaceToDb(name, embedding);
	knownFaces_[name] = embedding;

	OE_LOG_INFO("face_registered: name={}", name);
	publishFaceRegistered(name, true);
	return true;
}

// --- cosineSimilarity() — static utility ---

float FaceRecognitionNode::cosineSimilarity(
	const std::vector<float>& a,
	const std::vector<float>& b) noexcept
{
	if (a.empty() || a.size() != b.size()) return 0.0f;

	float dot = 0.0f, normA = 0.0f, normB = 0.0f;
	for (std::size_t i = 0; i < a.size(); ++i) {
		dot   += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	const float denom = std::sqrt(normA) * std::sqrt(normB);
	return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

// --- processVideoFrame() ---
//
// Read BGR frame from SHM → detect largest face → identify against gallery
// → publish identity if confidence exceeds threshold.

void FaceRecognitionNode::processVideoFrame(const nlohmann::json& /*meta*/)
{
	OE_ZONE_SCOPED;
	// 1. Read latest BGR24 frame from input SHM
	const auto frame = readLatestBgrFrame(*shmIn_);
	if (!frame.data) return;

	// 2. VRAM pre-flight + detect + extract embedding
	if (auto check = ensureVramAvailableMiB(kFaceRecogInferenceHeadroomMiB); !check) {
		OE_LOG_WARN("fr_vram_preflight_failed: {}", check.error());
		return;
	}
	auto result = inferencer_->detect(frame.data, frame.width, frame.height);
	if (!result) {
		OE_LOG_WARN("fr_detect_error: error={}", result.error());
		return;
	}

	// 3. Stale-read guard — discard if input was overwritten during detection
	if (currentShmSlotIndex(*shmIn_) != frame.slotIndex) {
		OE_LOG_DEBUG("fr_stale_read: slot changed during detection");
		return;
	}

	if (result.value().empty()) return;

	// 4. Identify against gallery
	const FaceDetection& det = result.value().front();
	auto [name, confidence] = identify(det.embedding);

	OE_LOG_DEBUG("fr_identify_result: name={}, confidence={:.3f}, threshold={}, "
	             "vram_used={}MiB, host_rss={}KiB",
	           name, confidence, config_.recognitionThreshold,
	           queryVramMiB().usedMiB, hostRssKiB());

	if (confidence >= config_.recognitionThreshold) {
		publishIdentity(name, confidence, det.bbox, det.landmarks);
	}
}

// --- handleUiCommand() ---

void FaceRecognitionNode::handleUiCommand(const nlohmann::json& cmd)
{
	if (cmd.value("action", std::string{}) != "register_face") return;

	const std::string name = cmd.value("name", std::string{});
	if (name.empty()) {
		OE_LOG_WARN("register_face_missing_name");
		publishFaceRegistered("", false, "name field required");
		return;
	}

	// Read the current video frame as the registration source
	const auto frame = readLatestBgrFrame(*shmIn_);
	if (!frame.data) {
		publishFaceRegistered(name, false, "no video frame available");
		return;
	}

	registerFace(name, frame.data, frame.width, frame.height);
}

// --- identify() ---

std::pair<std::string, float>
FaceRecognitionNode::identify(const std::vector<float>& embedding)
{
	std::string bestName;
	float       bestScore = -1.0f;

	for (const auto& [name, known] : knownFaces_) {
		const float score = cosineSimilarity(embedding, known);
		if (score > bestScore) {
			bestScore = score;
			bestName  = name;
		}
	}
	return {bestName, bestScore};
}

// --- ZMQ publish helpers ---

void FaceRecognitionNode::publishIdentity(const std::string&    name,
                                           float                 confidence,
                                           const FaceBBox&       bbox,
                                           const FaceLandmarks&  landmarks)
{
	nlohmann::json lm = nlohmann::json::array();
	for (int i = 0; i < 5; ++i) {
		lm.push_back({landmarks.pts[i][0], landmarks.pts[i][1]});
	}

	nlohmann::json payload = {
		{"v",          kSchemaVersion},
		{"type",       "identity"},
		{"name",       name},
		{"confidence", confidence},
		{"bbox",       {bbox.x, bbox.y, bbox.w, bbox.h}},
		{"landmarks",  lm},
	};
	messageRouter_.publish("identity", std::move(payload));
}

void FaceRecognitionNode::publishFaceRegistered(const std::string& name,
                                                 bool               success,
                                                 const std::string& errorMsg)
{
	nlohmann::json payload = {
		{"v",       kSchemaVersion},
		{"type",    "face_registered"},
		{"name",    name},
		{"success", success},
	};
	if (!errorMsg.empty()) {
		payload["error"] = errorMsg;
	}
	messageRouter_.publish("face_registered", std::move(payload));
}

// --- SQLite persistence ---

void FaceRecognitionNode::loadFacesFromDb()
{
	sqlite3* db = nullptr;
	if (sqlite3_open(config_.knownFacesDb.c_str(), &db) != SQLITE_OK) {
		OE_LOG_INFO("known_faces_db_not_found: path={}", config_.knownFacesDb);
		if (db) sqlite3_close(db);
		return;
	}

	// Create table if first run
	const char* createSql =
		"CREATE TABLE IF NOT EXISTS known_faces "
		"(id INTEGER PRIMARY KEY, name TEXT NOT NULL, embedding BLOB NOT NULL);";
	char* errMsg = nullptr;
	const int createRc = sqlite3_exec(db, createSql, nullptr, nullptr, &errMsg);
	if (createRc != SQLITE_OK) {
		OE_LOG_WARN("face_db_create_table_failed: rc={}, err={}",
		            createRc, errMsg ? errMsg : "unknown");
		if (errMsg) sqlite3_free(errMsg);
		sqlite3_close(db);
		return;
	}
	if (errMsg) sqlite3_free(errMsg);

	// Load all embeddings into memory
	sqlite3_stmt* stmt = nullptr;
	const char* selectSql = "SELECT name, embedding FROM known_faces;";
	if (sqlite3_prepare_v2(db, selectSql, -1, &stmt, nullptr) == SQLITE_OK) {
		while (sqlite3_step(stmt) == SQLITE_ROW) {
			const char* name = reinterpret_cast<const char*>(
				sqlite3_column_text(stmt, 0));
			const auto* blob = reinterpret_cast<const float*>(
				sqlite3_column_blob(stmt, 1));
			const int blobBytes = sqlite3_column_bytes(stmt, 1);
			const auto dims =
				static_cast<std::size_t>(blobBytes) / sizeof(float);
			if (name && blob && dims > 0) {
				knownFaces_[name] = std::vector<float>(blob, blob + dims);
			}
		}
		sqlite3_finalize(stmt);
	}
	sqlite3_close(db);

	OE_LOG_INFO("known_faces_loaded: count={}",
		    static_cast<int>(knownFaces_.size()));
}

void FaceRecognitionNode::insertFaceToDb(const std::string&         name,
                                          const std::vector<float>&  embedding)
{
	sqlite3* db = nullptr;
	if (sqlite3_open(config_.knownFacesDb.c_str(), &db) != SQLITE_OK) {
		OE_LOG_WARN("face_db_open_failed: path={}", config_.knownFacesDb);
		return;
	}

	const char* createSql =
		"CREATE TABLE IF NOT EXISTS known_faces "
		"(id INTEGER PRIMARY KEY, name TEXT NOT NULL, embedding BLOB NOT NULL);";
	char* createErr = nullptr;
	const int createRc = sqlite3_exec(db, createSql, nullptr, nullptr, &createErr);
	if (createRc != SQLITE_OK) {
		OE_LOG_WARN("face_db_create_table_failed: rc={}, err={}",
		            createRc, createErr ? createErr : "unknown");
		if (createErr) sqlite3_free(createErr);
		sqlite3_close(db);
		return;
	}
	if (createErr) sqlite3_free(createErr);

	sqlite3_stmt* stmt = nullptr;
	const char* insertSql =
		"INSERT OR REPLACE INTO known_faces (name, embedding) VALUES (?, ?);";
	if (sqlite3_prepare_v2(db, insertSql, -1, &stmt, nullptr) == SQLITE_OK) {
		sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
		sqlite3_bind_blob(stmt, 2,
		                  embedding.data(),
		                  static_cast<int>(embedding.size() * sizeof(float)),
		                  SQLITE_TRANSIENT);
		const int rc = sqlite3_step(stmt);
		if (rc != SQLITE_DONE) {
			OE_LOG_WARN("face_db_insert_failed: name={}, rc={}", name, rc);
		}
		sqlite3_finalize(stmt);
	} else {
		OE_LOG_WARN("face_db_prepare_failed: {}", sqlite3_errmsg(db));
	}
	sqlite3_close(db);
}

