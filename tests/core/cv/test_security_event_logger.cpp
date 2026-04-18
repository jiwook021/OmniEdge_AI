#include <gtest/gtest.h>

#include "cv/security_event_logger.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// SecurityEventLogger Tests
//
// Purpose: Verify the append-only JSON Lines event logger that records
// detection events for Security Camera Mode.  From the user's perspective:
//   "Every detection is persisted immediately and survives power loss."
//   "I can query events by time range for VLM analysis."
//   "The log rotates before filling the disk."
//
// What these tests catch:
//   - JSON serialisation produces malformed lines
//   - Time-range query has off-by-one boundary errors
//   - File rotation fails to open a fresh log
//   - Event count drifts out of sync with actual writes
//   - Concurrent appends corrupt interleaved JSON lines
//
// All tests use a temporary directory (auto-cleaned by the fixture).
// No GPU, no network — runs everywhere.
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;
using oe::security::SecurityEvent;
using oe::security::SecurityEventLogger;

// ---------------------------------------------------------------------------
// Fixture — temp directory per test
// ---------------------------------------------------------------------------

class SecurityEventLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmpDir_ = fs::temp_directory_path() / ("oe_test_security_log_" +
                  std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch().count()));
        fs::create_directories(tmpDir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmpDir_, ec);
    }

    /// Create a SecurityEvent with sensible defaults.
    static SecurityEvent makeEvent(const std::string& timestamp,
                                   const std::string& cls = "person",
                                   float confidence = 0.92f) {
        return SecurityEvent{
            .timestamp     = timestamp,
            .detectedClass = cls,
            .confidence    = confidence,
            .bboxX         = 0.1f,
            .bboxY         = 0.2f,
            .bboxW         = 0.3f,
            .bboxH         = 0.4f,
            .recordingFile = "security_00001.mp4",
            .frameSeq      = 42,
        };
    }

    /// Read all lines from the log file.
    std::vector<std::string> readLogLines() const {
        std::vector<std::string> lines;
        std::ifstream in(tmpDir_ / "security_events.jsonl");
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) lines.push_back(line);
        }
        return lines;
    }

    fs::path tmpDir_;
};

// ===========================================================================
// Append — "every detection is persisted"
// ===========================================================================

TEST_F(SecurityEventLoggerTest, AppendEvent_WritesOneJsonLine)
{
    SecurityEventLogger logger(tmpDir_);

    auto result = logger.appendEvent(makeEvent("2026-04-12T14:30:15.000Z"));
    ASSERT_TRUE(result.has_value()) << result.error();

    auto lines = readLogLines();
    ASSERT_EQ(lines.size(), 1u);

    // Verify it's valid JSON with expected fields.
    auto j = nlohmann::json::parse(lines[0]);
    EXPECT_EQ(j.at("class").get<std::string>(), "person");
    EXPECT_EQ(j.at("timestamp").get<std::string>(), "2026-04-12T14:30:15.000Z");
    EXPECT_FLOAT_EQ(j.at("confidence").get<float>(), 0.92f);
    EXPECT_FLOAT_EQ(j.at("bbox").at("x").get<float>(), 0.1f);
    EXPECT_FLOAT_EQ(j.at("bbox").at("y").get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j.at("bbox").at("w").get<float>(), 0.3f);
    EXPECT_FLOAT_EQ(j.at("bbox").at("h").get<float>(), 0.4f);
    EXPECT_EQ(j.at("recording_file").get<std::string>(), "security_00001.mp4");
    EXPECT_EQ(j.at("frame_seq").get<uint64_t>(), 42u);
}

TEST_F(SecurityEventLoggerTest, AppendMultipleEvents_AllPersisted)
{
    SecurityEventLogger logger(tmpDir_);

    for (int i = 0; i < 10; ++i) {
        auto ts = "2026-04-12T14:30:" + std::to_string(10 + i) + ".000Z";
        auto result = logger.appendEvent(makeEvent(ts));
        ASSERT_TRUE(result.has_value()) << "Event " << i << ": " << result.error();
    }

    auto lines = readLogLines();
    EXPECT_EQ(lines.size(), 10u);
}

TEST_F(SecurityEventLoggerTest, EventCount_MatchesAppendCalls)
{
    SecurityEventLogger logger(tmpDir_);
    EXPECT_EQ(logger.eventCount(), 0u);

    ASSERT_TRUE(logger.appendEvent(makeEvent("2026-04-12T14:00:00.000Z")).has_value());
    EXPECT_EQ(logger.eventCount(), 1u);

    ASSERT_TRUE(logger.appendEvent(makeEvent("2026-04-12T14:01:00.000Z")).has_value());
    ASSERT_TRUE(logger.appendEvent(makeEvent("2026-04-12T14:02:00.000Z")).has_value());
    EXPECT_EQ(logger.eventCount(), 3u);
}

// ===========================================================================
// Query — "find events in a time range for VLM analysis"
// ===========================================================================

TEST_F(SecurityEventLoggerTest, QueryEvents_ReturnsMatchingTimeRange)
{
    SecurityEventLogger logger(tmpDir_);

    (void)logger.appendEvent(makeEvent("2026-04-12T10:00:00.000Z", "person"));
    (void)logger.appendEvent(makeEvent("2026-04-12T12:00:00.000Z", "car"));
    (void)logger.appendEvent(makeEvent("2026-04-12T14:00:00.000Z", "truck"));
    (void)logger.appendEvent(makeEvent("2026-04-12T16:00:00.000Z", "bicycle"));

    auto result = logger.queryEvents("2026-04-12T11:00:00.000Z",
                                      "2026-04-12T15:00:00.000Z");
    ASSERT_TRUE(result.has_value()) << result.error();

    // Should return the 12:00 and 14:00 events.
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].detectedClass, "car");
    EXPECT_EQ((*result)[1].detectedClass, "truck");
}

TEST_F(SecurityEventLoggerTest, QueryEvents_InclusiveBoundaries)
{
    SecurityEventLogger logger(tmpDir_);

    (void)logger.appendEvent(makeEvent("2026-04-12T10:00:00.000Z", "person"));
    (void)logger.appendEvent(makeEvent("2026-04-12T12:00:00.000Z", "car"));
    (void)logger.appendEvent(makeEvent("2026-04-12T14:00:00.000Z", "truck"));

    // Query exactly matching the first and last timestamps.
    auto result = logger.queryEvents("2026-04-12T10:00:00.000Z",
                                      "2026-04-12T14:00:00.000Z");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 3u) << "Boundary timestamps must be inclusive";
}

TEST_F(SecurityEventLoggerTest, QueryEvents_NoMatchesReturnsEmpty)
{
    SecurityEventLogger logger(tmpDir_);

    (void)logger.appendEvent(makeEvent("2026-04-12T10:00:00.000Z"));

    auto result = logger.queryEvents("2026-04-13T00:00:00.000Z",
                                      "2026-04-13T23:59:59.000Z");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST_F(SecurityEventLoggerTest, QueryEvents_EmptyLogReturnsEmpty)
{
    SecurityEventLogger logger(tmpDir_);

    auto result = logger.queryEvents("2026-04-12T00:00:00.000Z",
                                      "2026-04-12T23:59:59.000Z");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

// ===========================================================================
// Round-trip — "stored fields match original SecurityEvent"
// ===========================================================================

TEST_F(SecurityEventLoggerTest, EventRoundTrip_AllFieldsPreserved)
{
    SecurityEventLogger logger(tmpDir_);

    SecurityEvent original{
        .timestamp     = "2026-04-12T14:30:15.123Z",
        .detectedClass = "backpack",
        .confidence    = 0.87f,
        .bboxX         = 0.123f,
        .bboxY         = 0.456f,
        .bboxW         = 0.111f,
        .bboxH         = 0.222f,
        .recordingFile = "security_00042.mp4",
        .frameSeq      = 99999,
    };

    (void)logger.appendEvent(original);

    auto result = logger.queryEvents("2026-04-12T14:00:00.000Z",
                                      "2026-04-12T15:00:00.000Z");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);

    const auto& ev = (*result)[0];
    EXPECT_EQ(ev.timestamp, original.timestamp);
    EXPECT_EQ(ev.detectedClass, original.detectedClass);
    EXPECT_NEAR(ev.confidence, original.confidence, 1e-5f);
    EXPECT_NEAR(ev.bboxX, original.bboxX, 1e-5f);
    EXPECT_NEAR(ev.bboxY, original.bboxY, 1e-5f);
    EXPECT_NEAR(ev.bboxW, original.bboxW, 1e-5f);
    EXPECT_NEAR(ev.bboxH, original.bboxH, 1e-5f);
    EXPECT_EQ(ev.recordingFile, original.recordingFile);
    EXPECT_EQ(ev.frameSeq, original.frameSeq);
}

// ===========================================================================
// Rotation — "log doesn't fill the disk"
// ===========================================================================

TEST_F(SecurityEventLoggerTest, Rotation_RenamesFileWhenSizeExceeded)
{
    // Use a tiny max size (1 byte) to force immediate rotation.
    SecurityEventLogger logger(tmpDir_, "security_events.jsonl", /*maxSizeMB=*/0);

    // First event opens the file.
    (void)logger.appendEvent(makeEvent("2026-04-12T10:00:00.000Z"));

    // Second event triggers rotation (file > 0 MB threshold).
    (void)logger.appendEvent(makeEvent("2026-04-12T10:01:00.000Z"));

    // There should now be a rotated file AND a current file.
    int jsonlFiles = 0;
    for (const auto& entry : fs::directory_iterator(tmpDir_)) {
        if (entry.path().string().find("security_events") != std::string::npos) {
            ++jsonlFiles;
        }
    }
    EXPECT_GE(jsonlFiles, 2) << "Rotation should create a renamed old file + a fresh file";
}

TEST_F(SecurityEventLoggerTest, CurrentLogPath_PointsToActiveFile)
{
    SecurityEventLogger logger(tmpDir_);

    auto path = logger.currentLogPath();
    EXPECT_EQ(path.filename().string(), "security_events.jsonl");
    EXPECT_EQ(path.parent_path(), tmpDir_);
}

// ===========================================================================
// Malformed data resilience
// ===========================================================================

TEST_F(SecurityEventLoggerTest, QuerySkipsMalformedLines)
{
    // Pre-populate the log with a mix of valid and invalid lines.
    {
        std::ofstream out(tmpDir_ / "security_events.jsonl");
        out << R"({"timestamp":"2026-04-12T10:00:00.000Z","class":"person","confidence":0.9,"bbox":{"x":0.1,"y":0.2,"w":0.3,"h":0.4},"recording_file":"a.mp4","frame_seq":1})" << '\n';
        out << "THIS IS NOT JSON\n";
        out << R"({"timestamp":"2026-04-12T12:00:00.000Z","class":"car","confidence":0.8,"bbox":{"x":0.5,"y":0.6,"w":0.1,"h":0.2},"recording_file":"b.mp4","frame_seq":2})" << '\n';
    }

    SecurityEventLogger logger(tmpDir_);
    auto result = logger.queryEvents("2026-04-12T00:00:00.000Z",
                                      "2026-04-12T23:59:59.000Z");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2u) << "Valid lines must be returned; malformed lines skipped";
}

// ===========================================================================
// Thread safety — "concurrent appends don't corrupt the log"
// ===========================================================================

TEST_F(SecurityEventLoggerTest, ConcurrentAppends_NoCorruption)
{
    SecurityEventLogger logger(tmpDir_);
    constexpr int kThreads = 4;
    constexpr int kEventsPerThread = 50;

    auto worker = [&](int threadId) {
        for (int i = 0; i < kEventsPerThread; ++i) {
            char ts[64];
            std::snprintf(ts, sizeof(ts), "2026-04-12T%02d:%02d:%02d.000Z",
                          threadId, i / 60, i % 60);
            auto result = logger.appendEvent(makeEvent(ts, "person",
                                             0.5f + threadId * 0.1f));
            EXPECT_TRUE(result.has_value());
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(logger.eventCount(), kThreads * kEventsPerThread);

    // Verify every line is valid JSON.
    auto lines = readLogLines();
    EXPECT_EQ(lines.size(), static_cast<size_t>(kThreads * kEventsPerThread));
    for (size_t i = 0; i < lines.size(); ++i) {
        EXPECT_NO_THROW(nlohmann::json::parse(lines[i]))
            << "Line " << i << " is not valid JSON — concurrent write corruption";
    }
}
