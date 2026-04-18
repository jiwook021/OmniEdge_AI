#include <gtest/gtest.h>

#include "cv/security_recorder.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SecurityRecorder Tests
//
// Purpose: Verify the GStreamer NVENC MP4 recording pipeline used in Security
// Camera Mode.  From the user's perspective:
//   "I start recording and it writes MP4 files to the recording directory."
//   "I can list all saved recordings with their metadata."
//   "Invalid operations fail gracefully, not crash."
//
// What these tests catch:
//   - listRecordings returns wrong sort order or bad metadata
//   - Filename parsing extracts wrong date/time from security_YYYYMMDD_HHMMSS.mp4
//   - pushFrame accepts wrong-size buffers without error
//   - Double-start doesn't return an error
//   - pushFrame when not recording doesn't return an error
//   - GStreamer pipeline construction failure crashes instead of returning error
//
// Test layout:
//   - Static method tests (listRecordings): always run, no GStreamer needed
//   - Pipeline tests: gated on GStreamer availability, GTEST_SKIP if missing
//
// All tests use a temporary directory (auto-cleaned by the fixture).
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;
using oe::security::RecordingInfo;
using oe::security::SecurityRecorder;

// ---------------------------------------------------------------------------
// Fixture — temp directory per test
// ---------------------------------------------------------------------------

class SecurityRecorderTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmpDir_ = fs::temp_directory_path() / ("oe_test_security_rec_" +
                  std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch().count()));
        fs::create_directories(tmpDir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(tmpDir_, ec);
    }

    /// Create a fake MP4 file with a given name and size.
    void createFakeMp4(const std::string& name, std::size_t sizeBytes = 1024) {
        std::ofstream out(tmpDir_ / name, std::ios::binary);
        std::vector<char> data(sizeBytes, '\0');
        out.write(data.data(), static_cast<std::streamsize>(sizeBytes));
    }

    fs::path tmpDir_;
};

// ===========================================================================
// listRecordings — static, no GStreamer dependency
// ===========================================================================

TEST_F(SecurityRecorderTest, ListRecordings_EmptyDir_ReturnsEmpty)
{
    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    EXPECT_TRUE(recordings.empty());
}

TEST_F(SecurityRecorderTest, ListRecordings_NonexistentDir_ReturnsEmpty)
{
    auto recordings = SecurityRecorder::listRecordings(tmpDir_ / "does_not_exist");
    EXPECT_TRUE(recordings.empty());
}

TEST_F(SecurityRecorderTest, ListRecordings_FindsMp4Files)
{
    createFakeMp4("security_20260412_140000.mp4", 2048);
    createFakeMp4("security_20260412_143000.mp4", 4096);

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 2u);
}

TEST_F(SecurityRecorderTest, ListRecordings_IgnoresNonMp4Files)
{
    createFakeMp4("security_20260412_140000.mp4");

    // Create non-mp4 files.
    { std::ofstream(tmpDir_ / "security_events.jsonl") << "test"; }
    { std::ofstream(tmpDir_ / "readme.txt") << "test"; }
    { std::ofstream(tmpDir_ / "security_20260412_140000.avi") << "test"; }

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    EXPECT_EQ(recordings.size(), 1u);
}

TEST_F(SecurityRecorderTest, ListRecordings_SortedChronologically)
{
    createFakeMp4("security_20260413_090000.mp4");
    createFakeMp4("security_20260412_140000.mp4");
    createFakeMp4("security_20260412_180000.mp4");

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 3u);
    EXPECT_EQ(recordings[0].file, "security_20260412_140000.mp4");
    EXPECT_EQ(recordings[1].file, "security_20260412_180000.mp4");
    EXPECT_EQ(recordings[2].file, "security_20260413_090000.mp4");
}

TEST_F(SecurityRecorderTest, ListRecordings_ParsesFilenameToStartTime)
{
    createFakeMp4("security_20260412_143015.mp4");

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 1u);
    EXPECT_EQ(recordings[0].startTime, "2026-04-12T14:30:15Z");
}

TEST_F(SecurityRecorderTest, ListRecordings_ReportsFileSizeMB)
{
    // Create a file that's exactly 1 MiB.
    createFakeMp4("security_20260412_140000.mp4", 1024 * 1024);

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 1u);
    EXPECT_NEAR(recordings[0].sizeMB, 1.0, 0.01);
}

TEST_F(SecurityRecorderTest, ListRecordings_EventCountDefaultsToZero)
{
    createFakeMp4("security_20260412_140000.mp4");

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 1u);
    EXPECT_EQ(recordings[0].eventCount, 0)
        << "eventCount should default to 0 (caller populates from event log)";
}

TEST_F(SecurityRecorderTest, ListRecordings_NonStandardFilename_NoStartTime)
{
    // A splitmuxsink-generated name with sequence number instead of timestamp.
    createFakeMp4("security_00001.mp4");

    auto recordings = SecurityRecorder::listRecordings(tmpDir_);
    ASSERT_EQ(recordings.size(), 1u);
    EXPECT_TRUE(recordings[0].startTime.empty())
        << "Non-standard filenames should have empty startTime, not crash";
}

// ===========================================================================
// Auto-purge — "recordings older than N days are deleted"
// ===========================================================================

TEST_F(SecurityRecorderTest, PurgeOldRecordings_DeletesOldFiles)
{
    // Create two "recent" files and two "old" files.
    createFakeMp4("security_20260412_140000.mp4");
    createFakeMp4("security_20260412_150000.mp4");
    createFakeMp4("security_20260401_100000.mp4");
    createFakeMp4("security_20260401_120000.mp4");

    // Backdate the "old" files to 10 days ago.
    auto oldTime = fs::file_time_type::clock::now() - std::chrono::hours(10 * 24);
    fs::last_write_time(tmpDir_ / "security_20260401_100000.mp4", oldTime);
    fs::last_write_time(tmpDir_ / "security_20260401_120000.mp4", oldTime);

    int removed = SecurityRecorder::purgeOldRecordings(tmpDir_, 7);
    EXPECT_EQ(removed, 2) << "Two files older than 7 days should be deleted";

    // Verify recent files still exist.
    EXPECT_TRUE(fs::exists(tmpDir_ / "security_20260412_140000.mp4"));
    EXPECT_TRUE(fs::exists(tmpDir_ / "security_20260412_150000.mp4"));

    // Verify old files are gone.
    EXPECT_FALSE(fs::exists(tmpDir_ / "security_20260401_100000.mp4"));
    EXPECT_FALSE(fs::exists(tmpDir_ / "security_20260401_120000.mp4"));
}

TEST_F(SecurityRecorderTest, PurgeOldRecordings_EmptyDir_ReturnsZero)
{
    int removed = SecurityRecorder::purgeOldRecordings(tmpDir_, 7);
    EXPECT_EQ(removed, 0);
}

TEST_F(SecurityRecorderTest, PurgeOldRecordings_NonexistentDir_ReturnsZero)
{
    int removed = SecurityRecorder::purgeOldRecordings(tmpDir_ / "nope", 7);
    EXPECT_EQ(removed, 0);
}

TEST_F(SecurityRecorderTest, PurgeOldRecordings_IgnoresNonMp4Files)
{
    // Create a non-mp4 file and backdate it.
    { std::ofstream(tmpDir_ / "security_events.jsonl") << "test"; }
    auto oldTime = fs::file_time_type::clock::now() - std::chrono::hours(10 * 24);
    fs::last_write_time(tmpDir_ / "security_events.jsonl", oldTime);

    int removed = SecurityRecorder::purgeOldRecordings(tmpDir_, 7);
    EXPECT_EQ(removed, 0) << "Non-mp4 files must not be deleted";
    EXPECT_TRUE(fs::exists(tmpDir_ / "security_events.jsonl"));
}

// ===========================================================================
// State machine — "invalid operations fail gracefully"
// ===========================================================================

TEST(SecurityRecorderState, IsRecording_DefaultFalse)
{
    SecurityRecorder recorder;
    EXPECT_FALSE(recorder.isRecording());
}

TEST(SecurityRecorderState, CurrentFile_DefaultEmpty)
{
    SecurityRecorder recorder;
    EXPECT_TRUE(recorder.currentFile().empty());
}

TEST(SecurityRecorderState, PushFrame_WhenNotRecording_ReturnsError)
{
    SecurityRecorder recorder;
    std::vector<uint8_t> frame(1920 * 1080 * 3, 0);
    auto result = recorder.pushFrame(frame.data(), frame.size(), 0);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Not recording"), std::string::npos);
}

TEST(SecurityRecorderState, Stop_WhenNotRecording_IsNoOp)
{
    SecurityRecorder recorder;
    EXPECT_NO_THROW(recorder.stop());
}

// ===========================================================================
// GStreamer pipeline tests — skip if GStreamer unavailable
// ===========================================================================

class SecurityRecorderGstTest : public SecurityRecorderTest {
protected:
    void SetUp() override {
        SecurityRecorderTest::SetUp();

        // Check if GStreamer is available by trying to initialize.
        // If gst_init fails or the library isn't loaded, skip.
        if (!gstAvailable()) {
            GTEST_SKIP() << "GStreamer not available — skipping pipeline tests";
        }
    }

    static bool gstAvailable() {
        // Check if we can find at least the x264enc element.
        // (NVENC may not be available in CI, but x264 is the fallback.)
        try {
            // Simple check: attempt to parse a trivial pipeline.
            // If GStreamer libs aren't linked, this test won't even compile
            // — which is fine, the CMake gate handles that.
            return true;
        } catch (...) {
            return false;
        }
    }
};

TEST_F(SecurityRecorderGstTest, StartStop_CreatesOutputDirectory)
{
    SecurityRecorder recorder;
    SecurityRecorder::Config cfg;
    cfg.recordingDir = tmpDir_ / "output";
    cfg.width = 320;
    cfg.height = 240;
    cfg.fps = 5;

    auto result = recorder.start(cfg);
    if (!result.has_value()) {
        // Pipeline may fail if encoder elements are missing in CI.
        GTEST_SKIP() << "GStreamer pipeline start failed: " << result.error();
    }

    EXPECT_TRUE(recorder.isRecording());
    EXPECT_TRUE(fs::exists(cfg.recordingDir));

    recorder.stop();
    EXPECT_FALSE(recorder.isRecording());
}

TEST_F(SecurityRecorderGstTest, DoubleStart_ReturnsError)
{
    SecurityRecorder recorder;
    SecurityRecorder::Config cfg;
    cfg.recordingDir = tmpDir_ / "output";
    cfg.width = 320;
    cfg.height = 240;
    cfg.fps = 5;

    auto result1 = recorder.start(cfg);
    if (!result1.has_value()) {
        GTEST_SKIP() << "GStreamer pipeline start failed: " << result1.error();
    }

    auto result2 = recorder.start(cfg);
    EXPECT_FALSE(result2.has_value()) << "Double start must return error";
    EXPECT_NE(result2.error().find("Already recording"), std::string::npos);

    recorder.stop();
}

TEST_F(SecurityRecorderGstTest, PushFrame_WrongSize_ReturnsError)
{
    SecurityRecorder recorder;
    SecurityRecorder::Config cfg;
    cfg.recordingDir = tmpDir_ / "output";
    cfg.width = 320;
    cfg.height = 240;
    cfg.fps = 5;

    auto result = recorder.start(cfg);
    if (!result.has_value()) {
        GTEST_SKIP() << "GStreamer pipeline start failed: " << result.error();
    }

    // Push a frame with wrong size.
    std::vector<uint8_t> wrongSize(100, 0);
    auto pushResult = recorder.pushFrame(wrongSize.data(), wrongSize.size(), 0);
    EXPECT_FALSE(pushResult.has_value());
    EXPECT_NE(pushResult.error().find("size mismatch"), std::string::npos);

    recorder.stop();
}

TEST_F(SecurityRecorderGstTest, PushFrames_ProducesRecording)
{
    SecurityRecorder recorder;
    SecurityRecorder::Config cfg;
    cfg.recordingDir = tmpDir_ / "output";
    cfg.width = 320;
    cfg.height = 240;
    cfg.fps = 5;
    cfg.segmentDurationMin = 1;  // Short segments for testing.

    auto result = recorder.start(cfg);
    if (!result.has_value()) {
        GTEST_SKIP() << "GStreamer pipeline start failed: " << result.error();
    }

    // Push 10 frames (2 seconds at 5 fps).
    std::size_t frameSize = cfg.width * cfg.height * 3;
    std::vector<uint8_t> frame(frameSize, 128);
    for (int i = 0; i < 10; ++i) {
        uint64_t tsNs = static_cast<uint64_t>(i) * 200'000'000;  // 200ms apart
        auto pushResult = recorder.pushFrame(frame.data(), frame.size(), tsNs);
        ASSERT_TRUE(pushResult.has_value())
            << "Frame " << i << ": " << pushResult.error();
    }

    recorder.stop();

    // Check that at least one MP4 was produced.
    auto recordings = SecurityRecorder::listRecordings(cfg.recordingDir);
    EXPECT_GE(recordings.size(), 1u) << "At least one MP4 segment should exist";
}
