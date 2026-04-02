#pragma once

#include <atomic>
#include <functional>
#include <stdexcept>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <tl/expected.hpp>

// ---------------------------------------------------------------------------
// OmniEdge_AI — GStreamer Pipeline RAII Wrapper
//
// Wraps a single GStreamer pipeline (created from a launch string) and
// exposes a typed appsink callback so the caller never touches raw GLib.
//
// Lifecycle:
//   GStreamerPipeline pipeline(launchStr, "appsink_name", onSample);
//   pipeline.start();    // gst_element_set_state(GST_STATE_PLAYING)
//   // ... appsink fires onSample on the GStreamer streaming thread ...
//   pipeline.stop();     // GST_STATE_NULL + cleanup
//
// Thread safety:
//   onSample() fires on the GStreamer streaming thread (internal to GLib).
//   Callers must synchronise access to shared state from inside onSample().
//   start() / stop() must be called from the same thread that owns the node.
// ---------------------------------------------------------------------------


/**
 * @brief RAII wrapper for a GStreamer parse-launch pipeline with one appsink.
 *
 * Non-copyable, non-movable — owns GStreamer element references.
 */
class GStreamerPipeline {
public:
	/**
	 * @brief Construct and parse the pipeline.  Does NOT start playback.
	 *
	 * @param launchString   gst-launch-1.0 style pipeline description,
	 *                       e.g. "tcpclientsrc ... ! appsink name=appsink0"
	 * @param appsinkName    Name of the appsink element inside the pipeline.
	 * @param callback       Invoked on the GStreamer streaming thread for each
	 *                       decoded sample.
	 * @throws std::runtime_error if gst_parse_launch fails.
	 */
	GStreamerPipeline(const std::string&  launchString,
					  const std::string&  appsinkName,
					  std::function<void(const uint8_t*, std::size_t, uint64_t)> callback);

	~GStreamerPipeline();

	GStreamerPipeline(const GStreamerPipeline&)            = delete;
	GStreamerPipeline& operator=(const GStreamerPipeline&) = delete;

	/**
	 * @brief Transition pipeline to PLAYING.
	 * Waits up to 5 s for live-source async state change to complete.
	 * @throws std::runtime_error if state change fails.
	 */
	void start();

	/**
	 * @brief Drain EOS and transition pipeline to NULL (releases resources).
	 * Safe to call from any thread.  Idempotent.
	 */
	void stop() noexcept;

	/** @brief true between start() and stop(). */
	[[nodiscard]] bool isRunning() const noexcept { return running_.load(std::memory_order_acquire); }

	/**
	 * @brief Non-blocking poll of the GStreamer bus for ERROR or EOS messages.
	 * Logs any error found and sets running_ to false.
	 * @return true if an error/EOS was found (pipeline is no longer healthy).
	 */
	bool checkBusErrors() noexcept;

private:
	static GstFlowReturn onNewSampleStatic(GstAppSink* appsink, gpointer userData);
	GstFlowReturn onNewSample(GstAppSink* appsink);

	GstElement*     pipeline_ = nullptr;
	GstAppSink*     appsink_  = nullptr;
	std::function<void(const uint8_t*, std::size_t, uint64_t)> callback_;
	std::atomic<bool> running_{false};
};

