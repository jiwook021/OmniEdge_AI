#include "ingest/gstreamer_pipeline.hpp"

#include <format>
#include <stdexcept>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

GStreamerPipeline::GStreamerPipeline(const std::string& launchString,
									  const std::string& appsinkName,
									  std::function<void(const uint8_t*, std::size_t, uint64_t)> callback)
	: callback_(std::move(callback))
{
	GError* err = nullptr;
	pipeline_ = gst_parse_launch(launchString.c_str(), &err);
	if (!pipeline_ || err) {
		const std::string msg = err ? err->message : "unknown error";
		if (err) { g_error_free(err); }
		throw std::runtime_error(
			std::format("[GStreamerPipeline] gst_parse_launch failed: {}", msg));
	}

	// Retrieve the appsink by name so we can set its callback
	GstElement* sinkElement = gst_bin_get_by_name(GST_BIN(pipeline_),
												   appsinkName.c_str());
	if (!sinkElement) {
		gst_object_unref(pipeline_);
		pipeline_ = nullptr;
		throw std::runtime_error(
			std::format("[GStreamerPipeline] appsink '{}' not found in pipeline",
						appsinkName));
	}
	appsink_ = GST_APP_SINK(sinkElement);

	// Wire up the new-sample callback
	GstAppSinkCallbacks callbacks{};
	callbacks.new_sample = &GStreamerPipeline::onNewSampleStatic;
	gst_app_sink_set_callbacks(appsink_, &callbacks, this, nullptr);

	// Emit signals via callback only — not via GSignal (avoids main-loop dep)
	gst_app_sink_set_emit_signals(appsink_, FALSE);

	OE_LOG_DEBUG("gstreamer_pipeline_created: sink={}", appsinkName);
}

GStreamerPipeline::~GStreamerPipeline()
{
	stop();
	// Release the extra ref from gst_bin_get_by_name before unrefing pipeline
	if (appsink_) {
		gst_object_unref(GST_ELEMENT(appsink_));
		appsink_ = nullptr;
	}
	if (pipeline_) {
		gst_object_unref(pipeline_);
		pipeline_ = nullptr;
	}
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void GStreamerPipeline::start()
{
	const GstStateChangeReturn ret =
		gst_element_set_state(pipeline_, GST_STATE_PLAYING);

	if (ret == GST_STATE_CHANGE_FAILURE) {
		// Log a detailed error instead of throwing immediately — GStreamer may
		// have posted error messages to the bus with more context.
		GstBus* bus = gst_element_get_bus(pipeline_);
		if (bus) {
			GstMessage* errMsg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
			if (errMsg) {
				GError* err = nullptr;
				gchar* debugInfo = nullptr;
				gst_message_parse_error(errMsg, &err, &debugInfo);
				OE_LOG_ERROR("gstreamer_pipeline_error: msg={}, debug={}",
				           err ? err->message : "unknown",
				           debugInfo ? debugInfo : "none");
				if (err) g_error_free(err);
				g_free(debugInfo);
				gst_message_unref(errMsg);
			}
			gst_object_unref(bus);
		}
		throw std::runtime_error("[GStreamerPipeline] Failed to set PLAYING state");
	}

	// Live sources (v4l2src, tcpclientsrc) return ASYNC — wait up to 2 s
	// for the pipeline to reach PLAYING.  Must be well under the daemon's
	// heartbeat timeout (5 s) because onBeforeRun() blocks before the
	// poll loop starts publishing heartbeats.
	if (ret == GST_STATE_CHANGE_ASYNC) {
		GstState state     = GST_STATE_NULL;
		GstState pending   = GST_STATE_NULL;
		const GstStateChangeReturn waited =
			gst_element_get_state(pipeline_, &state, &pending,
			                      static_cast<GstClockTime>(2 * GST_SECOND));
		if (waited == GST_STATE_CHANGE_FAILURE) {
			throw std::runtime_error(
				"[GStreamerPipeline] Pipeline failed to reach PLAYING state");
		}
		OE_LOG_DEBUG("gstreamer_pipeline_async_done: state={}",
		        static_cast<int>(state));
	}

	running_.store(true, std::memory_order_release);
	OE_LOG_INFO("gstreamer_pipeline_started");
}

void GStreamerPipeline::stop() noexcept
{
	if (!running_.exchange(false)) {
		return;  // Already stopped — idempotent
	}
	if (pipeline_) {
		// Send EOS downstream so all elements flush cleanly
		gst_element_send_event(pipeline_, gst_event_new_eos());

		// Wait for EOS to reach the appsink (max 2 s) or just go NULL directly
		gst_element_set_state(pipeline_, GST_STATE_NULL);
	}
	OE_LOG_INFO("gstreamer_pipeline_stopped");
}

// ---------------------------------------------------------------------------
// appsink callback
// ---------------------------------------------------------------------------

// static
GstFlowReturn GStreamerPipeline::onNewSampleStatic(GstAppSink* appsink,
													gpointer    userData)
{
	return static_cast<GStreamerPipeline*>(userData)->onNewSample(appsink);
}

GstFlowReturn GStreamerPipeline::onNewSample(GstAppSink* appsink)
{
	GstSample* sample = gst_app_sink_pull_sample(appsink);
	if (!sample) {
		return GST_FLOW_ERROR;
	}

	GstBuffer* buffer = gst_sample_get_buffer(sample);
	GstMapInfo  map{};
	if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
		gst_sample_unref(sample);
		return GST_FLOW_ERROR;
	}

	const uint64_t pts = GST_BUFFER_PTS(buffer);
	OE_LOG_DEBUG("gstreamer_sample: bytes={}, pts={}",
	           static_cast<std::size_t>(map.size), pts);
	try {
		callback_(static_cast<const uint8_t*>(map.data),
				  static_cast<std::size_t>(map.size),
				  pts);
	} catch (const std::exception& e) {
		OE_LOG_WARN("gstreamer_callback_exception: what={}", e.what());
	}

	gst_buffer_unmap(buffer, &map);
	gst_sample_unref(sample);
	return GST_FLOW_OK;
}

// ---------------------------------------------------------------------------
// checkBusErrors() — non-blocking bus poll
// ---------------------------------------------------------------------------

bool GStreamerPipeline::checkBusErrors() noexcept
{
	GstBus* bus = gst_element_get_bus(pipeline_);
	if (!bus) { return false; }

	// Drain all pending ERROR / EOS messages without blocking
	bool hadError = false;
	GstMessage* msg = nullptr;
	while ((msg = gst_bus_pop_filtered(
				bus,
				static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)))
	       != nullptr)
	{
		hadError = true;
		if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
			GError* err = nullptr;
			gchar*  dbg = nullptr;
			gst_message_parse_error(msg, &err, &dbg);
			OE_LOG_ERROR("gstreamer_bus_error: error={}, debug={}",
				err ? err->message : "unknown",
				dbg ? dbg : "");
			if (err) { g_error_free(err); }
			if (dbg) { g_free(dbg); }
		} else {
			OE_LOG_INFO("gstreamer_bus_eos");
		}
		gst_message_unref(msg);
	}

	gst_object_unref(bus);

	if (hadError) {
		running_.store(false, std::memory_order_release);
	}
	return hadError;
}

