#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — ZMQ Socket Compile-Time Constants
//
// High-water marks and poll timeout for the ZMQ control and data planes.
// Used by ZmqPublisher, ZmqSubscriber, and MessageRouter in common/.
//
// Data-plane topics (video_frame, audio_chunk): low HWM — drop stale data.
// Control-plane topics (module_status, llm_prompt): high HWM — never drop.
// ---------------------------------------------------------------------------


/// PUB socket HWM for data-plane topics (video_frame, audio_chunk).
/// A depth of 2 keeps at most one queued frame per subscriber without
/// unbounded memory growth on slow consumers.
inline constexpr int kPublisherDataHighWaterMark = 2;

/// PUB socket HWM for control-plane topics (module_status, llm_prompt, etc.).
/// Higher depth to tolerate bursty control messages without dropping.
inline constexpr int kPublisherControlHighWaterMark = 100;

/// SUB socket HWM for control-plane subscriptions.
inline constexpr int kSubscriberControlHighWaterMark = 1000;

/// Maximum blocking time per zmq::poll() call across all modules (ms).
/// Governs worst-case stop() latency. Must stay well below SIGTERM grace.
inline constexpr int kPollTimeoutMs = 50;

/// ZMQ slow-joiner mitigation delay (ms). SUB sockets that connect() after a
/// PUB socket bind() silently drop messages published before the TCP handshake.
/// This sleep gives the subscriber time to complete the handshake on localhost.
inline constexpr int kSlowJoinerDelayMs = 100;

