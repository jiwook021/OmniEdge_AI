#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI -- Tracy Profiler Integration
//
// Zero-overhead wrapper macros for Tracy (https://github.com/wolfpld/tracy).
// When OE_ENABLE_TRACY is defined (via cmake -DOE_ENABLE_TRACY=ON), these
// macros expand to real Tracy calls. Otherwise they compile to nothing.
//
// Usage in any module .cpp:
//
//   #include "common/oe_tracy.hpp"
//
//   void MyNode::processFrame(const Frame& f)
//   {
//       OE_ZONE_SCOPED;                          // auto-named from __func__
//       OE_ZONE_TEXT(f.id.data(), f.id.size());   // attach metadata
//       // ... hot-path work ...
//   }
//
//   void MyNode::run()
//   {
//       while (running_.load()) {
//           OE_FRAME_MARK;                        // mark iteration boundary
//           poll();
//       }
//   }
//
// Convention: place OE_ZONE_SCOPED as the first statement in every function
// that appears on the hot path (run loops, processFrame, inference calls).
// Use OE_ZONE_SCOPED_N("descriptiveName") when the function name alone is
// ambiguous (lambdas, operator(), template instantiations).
//
// IMPORTANT: Tracy macros create local variables -- they are statement-like,
// not expressions.  Always place them on their own line.
// ---------------------------------------------------------------------------

#ifdef OE_ENABLE_TRACY

#include <tracy/Tracy.hpp>

// ---- CPU zones ------------------------------------------------------------

/// Instrument the enclosing scope with an auto-named zone (__func__ + line).
#define OE_ZONE_SCOPED              ZoneScoped

/// Instrument the enclosing scope with an explicit name (string literal).
#define OE_ZONE_SCOPED_N(name)      ZoneScopedN(name)

/// Attach a dynamic text annotation to the current zone.
#define OE_ZONE_TEXT(text, len)      ZoneText(text, len)

/// Attach a dynamic name to the current zone (overrides static name).
#define OE_ZONE_NAME(text, len)      ZoneName(text, len)

/// Set a colour on the current zone (use Tracy colour constants).
#define OE_ZONE_COLOR(color)         ZoneColor(color)

// ---- Frame marks ----------------------------------------------------------

/// Mark an unnamed frame boundary (main loop iteration).
#define OE_FRAME_MARK                FrameMark

/// Mark a named frame boundary (e.g., "stt_inference", "llm_generate").
#define OE_FRAME_MARK_N(name)        FrameMarkNamed(name)

/// Begin/end a discontinuous frame (for async work spanning multiple scopes).
#define OE_FRAME_MARK_START(name)    FrameMarkStart(name)
#define OE_FRAME_MARK_END(name)      FrameMarkEnd(name)

// ---- Plots ----------------------------------------------------------------

/// Plot a named numeric value on Tracy's timeline.
#define OE_PLOT(name, val)           TracyPlot(name, val)

// ---- Messages -------------------------------------------------------------

/// Log a message on Tracy's timeline (visible in the profiler UI).
#define OE_MESSAGE(text, len)        TracyMessage(text, len)

/// Log a literal string message (length computed at compile time).
#define OE_MESSAGE_L(text)           TracyMessageL(text)

// ---- Memory tracking ------------------------------------------------------

/// Track a heap allocation (pair with OE_FREE).
#define OE_ALLOC(ptr, size)          TracyAlloc(ptr, size)

/// Track a heap deallocation.
#define OE_FREE(ptr)                 TracyFree(ptr)

// ---- Mutex instrumentation ------------------------------------------------

/// Use TracyLockable(std::mutex, varName) to track lock contention.
/// Example:  TracyLockable(std::mutex, outputMutex_);
///           std::lock_guard<LockableBase(std::mutex)> lock(outputMutex_);
#define OE_LOCKABLE(type, var)       TracyLockable(type, var)
#define OE_LOCKABLE_BASE(type)       LockableBase(type)

#else // !OE_ENABLE_TRACY

// ---- No-op stubs when Tracy is disabled -----------------------------------

#define OE_ZONE_SCOPED               (void)0
#define OE_ZONE_SCOPED_N(name)       (void)0
#define OE_ZONE_TEXT(text, len)       (void)0
#define OE_ZONE_NAME(text, len)       (void)0
#define OE_ZONE_COLOR(color)          (void)0

#define OE_FRAME_MARK                 (void)0
#define OE_FRAME_MARK_N(name)         (void)0
#define OE_FRAME_MARK_START(name)     (void)0
#define OE_FRAME_MARK_END(name)       (void)0

#define OE_PLOT(name, val)            (void)0

#define OE_MESSAGE(text, len)         (void)0
#define OE_MESSAGE_L(text)            (void)0

#define OE_ALLOC(ptr, size)           (void)0
#define OE_FREE(ptr)                  (void)0

#define OE_LOCKABLE(type, var)        type var
#define OE_LOCKABLE_BASE(type)        type

#endif // OE_ENABLE_TRACY
