#pragma once

// ---------------------------------------------------------------------------
// OmniEdge_AI — Typed In-Process Event Bus
//
// Decouples cross-component notifications inside a single process (the daemon).
// Components subscribe to typed events and publish them without knowing about
// each other. This replaces hard-wired direct calls between ModeOrchestrator,
// VramGate, StateMachine, and the watchdog.
//
// Design:
//   - Type-safe: publish<ModeChanged>(evt) only reaches ModeChanged handlers
//   - Single-threaded: NOT thread-safe — all calls from zmq_poll() loop
//   - Zero allocation on publish: handlers stored in pre-allocated vectors
//   - No virtual dispatch: std::function with type-erased void* cast
//
// Usage:
//   EventBus bus;
//   bus.subscribe<ModeChanged>([](const ModeChanged& e) { ... });
//   bus.publish(ModeChanged{"conversation"});
// ---------------------------------------------------------------------------

#include <cstddef>
#include <functional>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "common/oe_logger.hpp"
#include "common/oe_tracy.hpp"


// ---------------------------------------------------------------------------
// Daemon-internal event types — lightweight POD structs
// ---------------------------------------------------------------------------


/// A UI-initiated mode switch completed (or failed).
struct ModeChanged {
    std::string modeName;
    bool        success{true};
};

/// A module's VRAM allocation changed (loaded/unloaded/evicted).
struct VramChanged {
    std::string moduleName;
    std::size_t budgetMiB{0};
    bool        loaded{false};  ///< true = loaded, false = unloaded
};

/// Watchdog detected a module crash.
struct ModuleCrashed {
    std::string moduleName;
    int         pid{-1};
    int         signal{0};       ///< 0 = normal exit, >0 = signal number
};

/// A module was restarted after a crash.
struct ModuleRestarted {
    std::string moduleName;
    int         newPid{-1};
};

/// A module was permanently disabled (e.g. exceeded segfault limit).
struct ModuleDisabled {
    std::string moduleName;
    std::string reason;
};

/// Watchdog detected a module that stopped sending heartbeats (likely hung/deadlocked).
struct ModuleHung {
    std::string moduleName;
    int         pid{-1};
    int         silenceMs{0};   ///< Milliseconds since last heartbeat
};

/// VRAM watchdog evicted a module due to pressure.
struct VramEviction {
    std::string moduleName;
    std::size_t usedMb{0};
    std::size_t capMb{0};
};

/// State machine transitioned to a new state.
struct StateTransition {
    std::size_t fromIndex{0};
    std::size_t toIndex{0};
};

/// Interaction profile changed — priorities shifted for all modules.
struct InteractionProfileChanged {
    std::string fromProfile;
    std::string toProfile;
    std::size_t modulesChanged{0};  ///< Number of modules whose priority changed
};

/// Conversation model changed (four-mode architecture).
/// Fired when the user selects a different model within Conversation mode
/// (Qwen2.5-Omni-7B / 3B / Gemma 4 E4B).  The daemon kills the previous
/// model process and spawns the newly selected one.
struct ConversationModelChanged {
    std::string previousModel;     ///< e.g. "qwen_omni_7b"
    std::string newModel;          ///< e.g. "gemma_e4b"
    bool        needsTts{false};   ///< true if the new model requires a companion TTS process
};

/// Image transform result (four-mode architecture).
/// Fired when the Stable Diffusion pipeline completes a style transfer.
struct ImageTransformResult {
    std::string style;             ///< Selected style (cat, anime, cartoon, etc.)
    bool        success{true};
    std::string errorMessage;
};


// ---------------------------------------------------------------------------
// EventBus — typed publish/subscribe for in-process daemon events
// ---------------------------------------------------------------------------

class EventBus {
public:
    /// Subscribe a handler for a specific event type.
    /// Returns a subscription index that can be used for unsubscribe.
    template <typename Event>
    std::size_t subscribe(std::function<void(const Event&)> handler)
    {
        auto& vec = handlers_[std::type_index(typeid(Event))];
        vec.push_back([h = std::move(handler)](const void* e) {
            h(*static_cast<const Event*>(e));
        });
        return vec.size() - 1;
    }

    /// Publish an event to all subscribed handlers of that type.
    template <typename Event>
    void publish(const Event& event)
    {
        OE_ZONE_SCOPED;
        auto it = handlers_.find(std::type_index(typeid(Event)));
        if (it == handlers_.end()) return;
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            try {
                it->second[i](&event);
            } catch (const std::exception& ex) {
                ++failureCount_;
                SPDLOG_ERROR("EventBus handler threw (failures={}): {}",
                             failureCount_, ex.what());
            } catch (...) {
                ++failureCount_;
                SPDLOG_ERROR("EventBus handler threw unknown exception (failures={})",
                             failureCount_);
            }
        }
    }

    /// Remove all handlers for a specific event type.
    template <typename Event>
    void clearHandlers()
    {
        handlers_.erase(std::type_index(typeid(Event)));
    }

    /// Remove all handlers for all event types.
    void clearAll() noexcept
    {
        handlers_.clear();
    }

    /// Number of handlers registered for a specific event type.
    template <typename Event>
    [[nodiscard]] std::size_t handlerCount() const
    {
        auto it = handlers_.find(std::type_index(typeid(Event)));
        return (it != handlers_.end()) ? it->second.size() : 0;
    }

    /// Total number of handler exceptions caught since bus creation.
    /// Daemon health probe can check this to detect degraded state.
    [[nodiscard]] std::size_t failureCount() const noexcept { return failureCount_; }

private:
    std::unordered_map<std::type_index,
        std::vector<std::function<void(const void*)>>> handlers_;
    std::size_t failureCount_{0};
};

