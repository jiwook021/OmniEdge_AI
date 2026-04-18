// ---------------------------------------------------------------------------
// OmniEdge_AI — EventBus Unit Tests
//
// Tests the typed in-process event bus used by the daemon to decouple
// cross-component notifications (PipelineOrchestrator, VramGate,
// StateMachine, watchdog).
//
// Bugs these tests catch:
//   - Handler receives wrong event type (type-erasure cast error)
//   - clearHandlers() doesn't actually remove handlers
//   - Multiple handlers for same event — all fire, not just first/last
//   - handlerCount() returns stale count after clear
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/event_bus.hpp"
#include "common/oe_logger.hpp"

namespace {

// ── Test-local event types ──────────────────────────────────────────────

struct TestEventA {
    int value{0};
};

struct TestEventB {
    std::string name;
    double score{0.0};
};

struct TestEventC {};  // empty event (fire-and-forget signal)

// ── Fixture ─────────────────────────────────────────────────────────────

class EventBusTest : public ::testing::Test {
protected:
    EventBus bus;
};

// ── Subscribe + Publish ─────────────────────────────────────────────────

TEST_F(EventBusTest, SingleSubscriberReceivesEvent)
{
    int received = -1;
    bus.subscribe<TestEventA>([&](const TestEventA& e) {
        received = e.value;
    });

    bus.publish(TestEventA{.value = 42});
    EXPECT_EQ(received, 42);
}

TEST_F(EventBusTest, MultipleSubscribersAllFire)
{
    std::vector<int> results;
    bus.subscribe<TestEventA>([&](const TestEventA& e) {
        results.push_back(e.value * 2);
    });
    bus.subscribe<TestEventA>([&](const TestEventA& e) {
        results.push_back(e.value * 3);
    });

    bus.publish(TestEventA{.value = 10});
    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0], 20);
    EXPECT_EQ(results[1], 30);
}

TEST_F(EventBusTest, DifferentEventTypesAreIsolated)
{
    int receivedA = 0;
    std::string receivedB;

    bus.subscribe<TestEventA>([&](const TestEventA& e) { receivedA = e.value; });
    bus.subscribe<TestEventB>([&](const TestEventB& e) { receivedB = e.name; });

    bus.publish(TestEventA{.value = 7});
    EXPECT_EQ(receivedA, 7);
    EXPECT_TRUE(receivedB.empty()) << "TestEventB handler should not fire on TestEventA publish";

    bus.publish(TestEventB{.name = "hello", .score = 1.5});
    EXPECT_EQ(receivedB, "hello");
    EXPECT_EQ(receivedA, 7) << "TestEventA handler should not fire again on TestEventB publish";
}

// ── clearHandlers ───────────────────────────────────────────────────────

TEST_F(EventBusTest, ClearHandlersRemovesSpecificType)
{
    int countA = 0;
    int countB = 0;
    bus.subscribe<TestEventA>([&](const TestEventA&) { ++countA; });
    bus.subscribe<TestEventB>([&](const TestEventB&) { ++countB; });

    bus.clearHandlers<TestEventA>();

    bus.publish(TestEventA{});
    bus.publish(TestEventB{.name = "test"});
    EXPECT_EQ(countA, 0) << "Handler should have been cleared";
    EXPECT_EQ(countB, 1) << "Unrelated handler should still fire";
}

TEST_F(EventBusTest, ClearAllRemovesEverything)
{
    int countA = 0;
    int countB = 0;
    bus.subscribe<TestEventA>([&](const TestEventA&) { ++countA; });
    bus.subscribe<TestEventB>([&](const TestEventB&) { ++countB; });

    bus.clearAll();

    bus.publish(TestEventA{});
    bus.publish(TestEventB{.name = "test"});
    EXPECT_EQ(countA, 0);
    EXPECT_EQ(countB, 0);
}

// ── handlerCount ────────────────────────────────────────────────────────

TEST_F(EventBusTest, HandlerCountTracksRegistrations)
{
    EXPECT_EQ(bus.handlerCount<TestEventA>(), 0U);

    bus.subscribe<TestEventA>([](const TestEventA&) {});
    EXPECT_EQ(bus.handlerCount<TestEventA>(), 1U);

    bus.subscribe<TestEventA>([](const TestEventA&) {});
    EXPECT_EQ(bus.handlerCount<TestEventA>(), 2U);

    // Different type is still zero
    EXPECT_EQ(bus.handlerCount<TestEventB>(), 0U);

    bus.clearHandlers<TestEventA>();
    EXPECT_EQ(bus.handlerCount<TestEventA>(), 0U);
}

// ── Daemon event types ( namespace) ────────────────────────────────
// Verify the actual daemon event types work correctly through the bus.

TEST_F(EventBusTest, BusModeChangedEvent)
{
    ModeChanged received;
    bus.subscribe<ModeChanged>([&](const ModeChanged& e) {
        received = e;
    });

    bus.publish(ModeChanged{.modeName = "conversation", .success = true});
    EXPECT_EQ(received.modeName, "conversation");
    EXPECT_TRUE(received.success);
}

TEST_F(EventBusTest, BusVramChangedEvent)
{
    std::vector<VramChanged> events;
    bus.subscribe<VramChanged>([&](const VramChanged& e) {
        events.push_back(e);
    });

    bus.publish(VramChanged{.moduleName = "llm", .budgetMiB = 4300, .loaded = true});
    bus.publish(VramChanged{.moduleName = "stt", .budgetMiB = 1500, .loaded = false});

    ASSERT_EQ(events.size(), 2U);
    EXPECT_EQ(events[0].moduleName, "llm");
    EXPECT_EQ(events[0].budgetMiB, 4300U);
    EXPECT_TRUE(events[0].loaded);
    EXPECT_EQ(events[1].moduleName, "stt");
    EXPECT_FALSE(events[1].loaded);
}

TEST_F(EventBusTest, BusModuleCrashedEvent)
{
    ModuleCrashed received;
    bus.subscribe<ModuleCrashed>([&](const ModuleCrashed& e) {
        received = e;
    });

    bus.publish(ModuleCrashed{.moduleName = "stt", .pid = 12345, .signal = 11});
    EXPECT_EQ(received.moduleName, "stt");
    EXPECT_EQ(received.pid, 12345);
    EXPECT_EQ(received.signal, 11);  // SIGSEGV
}

TEST_F(EventBusTest, BusVramEvictionEvent)
{
    VramEviction received;
    bus.subscribe<VramEviction>([&](const VramEviction& e) {
        received = e;
    });

    bus.publish(VramEviction{.moduleName = "bg_blur", .usedMb = 11500, .capMb = 11264});
    EXPECT_EQ(received.moduleName, "bg_blur");
    EXPECT_EQ(received.usedMb, 11500U);
    EXPECT_EQ(received.capMb, 11264U);
}

TEST_F(EventBusTest, BusModuleDisabledEvent)
{
    ModuleDisabled received;
    bus.subscribe<ModuleDisabled>([&](const ModuleDisabled& e) {
        received = e;
    });

    bus.publish(ModuleDisabled{
        .moduleName = "tts",
        .reason     = "exceeded max segfault restarts (3)",
    });
    EXPECT_EQ(received.moduleName, "tts");
    EXPECT_EQ(received.reason, "exceeded max segfault restarts (3)");
}

} // namespace
