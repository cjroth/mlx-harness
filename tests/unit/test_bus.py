from __future__ import annotations

import threading
import time

from mlxharness.bus import EventBus
from mlxharness.events import DoneEvent, TokenEvent


class TestEventBus:
    def test_single_subscriber_receives_event(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.publish("a1", TokenEvent(text="hello"))
        agent_id, event = q.get(timeout=1)
        assert agent_id == "a1"
        assert event == TokenEvent(text="hello")

    def test_multiple_subscribers_each_receive(self):
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.publish("a1", DoneEvent())
        assert q1.get(timeout=1)[0] == "a1"
        assert q2.get(timeout=1)[0] == "a1"

    def test_fifo_order_within_subscriber(self):
        bus = EventBus()
        q = bus.subscribe()
        for i in range(5):
            bus.publish("a1", TokenEvent(text=str(i)))
        received = [q.get(timeout=1)[1].text for _ in range(5)]
        assert received == ["0", "1", "2", "3", "4"]

    def test_thread_safety_under_concurrent_publishers(self):
        bus = EventBus()
        q = bus.subscribe()
        N = 4
        M = 50

        def publish_many(agent_id: str):
            for i in range(M):
                bus.publish(agent_id, TokenEvent(text=f"{agent_id}:{i}"))

        threads = [threading.Thread(target=publish_many, args=(f"a{n}",)) for n in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        received = []
        deadline = time.time() + 2
        while len(received) < N * M and time.time() < deadline:
            try:
                received.append(q.get(timeout=0.1))
            except Exception:
                pass
        assert len(received) == N * M

        # Per-agent FIFO must still hold
        for n in range(N):
            agent_id = f"a{n}"
            agent_texts = [ev.text for (aid, ev) in received if aid == agent_id]
            assert agent_texts == [f"{agent_id}:{i}" for i in range(M)]

    def test_subscribe_after_publish_does_not_replay(self):
        bus = EventBus()
        bus.publish("a1", TokenEvent(text="before"))
        q = bus.subscribe()
        bus.publish("a1", TokenEvent(text="after"))
        agent_id, event = q.get(timeout=1)
        assert event.text == "after"
        assert q.empty()
