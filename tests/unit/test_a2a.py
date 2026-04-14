from __future__ import annotations

import threading
import time

import pytest

from mlxharness.a2a import (
    A2AMessage,
    AgentHandle,
    AgentRegistry,
    AgentStatus,
    Inbox,
)


class TestA2AMessage:
    def test_fields(self):
        msg = A2AMessage(from_id="orchestrator", to_id="agent-1", content="hi")
        assert msg.from_id == "orchestrator"
        assert msg.to_id == "agent-1"
        assert msg.content == "hi"


class TestInbox:
    def test_fifo(self):
        inbox = Inbox()
        inbox.put(A2AMessage("a", "b", "one"))
        inbox.put(A2AMessage("a", "b", "two"))
        assert inbox.get(timeout=0.1).content == "one"
        assert inbox.get(timeout=0.1).content == "two"

    def test_blocking_get_timeout_returns_none(self):
        inbox = Inbox()
        start = time.monotonic()
        assert inbox.get(timeout=0.05) is None
        assert time.monotonic() - start >= 0.05

    def test_blocking_get_wakes_on_put(self):
        inbox = Inbox()
        received: list[A2AMessage] = []

        def consumer():
            msg = inbox.get(timeout=1)
            if msg:
                received.append(msg)

        t = threading.Thread(target=consumer)
        t.start()
        time.sleep(0.02)
        inbox.put(A2AMessage("x", "y", "hello"))
        t.join(timeout=1)
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_drain_returns_all_and_empties(self):
        inbox = Inbox()
        for i in range(3):
            inbox.put(A2AMessage("a", "b", str(i)))
        msgs = inbox.drain()
        assert [m.content for m in msgs] == ["0", "1", "2"]
        assert inbox.drain() == []


class TestAgentRegistry:
    def _make_handle(self, agent_id: str = "a1", role: str = "worker") -> AgentHandle:
        return AgentHandle(
            id=agent_id,
            role=role,
            status=AgentStatus.IDLE,
            inbox=Inbox(),
            outbox=Inbox(),
            thread=None,
        )

    def test_register_and_get(self):
        reg = AgentRegistry()
        h = self._make_handle("a1")
        reg.register(h)
        assert reg.get("a1") is h

    def test_get_missing_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("nope") is None

    def test_list_snapshot(self):
        reg = AgentRegistry()
        reg.register(self._make_handle("a1"))
        reg.register(self._make_handle("a2"))
        ids = {h.id for h in reg.list()}
        assert ids == {"a1", "a2"}

    def test_register_duplicate_id_raises(self):
        reg = AgentRegistry()
        reg.register(self._make_handle("a1"))
        with pytest.raises(ValueError):
            reg.register(self._make_handle("a1"))

    def test_next_id_generates_sequential_ids(self):
        reg = AgentRegistry()
        assert reg.next_id() == "agent-1"
        assert reg.next_id() == "agent-2"
        assert reg.next_id() == "agent-3"


class TestAgentStatus:
    def test_enum_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.DONE.value == "done"
        assert AgentStatus.ERROR.value == "error"
