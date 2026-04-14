from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from mlxharness.a2a import AgentStatus
from mlxharness.bus import EventBus
from mlxharness.events import DoneEvent, TokenEvent
from mlxharness.executor import CommandResult
from mlxharness.orchestrator import Orchestrator


def make_engine(script: dict[str, list[list[str]]]):
    """Build a mock engine whose generate() response depends on system prompt identity.

    Keyed by substring of system prompt to allow orchestrator vs subagent to return
    different token sequences. Fallback to key 'default'.
    """
    engine = MagicMock()
    engine.context_window = 128000
    engine.count_tokens = MagicMock(return_value=100)

    state = {k: iter(v) for k, v in script.items()}

    def generate(messages, tools=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        for key, it in state.items():
            if key != "default" and key in system:
                return iter(next(it))
        return iter(next(state["default"]))

    engine.generate = MagicMock(side_effect=generate)
    return engine


class TestOrchestrator:
    def test_submit_user_and_plain_response(self):
        engine = make_engine({"default": [["hi there"]]})
        executor = MagicMock()
        bus = EventBus()
        q = bus.subscribe()

        orch = Orchestrator(engine=engine, executor=executor, bus=bus)
        orch.start()
        orch.submit_user("hello")

        collected: list[tuple[str, object]] = []
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                collected.append(q.get(timeout=0.1))
            except Exception:
                continue
            if any(isinstance(ev, DoneEvent) for (_, ev) in collected):
                break
        orch.shutdown()

        tokens = [ev for (aid, ev) in collected if isinstance(ev, TokenEvent) and aid == "orchestrator"]
        assert any(t.text == "hi there" for t in tokens)

    def test_spawn_agent_creates_thread_and_registry_entry(self):
        engine = make_engine({
            "default": [
                ['{"tool": "spawn_agent", "role": "worker", "system_prompt": "WORKERTAG", "initial_task": "go"}'],
                ["ok done"],
            ],
            "WORKERTAG": [
                ["subagent reply"],
            ],
        })
        executor = MagicMock()
        executor.run = MagicMock(return_value=CommandResult(0, "", ""))
        bus = EventBus()
        q = bus.subscribe()

        orch = Orchestrator(engine=engine, executor=executor, bus=bus)
        orch.start()
        orch.submit_user("spawn a worker please")

        # Wait for orchestrator's second DoneEvent (after spawn result)
        deadline = time.time() + 3
        dones = 0
        subagent_events: list[tuple[str, object]] = []
        while time.time() < deadline:
            try:
                aid, ev = q.get(timeout=0.1)
            except Exception:
                continue
            if aid != "orchestrator":
                subagent_events.append((aid, ev))
            if isinstance(ev, DoneEvent) and aid == "orchestrator":
                dones += 1
                if dones >= 1:
                    break

        # Registry should now have the subagent
        agents = orch.registry.list()
        subagents = [a for a in agents if a.id != "orchestrator"]
        assert len(subagents) == 1
        assert subagents[0].role == "worker"

        # Subagent should also have emitted events
        deadline2 = time.time() + 2
        while time.time() < deadline2 and not any(
            isinstance(ev, DoneEvent) for (_, ev) in subagent_events
        ):
            try:
                aid, ev = q.get(timeout=0.1)
                if aid != "orchestrator":
                    subagent_events.append((aid, ev))
            except Exception:
                continue

        orch.shutdown()
        sub_tokens = [ev for (_, ev) in subagent_events if isinstance(ev, TokenEvent)]
        assert any("subagent reply" in t.text for t in sub_tokens)

    def test_shutdown_stops_all_agents(self):
        engine = make_engine({"default": [["done"]]})
        bus = EventBus()
        orch = Orchestrator(engine=engine, executor=MagicMock(), bus=bus)
        orch.start()
        orch.submit_user("hi")
        time.sleep(0.2)
        orch.shutdown()
        # Orchestrator thread should have stopped
        time.sleep(0.1)
        assert not orch.thread.is_alive()

    def test_orchestrator_registered_in_registry(self):
        bus = EventBus()
        orch = Orchestrator(engine=MagicMock(), executor=MagicMock(), bus=bus)
        assert orch.registry.get("orchestrator") is not None
        assert orch.registry.get("orchestrator").status in (
            AgentStatus.IDLE,
            AgentStatus.WAITING,
            AgentStatus.RUNNING,
        )


class TestOrchestratorConcurrency:
    def test_orchestrator_accepts_input_while_subagent_running(self):
        """Key requirement: submit_user must not block while a subagent is generating."""
        slow_done = threading.Event()
        resume = threading.Event()

        def slow_generate(messages, tools=None):
            # First call: orchestrator spawn decision
            system = messages[0]["content"]
            if "SLOWSUBAGENT" in system:
                slow_done.set()
                resume.wait(timeout=3)
                yield "slow reply"
                return
            # Orchestrator: spawn then done
            if len([m for m in messages if m["role"] == "assistant"]) == 0:
                yield '{"tool": "spawn_agent", "role": "slow", "system_prompt": "SLOWSUBAGENT", "initial_task": "x"}'
                return
            yield "orch ack"

        engine = MagicMock()
        engine.context_window = 128000
        engine.count_tokens = MagicMock(return_value=100)
        engine.generate = MagicMock(side_effect=slow_generate)

        bus = EventBus()
        orch = Orchestrator(engine=engine, executor=MagicMock(), bus=bus)
        orch.start()

        orch.submit_user("spawn slow")
        assert slow_done.wait(timeout=3), "subagent did not start"

        # Subagent is blocked inside generate — now try submitting another user message.
        # This must not block; submit_user returns immediately.
        start = time.monotonic()
        orch.submit_user("ping")
        elapsed = time.monotonic() - start
        assert elapsed < 0.2, f"submit_user blocked for {elapsed}s"

        resume.set()
        time.sleep(0.3)
        orch.shutdown()
