from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from mlxharness.bus import EventBus
from mlxharness.events import DoneEvent
from mlxharness.executor import CommandResult
from mlxharness.orchestrator import Orchestrator


def _build_engine():
    """Mock engine where every generation takes ~50ms, simulating GPU work.

    Orchestrator sequence (N=2 spawns):
      turn 1: spawn_agent worker-1
      turn 2: spawn_agent worker-2
      turn 3: plain "all spawned"
    Each subagent sequence: run 3 shell turns, then reply.
    """
    engine = MagicMock()
    engine.context_window = 128000
    engine.count_tokens = MagicMock(return_value=100)

    state = threading.local()

    orch_turns = iter([
        '{"tool": "spawn_agent", "role": "worker", "system_prompt": "WORKER-A", "initial_task": "do-a"}',
        '{"tool": "spawn_agent", "role": "worker", "system_prompt": "WORKER-B", "initial_task": "do-b"}',
        "all spawned",
    ])

    worker_a_turns = iter([
        '{"tool": "shell", "command": "echo a1"}',
        '{"tool": "shell", "command": "echo a2"}',
        '{"tool": "shell", "command": "echo a3"}',
        "A done",
    ])
    worker_b_turns = iter([
        '{"tool": "shell", "command": "echo b1"}',
        '{"tool": "shell", "command": "echo b2"}',
        '{"tool": "shell", "command": "echo b3"}',
        "B done",
    ])

    lock = threading.Lock()

    def generate(messages, tools=None):
        system = messages[0]["content"]
        with lock:
            if "WORKER-A" in system:
                txt = next(worker_a_turns)
            elif "WORKER-B" in system:
                txt = next(worker_b_turns)
            else:
                txt = next(orch_turns)
        # Simulate some inference work (also gives other threads a chance)
        time.sleep(0.03)
        yield txt

    engine.generate = MagicMock(side_effect=generate)
    return engine


class TestScheduler:
    def test_two_subagents_and_orchestrator_all_complete(self):
        engine = _build_engine()
        executor = MagicMock()
        executor.run = MagicMock(return_value=CommandResult(0, "ok", ""))

        bus = EventBus()
        q = bus.subscribe()

        orch = Orchestrator(engine=engine, executor=executor, bus=bus)
        orch.start()
        orch.submit_user("spawn both workers")

        # Collect events until we see DoneEvent from orchestrator, worker-1, worker-2.
        dones_by_agent: set[str] = set()
        deadline = time.time() + 10
        while time.time() < deadline and len(dones_by_agent) < 3:
            try:
                aid, ev = q.get(timeout=0.2)
            except Exception:
                continue
            if isinstance(ev, DoneEvent):
                dones_by_agent.add(aid)

        orch.shutdown()
        # Orchestrator emits multiple DoneEvents (one per processed user message).
        # We just require at least one Done from orchestrator and each subagent.
        assert "orchestrator" in dones_by_agent
        assert "agent-1" in dones_by_agent
        assert "agent-2" in dones_by_agent

    def test_no_deadlock_under_lock_contention(self):
        """Fast sanity: start orchestrator, spawn two workers, ensure all terminate within budget."""
        engine = _build_engine()
        executor = MagicMock()
        executor.run = MagicMock(return_value=CommandResult(0, "", ""))
        bus = EventBus()
        q = bus.subscribe()

        orch = Orchestrator(engine=engine, executor=executor, bus=bus)
        orch.start()
        orch.submit_user("go")

        start = time.monotonic()
        dones: set[str] = set()
        while time.monotonic() - start < 8 and len(dones) < 3:
            try:
                aid, ev = q.get(timeout=0.1)
            except Exception:
                continue
            if isinstance(ev, DoneEvent):
                dones.add(aid)

        orch.shutdown()
        assert len(dones) == 3, f"not all agents finished: {dones}"
