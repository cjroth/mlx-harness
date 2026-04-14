from __future__ import annotations

import time

import pytest

from mlxharness.agent import ContextWindowExhaustedError
from mlxharness.events import DoneEvent, TokenEvent, ToolCallEvent, ToolResultEvent


pytestmark = pytest.mark.e2e


class TestEndToEnd:

    def test_simple_response(self, agent):
        """Model responds without using tools."""
        events = list(agent.step("Say hello in exactly 3 words."))
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        assert len(tokens) > 0
        assert any(isinstance(e, DoneEvent) for e in events)

    def test_tool_call_and_result(self, agent):
        """Model uses shell tool and responds with result."""
        events = list(agent.step("What files are in /workspace? Use the shell tool."))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1

    def test_multi_step_tool_use(self, agent):
        """Model chains multiple commands."""
        events = list(agent.step(
            "Create a file called hello.py in /workspace that prints 'hello world', "
            "then run it and tell me the output."
        ))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2  # at least write + run

    def test_context_window_error(self, agent):
        """Hard error when context is exhausted."""
        # Stuff the context with large messages directly rather than
        # waiting for model generation to fill 128K tokens.
        big_block = "x" * 100000
        for _ in range(10):
            agent.messages.append({"role": "user", "content": big_block})
            agent.messages.append({"role": "assistant", "content": big_block})

        with pytest.raises(ContextWindowExhaustedError):
            list(agent.step("one more"))


class TestOrchestratorE2E:
    def _collect_until_done(self, q, expected_ids: set[str], budget: float = 120.0):
        """Collect events from bus until a DoneEvent has arrived from each expected agent."""
        dones: set[str] = set()
        all_events: list[tuple[str, object]] = []
        deadline = time.time() + budget
        while time.time() < deadline and not expected_ids.issubset(dones):
            try:
                aid, ev = q.get(timeout=0.5)
            except Exception:
                continue
            all_events.append((aid, ev))
            if isinstance(ev, DoneEvent):
                dones.add(aid)
        return all_events, dones

    def test_orchestrator_responds_to_user(self, orchestrator):
        q = orchestrator.bus.subscribe()
        orchestrator.submit_user("Say hello in three words.")
        events, dones = self._collect_until_done(q, {"orchestrator"}, budget=60)
        assert "orchestrator" in dones
        tokens = [ev for (aid, ev) in events if aid == "orchestrator" and isinstance(ev, TokenEvent)]
        assert len(tokens) > 0

    def test_orchestrator_spawns_subagent_and_reports(self, orchestrator):
        q = orchestrator.bus.subscribe()
        orchestrator.submit_user(
            "Spawn a subagent with role 'lister' and system_prompt "
            "'You list files.' and initial_task 'Run `ls /workspace` and tell me the result.' "
            "Then reply to me with 'spawned' once you have done so."
        )
        # Orchestrator should emit at least one Done, and a subagent (agent-1) should be registered.
        deadline = time.time() + 120
        while time.time() < deadline:
            agents = orchestrator.registry.list()
            if any(a.id.startswith("agent-") for a in agents):
                break
            time.sleep(0.2)
        subagents = [a for a in orchestrator.registry.list() if a.id.startswith("agent-")]
        assert len(subagents) >= 1

        # Wait for the subagent to emit a DoneEvent as well.
        subagent_id = subagents[0].id
        _, dones = self._collect_until_done(q, {subagent_id}, budget=120)
        assert subagent_id in dones

    def test_orchestrator_accepts_message_while_subagent_running(self, orchestrator):
        # Submit an initial message that spawns a subagent.
        orchestrator.submit_user(
            "Spawn a subagent with role 'slow' and system_prompt 'You run commands slowly.' "
            "and initial_task 'Run `sleep 2 && echo done` and reply with the output.'"
        )
        # Immediately submit a second message — this must not block.
        start = time.monotonic()
        orchestrator.submit_user("Please confirm you are still listening.")
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"submit_user blocked for {elapsed}s"
