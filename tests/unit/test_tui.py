from __future__ import annotations

import asyncio

from textual.widgets import Input

from mlxharness.bus import EventBus
from mlxharness.events import (
    DoneEvent,
    ErrorEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from mlxharness.executor import CommandResult
from mlxharness.tui import (
    AgentLabel,
    AgentSection,
    ErrorBlock,
    HarnessApp,
    ResponseText,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserMessage,
)


class StubOrchestrator:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.submitted: list[str] = []

    def submit_user(self, text: str) -> None:
        self.submitted.append(text)


async def _settle(app, q_check=None, iterations: int = 40) -> None:
    """Give the bus worker time to drain."""
    for _ in range(iterations):
        await asyncio.sleep(0.02)
        if q_check is not None and q_check():
            return


async def submit_input(pilot, text: str) -> None:
    inp = pilot.app.query_one("#prompt-input", Input)
    inp.value = text
    await pilot.press("enter")
    await pilot.pause()


class TestHarnessApp:
    async def test_compose_mounts_input_and_chat_log(self):
        app = HarnessApp(StubOrchestrator())
        async with app.run_test():
            assert app.query_one("#chat-log")
            assert app.query_one("#prompt-input", Input)

    async def test_user_message_shown_on_submit(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hello")
            msgs = app.query(UserMessage)
            assert len(msgs) == 1
            assert orch.submitted == ["hello"]

    async def test_submit_routes_to_orchestrator(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            await submit_input(pilot, "first")
            await submit_input(pilot, "second")
            assert orch.submitted == ["first", "second"]

    async def test_input_always_enabled(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hi")
            inp = app.query_one("#prompt-input", Input)
            assert inp.disabled is False

    async def test_token_event_creates_response_widget_in_agent_section(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", TokenEvent(text="Hi"))
            orch.bus.publish("orchestrator", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(ResponseText)) > 0)
            responses = app.query(ResponseText)
            assert len(responses) >= 1

    async def test_events_from_two_agents_render_separate_sections(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", TokenEvent(text="from orch"))
            orch.bus.publish("orchestrator", DoneEvent())
            orch.bus.publish("agent-1", TokenEvent(text="from sub"))
            orch.bus.publish("agent-1", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(AgentSection)) >= 2)
            sections = app.query(AgentSection)
            assert len(sections) >= 2
            ids = {s.agent_id for s in sections}
            assert "orchestrator" in ids
            assert "agent-1" in ids

    async def test_agent_label_prefix_rendered(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("agent-1", TokenEvent(text="hello"))
            orch.bus.publish("agent-1", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(AgentLabel)) > 0)
            labels = app.query(AgentLabel)
            assert any(l.agent_id == "agent-1" for l in labels)

    async def test_thinking_block_mounts_and_collapses(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", ThinkingEvent(text="reasoning"))
            orch.bus.publish("orchestrator", TokenEvent(text="answer"))
            orch.bus.publish("orchestrator", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(ThinkingBlock)) > 0)
            blocks = app.query(ThinkingBlock)
            assert len(blocks) == 1
            assert blocks.first().collapsed is True

    async def test_tool_call_and_result_displayed(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", ToolCallEvent(command="ls -la"))
            orch.bus.publish("orchestrator", ToolResultEvent(result=CommandResult(0, "out", "")))
            orch.bus.publish("orchestrator", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(ToolCallBlock)) > 0)
            assert len(app.query(ToolCallBlock)) == 1
            assert len(app.query(ToolResultBlock)) == 1

    async def test_error_event_mounts_error_block(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", ErrorEvent(message="broken"))
            orch.bus.publish("orchestrator", DoneEvent())
            await _settle(app, q_check=lambda: len(app.query(ErrorBlock)) > 0)
            assert len(app.query(ErrorBlock)) == 1

    async def test_quit_command_exits(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt-input", Input)
            inp.value = "/quit"
            await pilot.press("enter")
            await pilot.pause()
            assert not app.is_running

    async def test_empty_input_ignored(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt-input", Input)
            inp.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.query(UserMessage)) == 0
            assert orch.submitted == []

    async def test_subsequent_turn_creates_new_agent_section_for_same_agent(self):
        orch = StubOrchestrator()
        app = HarnessApp(orch)
        async with app.run_test() as pilot:
            orch.bus.publish("orchestrator", TokenEvent(text="turn1"))
            orch.bus.publish("orchestrator", DoneEvent())
            orch.bus.publish("orchestrator", TokenEvent(text="turn2"))
            orch.bus.publish("orchestrator", DoneEvent())
            await _settle(app, q_check=lambda: len(
                [s for s in app.query(AgentSection) if s.agent_id == "orchestrator"]
            ) >= 2)
            sections = [s for s in app.query(AgentSection) if s.agent_id == "orchestrator"]
            assert len(sections) == 2
