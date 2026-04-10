from __future__ import annotations

from unittest.mock import MagicMock

from textual.widgets import Input

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
    AssistantTurn,
    ErrorBlock,
    HarnessApp,
    ResponseText,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserMessage,
)


def make_mock_agent(events: list):
    """Create a mock agent whose step() yields the given events."""
    agent = MagicMock()
    agent.step = MagicMock(side_effect=lambda text: iter(events))
    return agent


async def submit_input(pilot, text: str) -> None:
    inp = pilot.app.query_one("#prompt-input", Input)
    inp.value = text
    await pilot.press("enter")
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


class TestHarnessApp:
    async def test_compose_mounts_input_and_chat_log(self):
        app = HarnessApp(make_mock_agent([]))
        async with app.run_test() as pilot:
            assert app.query_one("#chat-log")
            assert app.query_one("#prompt-input", Input)

    async def test_user_message_shown_on_submit(self):
        agent = make_mock_agent([DoneEvent()])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hello")
            msgs = app.query(UserMessage)
            assert len(msgs) == 1

    async def test_token_streaming_creates_response_widget(self):
        agent = make_mock_agent([
            TokenEvent(text="Hello"),
            TokenEvent(text=" world"),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hi")
            responses = app.query(ResponseText)
            assert len(responses) >= 1

    async def test_thinking_block_mounted_and_collapsed_on_token(self):
        agent = make_mock_agent([
            ThinkingEvent(text="Let me think"),
            TokenEvent(text="Answer"),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hi")
            thinking_blocks = app.query(ThinkingBlock)
            assert len(thinking_blocks) == 1
            # Should be collapsed after tokens arrived
            assert thinking_blocks.first().collapsed is True

    async def test_tool_call_and_result_displayed(self):
        agent = make_mock_agent([
            ToolCallEvent(command="ls -la"),
            ToolResultEvent(result=CommandResult(0, "file1\n", "")),
            TokenEvent(text="Done"),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "list files")
            assert len(app.query(ToolCallBlock)) == 1
            assert len(app.query(ToolResultBlock)) == 1

    async def test_error_event_mounts_error_block(self):
        agent = make_mock_agent([
            ErrorEvent(message="something broke"),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "break")
            errors = app.query(ErrorBlock)
            assert len(errors) == 1

    async def test_multi_tool_turn_creates_two_tool_blocks(self):
        agent = make_mock_agent([
            ToolCallEvent(command="pwd"),
            ToolResultEvent(result=CommandResult(0, "/tmp\n", "")),
            ToolCallEvent(command="ls"),
            ToolResultEvent(result=CommandResult(0, "file\n", "")),
            TokenEvent(text="Done."),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "do stuff")
            assert len(app.query(ToolCallBlock)) == 2
            assert len(app.query(ToolResultBlock)) == 2

    async def test_input_reenabled_after_done(self):
        agent = make_mock_agent([
            TokenEvent(text="Hi"),
            DoneEvent(),
        ])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "hello")
            inp = app.query_one("#prompt-input", Input)
            assert inp.disabled is False

    async def test_quit_command_exits(self):
        agent = make_mock_agent([])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt-input", Input)
            inp.value = "/quit"
            await pilot.press("enter")
            await pilot.pause()
            assert not app.is_running

    async def test_empty_input_ignored(self):
        agent = make_mock_agent([])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            inp = app.query_one("#prompt-input", Input)
            inp.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.query(UserMessage)) == 0
            agent.step.assert_not_called()

    async def test_assistant_turn_created_per_submit(self):
        agent = make_mock_agent([DoneEvent()])
        app = HarnessApp(agent)
        async with app.run_test() as pilot:
            await submit_input(pilot, "first")
            await submit_input(pilot, "second")
            turns = app.query(AssistantTurn)
            assert len(turns) == 2
