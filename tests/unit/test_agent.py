from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlxharness.agent import (
    Agent,
    ContextWindowExhaustedError,
    format_tool_result,
    parse_tool_call,
    truncate_result,
)
from mlxharness.events import DoneEvent, ThinkingEvent, TokenEvent, ToolCallEvent, ToolResultEvent
from mlxharness.executor import CommandResult


# --- parse_tool_call ---


class TestParseToolCall:
    def test_native_gemma_format(self):
        text = '<|tool_call>call:shell{command:<|"|>ls -la<|"|>}<tool_call|>'
        result = parse_tool_call(text)
        assert result == {"tool": "shell", "command": "ls -la"}

    def test_native_format_with_thought(self):
        text = '<|channel>thought\nLet me list the files.<channel|><|tool_call>call:shell{command:<|"|>ls<|"|>}<tool_call|>'
        result = parse_tool_call(text)
        assert result is not None
        assert result["command"] == "ls"

    def test_json_fallback(self):
        text = '{"tool": "shell", "command": "ls -la"}'
        result = parse_tool_call(text)
        assert result == {"tool": "shell", "command": "ls -la"}

    def test_json_embedded_in_prose(self):
        text = 'I will list the files.\n{"tool": "shell", "command": "ls"}\nDone.'
        result = parse_tool_call(text)
        assert result is not None
        assert result["command"] == "ls"

    def test_no_tool_call(self):
        text = "Here is the answer: the files are listed above."
        result = parse_tool_call(text)
        assert result is None

    def test_malformed_json(self):
        text = '{"tool": "shell", "command": "ls"'
        result = parse_tool_call(text)
        assert result is None

    def test_wrong_tool_name(self):
        text = '{"tool": "wrong", "command": "ls"}'
        result = parse_tool_call(text)
        assert result is None

    def test_missing_command_key(self):
        text = '{"tool": "shell"}'
        result = parse_tool_call(text)
        assert result is None

    def test_escaped_quotes_in_command(self):
        text = '{"tool": "shell", "command": "echo \\"hello\\""}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["command"] == 'echo "hello"'


# --- truncate_result ---


class TestTruncateResult:
    def test_no_truncation_needed(self):
        result = CommandResult(exit_code=0, stdout="short", stderr="")
        truncated = truncate_result(result, max_chars=4096)
        assert truncated.stdout == "short"
        assert truncated.stderr == ""

    def test_stdout_truncated(self):
        long_output = "x" * 10000
        result = CommandResult(exit_code=0, stdout=long_output, stderr="")
        truncated = truncate_result(result, max_chars=4096)
        assert truncated.stdout.startswith("[truncated, showing last 4096 chars]")
        assert truncated.stdout.endswith("x" * 4096)

    def test_stderr_truncated(self):
        long_err = "e" * 10000
        result = CommandResult(exit_code=1, stdout="", stderr=long_err)
        truncated = truncate_result(result, max_chars=4096)
        assert truncated.stderr.startswith("[truncated, showing last 4096 chars]")

    def test_exit_code_preserved(self):
        result = CommandResult(exit_code=42, stdout="x" * 10000, stderr="")
        truncated = truncate_result(result, max_chars=100)
        assert truncated.exit_code == 42


# --- format_tool_result ---


class TestFormatToolResult:
    def test_format(self):
        result = CommandResult(exit_code=0, stdout="hello\n", stderr="")
        formatted = format_tool_result(result)
        import json
        parsed = json.loads(formatted)
        assert parsed["exit_code"] == 0
        assert parsed["stdout"] == "hello\n"
        assert parsed["stderr"] == ""


# --- Agent ---


def make_mock_engine(responses: list[list[str]], context_window: int = 128000):
    """Create a mock engine that yields pre-defined token sequences.

    Each entry in `responses` is a list of tokens for one generate() call.
    """
    engine = MagicMock()
    engine.context_window = context_window
    engine.count_tokens = MagicMock(return_value=100)
    call_iter = iter(responses)
    engine.generate = MagicMock(side_effect=lambda msgs, tools=None: iter(next(call_iter)))
    return engine


def make_mock_executor(results: list[CommandResult] | None = None):
    executor = MagicMock()
    if results:
        executor.run = MagicMock(side_effect=results)
    else:
        executor.run = MagicMock(
            return_value=CommandResult(exit_code=0, stdout="ok\n", stderr="")
        )
    return executor


class TestAgent:
    def test_plain_text_response(self):
        engine = make_mock_engine([["Hello", " world"]])
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        events = list(agent.step("Hi"))
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        assert len(token_events) == 2
        assert token_events[0].text == "Hello"
        assert token_events[1].text == " world"
        assert any(isinstance(e, DoneEvent) for e in events)
        executor.run.assert_not_called()

    def test_tool_call_then_response(self):
        engine = make_mock_engine([
            ['{"tool": "shell", "command": "ls"}'],
            ["Here are the files."],
        ])
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        events = list(agent.step("List files"))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls) == 1
        assert tool_calls[0].command == "ls"
        assert len(tool_results) == 1
        assert any(isinstance(e, DoneEvent) for e in events)
        executor.run.assert_called_once_with("ls")

    def test_multi_step_tool_use(self):
        engine = make_mock_engine([
            ['{"tool": "shell", "command": "echo hello > test.py"}'],
            ['{"tool": "shell", "command": "python3 test.py"}'],
            ["Done. The output was hello."],
        ])
        executor = make_mock_executor([
            CommandResult(exit_code=0, stdout="", stderr=""),
            CommandResult(exit_code=0, stdout="hello\n", stderr=""),
        ])
        agent = Agent(engine=engine, executor=executor)

        events = list(agent.step("Create and run test.py"))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) == 2

    def test_message_list_construction(self):
        engine = make_mock_engine([["Hello"]])
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        list(agent.step("Hi"))

        # system + user "Hi" + assistant "Hello"
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[1]["content"] == "Hi"
        assert agent.messages[2]["role"] == "assistant"
        assert agent.messages[2]["content"] == "Hello"

    def test_thinking_tokens_yield_thinking_events(self):
        engine = make_mock_engine([
            ["<|channel>", "thought\n", "Let me think", "<channel|>", "Hello!"],
        ])
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        events = list(agent.step("Hi"))
        thinking = [e for e in events if isinstance(e, ThinkingEvent)]
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        assert len(thinking) == 1
        assert thinking[0].text == "Let me think"
        assert len(tokens) == 2
        assert tokens[0].text == "\n"
        assert tokens[1].text == "Hello!"

    def test_tool_call_tokens_suppressed(self):
        engine = make_mock_engine([
            ['<|tool_call>', 'call:shell{command:', '<|"|>', 'ls', '<|"|>', '}', '<tool_call|>'],
            ["Done."],
        ])
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        events = list(agent.step("List files"))
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        # Only "Done." should come through as a token, all tool_call tokens suppressed
        assert len(tokens) == 1
        assert tokens[0].text == "Done."

    def test_context_window_exhausted(self):
        engine = make_mock_engine([])
        engine.count_tokens = MagicMock(return_value=127600)
        engine.context_window = 128000
        executor = make_mock_executor()
        agent = Agent(engine=engine, executor=executor)

        with pytest.raises(ContextWindowExhaustedError):
            list(agent.step("Fill it up"))
