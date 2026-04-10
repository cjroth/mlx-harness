from __future__ import annotations

import io
from unittest.mock import patch

from harnessthing.events import DoneEvent, ErrorEvent, ThinkingEvent, TokenEvent, ToolCallEvent, ToolResultEvent
from harnessthing.executor import CommandResult
from harnessthing.tui import render_event

DIM = "\033[2m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


class TestRenderEvent:
    def test_token_event(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(TokenEvent(text="hello"))
            assert mock_stdout.getvalue() == "hello"

    def test_thinking_event(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(ThinkingEvent(text="Let me think..."))
            output = mock_stdout.getvalue()
            assert "Let me think..." in output
            assert DIM in output

    def test_tool_call_event(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(ToolCallEvent(command="ls -la"))
            output = mock_stdout.getvalue()
            assert "$ ls -la" in output
            assert DIM in output

    def test_tool_result_stdout(self):
        result = CommandResult(exit_code=0, stdout="file1\nfile2\n", stderr="")
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(ToolResultEvent(result=result))
            output = mock_stdout.getvalue()
            assert "file1" in output
            assert "file2" in output
            assert DIM in output

    def test_tool_result_stderr(self):
        result = CommandResult(exit_code=1, stdout="", stderr="warning\n")
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(ToolResultEvent(result=result))
            output = mock_stdout.getvalue()
            assert "warning" in output
            assert YELLOW in output

    def test_error_event(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(ErrorEvent(message="something broke"))
            output = mock_stdout.getvalue()
            assert "something broke" in output
            assert RED in output

    def test_done_event(self):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_event(DoneEvent())
            assert mock_stdout.getvalue() == "\n"
