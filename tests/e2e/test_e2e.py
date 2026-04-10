from __future__ import annotations

import pytest

from harnessthing.agent import ContextWindowExhaustedError
from harnessthing.events import DoneEvent, TokenEvent, ToolCallEvent, ToolResultEvent


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
