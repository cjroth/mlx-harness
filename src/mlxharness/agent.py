from __future__ import annotations

import json
import re
from typing import Iterator, Protocol

from mlxharness.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from mlxharness.executor import CommandResult, Executor

CONTEXT_HEADROOM = 512
MAX_TOOL_RESULT_CHARS = 4096


class ContextWindowExhaustedError(Exception):
    pass


class EngineProtocol(Protocol):
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[str]: ...
    def count_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int: ...

    @property
    def context_window(self) -> int: ...


def parse_tool_call(text: str) -> dict | None:
    # Gemma 4 native format: <|tool_call>call:shell{command:<|"|>...<|"|>}<tool_call|>
    match = re.search(r'<\|tool_call>call:shell\{command:<\|"\|>(.*?)<\|"\|>\}<tool_call\|>', text, re.DOTALL)
    if match:
        return {"tool": "shell", "command": match.group(1)}

    # Fallback: JSON format {"tool": "shell", "command": "..."}
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
            if obj.get("tool") == "shell" and "command" in obj:
                return obj
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def truncate_result(result: CommandResult, max_chars: int = MAX_TOOL_RESULT_CHARS) -> CommandResult:
    stdout = result.stdout
    stderr = result.stderr

    if len(stdout) > max_chars:
        stdout = f"[truncated, showing last {max_chars} chars]\n" + stdout[-max_chars:]
    if len(stderr) > max_chars:
        stderr = f"[truncated, showing last {max_chars} chars]\n" + stderr[-max_chars:]

    return CommandResult(
        exit_code=result.exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def format_tool_result(result: CommandResult) -> str:
    return json.dumps({
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    })


class Agent:
    def __init__(self, engine: EngineProtocol, executor: Executor):
        self.engine = engine
        self.executor = executor
        self.messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
        ]

    def _system_prompt(self) -> str:
        from mlxharness.engine import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _tools(self) -> list[dict]:
        from mlxharness.engine import TOOLS
        return TOOLS

    def _check_context_window(self) -> None:
        token_count = self.engine.count_tokens(self.messages, tools=self._tools())
        limit = self.engine.context_window
        if token_count + CONTEXT_HEADROOM >= limit:
            raise ContextWindowExhaustedError(
                f"Context window exhausted: {token_count} tokens used, "
                f"limit is {limit} (need {CONTEXT_HEADROOM} headroom)"
            )

    def _stream_tokens(self, messages: list[dict]) -> Iterator[tuple[str, Event | None]]:
        """Stream tokens from the engine, yielding (raw_token, event_or_none).

        Handles Gemma 4 special tokens:
        - <|channel>thought\\n...<channel|> → ThinkingEvent (dim display)
        - <|tool_call>...<tool_call|> → suppressed (parsed after generation)
        - <|"|> → suppressed
        """
        state = "normal"  # normal | thinking_header | thinking | tool_call
        for token in self.engine.generate(messages, tools=self._tools()):
            if "<|channel>" in token:
                state = "thinking_header"
                yield token, None
            elif "<channel|>" in token:
                state = "normal"
                yield token, TokenEvent(text="\n")
            elif state == "thinking_header":
                # Skip the "thought\n" channel name line
                state = "thinking"
                yield token, None
            elif state == "thinking":
                yield token, ThinkingEvent(text=token)
            elif "<|tool_call>" in token:
                state = "tool_call"
                yield token, None
            elif "<tool_call|>" in token:
                state = "normal"
                yield token, None
            elif state == "tool_call" or "<|\"|>" in token:
                yield token, None
            else:
                yield token, TokenEvent(text=token)

    def step(self, user_input: str) -> Iterator[Event]:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            self._check_context_window()

            response_text = ""
            for token, event in self._stream_tokens(self.messages):
                response_text += token
                if event is not None:
                    yield event

            tool_call = parse_tool_call(response_text)

            if tool_call is None:
                self.messages.append({"role": "assistant", "content": response_text})
                yield DoneEvent()
                break

            yield ToolCallEvent(command=tool_call["command"])

            result = self.executor.run(tool_call["command"])
            result = truncate_result(result, max_chars=MAX_TOOL_RESULT_CHARS)

            yield ToolResultEvent(result=result)

            self.messages.append({"role": "assistant", "content": response_text})
            self.messages.append({"role": "tool", "content": format_tool_result(result)})
