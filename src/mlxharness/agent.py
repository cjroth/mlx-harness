from __future__ import annotations

import contextlib
import json
import re
import threading
from typing import Callable, Iterator, Protocol

from mlxharness.a2a import A2AMessage, AgentStatus, Inbox
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
from mlxharness.tools import ShellTool, ToolContext, ToolRegistry

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


def _default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ShellTool())
    return reg


class Agent:
    def __init__(
        self,
        engine: EngineProtocol,
        executor: Executor | None = None,
        *,
        agent_id: str = "agent",
        tool_registry: ToolRegistry | None = None,
        inbox: Inbox | None = None,
        outbox: Inbox | None = None,
        gen_lock: threading.Lock | None = None,
        event_sink: Callable[[Event], None] | None = None,
        system_prompt: str | None = None,
        parent_id: str | None = None,
        agent_registry=None,
        spawn_agent_fn=None,
        allowed_message_targets: set[str] | None = None,
    ):
        self.engine = engine
        self.executor = executor
        self.agent_id = agent_id
        self.tool_registry = tool_registry if tool_registry is not None else _default_registry()
        self.inbox = inbox
        self.outbox = outbox
        self.gen_lock = gen_lock
        self.event_sink = event_sink
        self.parent_id = parent_id
        self.agent_registry = agent_registry
        self.spawn_agent_fn = spawn_agent_fn
        self.allowed_message_targets = allowed_message_targets
        self._stop = threading.Event()
        self.status: AgentStatus = AgentStatus.IDLE
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt or self._default_system_prompt()},
        ]

    @staticmethod
    def _default_system_prompt() -> str:
        from mlxharness.engine import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _tools(self) -> list[dict]:
        return self.tool_registry.schemas()

    def _check_context_window(self) -> None:
        token_count = self.engine.count_tokens(self.messages, tools=self._tools())
        limit = self.engine.context_window
        if token_count + CONTEXT_HEADROOM >= limit:
            raise ContextWindowExhaustedError(
                f"Context window exhausted: {token_count} tokens used, "
                f"limit is {limit} (need {CONTEXT_HEADROOM} headroom)"
            )

    @contextlib.contextmanager
    def _hold_gen_lock(self):
        if self.gen_lock is None:
            yield
        else:
            self.gen_lock.acquire()
            try:
                yield
            finally:
                self.gen_lock.release()

    def _stream_tokens(self, messages: list[dict]) -> Iterator[tuple[str, Event | None]]:
        """Stream tokens from the engine, yielding (raw_token, event_or_none).

        Handles Gemma 4 special tokens:
        - <|channel>thought\\n...<channel|> → ThinkingEvent (dim display)
        - <|tool_call>...<tool_call|> → suppressed (parsed after generation)
        - <|"|> → suppressed
        """
        state = "normal"
        with self._hold_gen_lock():
            for token in self.engine.generate(messages, tools=self._tools()):
                if self._stop.is_set():
                    break
                if "<|channel>" in token:
                    state = "thinking_header"
                    yield token, None
                elif "<channel|>" in token:
                    state = "normal"
                    yield token, TokenEvent(text="\n")
                elif state == "thinking_header":
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

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            executor=self.executor,
            agent_registry=self.agent_registry,
            caller_id=self.agent_id,
            spawn_agent=self.spawn_agent_fn,
            parent_id=self.parent_id,
            allowed_targets=self.allowed_message_targets,
        )

    def step(self, user_input: str) -> Iterator[Event]:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            if self._stop.is_set():
                return
            self._check_context_window()

            response_text = ""
            for token, event in self._stream_tokens(self.messages):
                response_text += token
                if event is not None:
                    self._emit(event)
                    yield event

            match = self.tool_registry.parse(response_text)

            if match is None:
                self.messages.append({"role": "assistant", "content": response_text})
                done = DoneEvent()
                self._emit(done)
                yield done
                break

            tool, args = match
            display = self._format_tool_call_display(tool.name, args)
            call_ev = ToolCallEvent(command=display)
            self._emit(call_ev)
            yield call_ev

            result = tool.execute(args, self._tool_context())
            tool_output, result_event = self._run_tool_result(tool.name, result)

            self._emit(result_event)
            yield result_event

            self.messages.append({"role": "assistant", "content": response_text})
            self.messages.append({"role": "tool", "content": tool_output})

    @staticmethod
    def _format_tool_call_display(name: str, args: dict) -> str:
        if name == "shell":
            return args.get("command", "")
        return f"{name}({json.dumps(args)})"

    def _run_tool_result(self, name: str, result) -> tuple[str, Event]:
        if isinstance(result, CommandResult):
            truncated = truncate_result(result, max_chars=MAX_TOOL_RESULT_CHARS)
            return format_tool_result(truncated), ToolResultEvent(result=truncated)
        # Non-shell tool: result is a plain string; wrap as a CommandResult for the TUI.
        text = str(result)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = f"[truncated, showing last {MAX_TOOL_RESULT_CHARS} chars]\n" + text[-MAX_TOOL_RESULT_CHARS:]
        pseudo = CommandResult(exit_code=0, stdout=text, stderr="")
        return text, ToolResultEvent(result=pseudo)

    # --- threaded mode ---

    def stop(self) -> None:
        self._stop.set()
        if self.inbox is not None:
            # Push a sentinel so a blocking inbox.get wakes up.
            self.inbox.put(A2AMessage(from_id="_system", to_id=self.agent_id, content=""))

    def _emit(self, ev: Event) -> None:
        if self.event_sink is not None:
            self.event_sink(ev)

    def run_forever(self) -> None:
        """Thread entrypoint: pull inbox messages and run them through step()."""
        if self.inbox is None:
            raise RuntimeError("Agent.run_forever requires an inbox")
        self.status = AgentStatus.IDLE
        while not self._stop.is_set():
            self.status = AgentStatus.WAITING
            msg = self.inbox.get(timeout=0.1)
            if msg is None:
                continue
            if self._stop.is_set():
                return
            if msg.from_id == "_system" and msg.content == "":
                continue
            self.status = AgentStatus.RUNNING
            try:
                for _ in self.step(msg.content):
                    if self._stop.is_set():
                        return
                # After a step finishes, notify parent (if any) with the last assistant text.
                if self.outbox is not None and self.parent_id is not None:
                    last_assistant = next(
                        (m["content"] for m in reversed(self.messages) if m["role"] == "assistant"),
                        "",
                    )
                    self.outbox.put(A2AMessage(
                        from_id=self.agent_id,
                        to_id=self.parent_id,
                        content=last_assistant,
                    ))
            except ContextWindowExhaustedError as e:
                self.status = AgentStatus.ERROR
                self._emit(ErrorEvent(message=str(e)))
                self._emit(DoneEvent())
                return
            except Exception as e:
                self.status = AgentStatus.ERROR
                self._emit(ErrorEvent(message=str(e)))
                self._emit(DoneEvent())
                continue
        self.status = AgentStatus.DONE
