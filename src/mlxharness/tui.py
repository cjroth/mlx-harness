from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Input, Markdown, Rule, Static
from textual.worker import get_current_worker

from mlxharness.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from mlxharness.executor import CommandResult

if TYPE_CHECKING:
    from mlxharness.orchestrator import Orchestrator

CHAT_LOG_ID = "chat-log"
PROMPT_INPUT_ID = "prompt-input"
BUS_WORKER_GROUP = "bus"


def agent_label(agent_id: str) -> str:
    if agent_id == "orchestrator":
        return "orchestrator"
    return agent_id


class UserMessage(Static):
    """A single user input line shown with an accent border."""


class AgentSection(Vertical):
    """Container for all widgets produced by one agent's emissions."""

    def __init__(self, agent_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent_id = agent_id


class AgentLabel(Static):
    """Prefix label rendered at the top of an AgentSection."""

    def __init__(self, agent_id: str, **kwargs) -> None:
        super().__init__(f"{agent_label(agent_id)}:", **kwargs)
        self.agent_id = agent_id


class ResponseText(Markdown):
    """Streaming assistant response, rendered incrementally as Markdown."""


class ThinkingBlock(Collapsible):
    """Collapsible container for streaming thinking content."""


class ThinkingBody(Markdown):
    """Markdown body inside a ThinkingBlock; separate class for CSS targeting."""


class ToolCallBlock(Static):
    """Displays a shell command about to be executed."""


class ToolResultBlock(Static):
    """Displays command stdout/stderr."""


class ErrorBlock(Static):
    """Displays an error message."""


class HarnessApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "mlxharness"

    BINDINGS = [
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+c", "cancel_all", "Cancel all agents", show=False, priority=True),
    ]

    def __init__(self, orchestrator: "Orchestrator") -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self._bus_queue: queue.Queue = orchestrator.bus.subscribe()
        self._chat_log: VerticalScroll | None = None
        self._prompt_input: Input | None = None
        # Per-agent state: the current section's current response/thinking widget.
        self._agent_response: dict[str, ResponseText | None] = {}
        self._agent_thinking: dict[str, ThinkingBlock | None] = {}
        self._agent_thinking_body: dict[str, ThinkingBody | None] = {}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id=CHAT_LOG_ID)
        yield Input(placeholder="Ask the orchestrator…", id=PROMPT_INPUT_ID)

    def on_mount(self) -> None:
        self._chat_log = self.query_one(f"#{CHAT_LOG_ID}", VerticalScroll)
        self._prompt_input = self.query_one(f"#{PROMPT_INPUT_ID}", Input)
        self._chat_log.anchor()
        self._prompt_input.focus()
        self._consume_bus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text == "/quit":
            self.exit()
            return

        assert self._chat_log is not None
        self._chat_log.mount(UserMessage(f"> {text}"))
        # Non-blocking submit — orchestrator may be generating; submit queues the message.
        self.orchestrator.submit_user(text)

    @work(thread=True, exclusive=True, group=BUS_WORKER_GROUP)
    def _consume_bus(self) -> None:
        worker = get_current_worker()
        while not worker.is_cancelled:
            try:
                agent_id, ev = self._bus_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.call_from_thread(self._dispatch_event, agent_id, ev)

    def _make_section(self, agent_id: str) -> AgentSection:
        assert self._chat_log is not None
        section = AgentSection(agent_id=agent_id)
        self._chat_log.mount(section)
        section.mount(AgentLabel(agent_id=agent_id))
        return section

    def _open_section(self, agent_id: str) -> AgentSection:
        """Return the most recent open AgentSection for agent_id, creating one if needed."""
        assert self._chat_log is not None
        sections = list(self._chat_log.query(AgentSection))
        for section in reversed(sections):
            if section.agent_id == agent_id and "closed" not in section.classes:
                return section
        return self._make_section(agent_id)

    def _close_section(self, agent_id: str) -> None:
        sections = list(self._chat_log.query(AgentSection)) if self._chat_log else []
        for section in reversed(sections):
            if section.agent_id == agent_id and "closed" not in section.classes:
                section.add_class("closed")
                return

    def _dispatch_event(self, agent_id: str, event: Event) -> None:
        match event:
            case ThinkingEvent(text=text):
                self._handle_thinking(agent_id, text)
            case TokenEvent(text=text):
                self._handle_token(agent_id, text)
            case ToolCallEvent(command=command):
                self._handle_tool_call(agent_id, command)
            case ToolResultEvent(result=result):
                self._handle_tool_result(agent_id, result)
            case ErrorEvent(message=message):
                self._handle_error(agent_id, message)
            case DoneEvent():
                self._handle_done(agent_id)

    def _handle_thinking(self, agent_id: str, text: str) -> None:
        section = self._open_section(agent_id)
        self._agent_response[agent_id] = None
        thinking = self._agent_thinking.get(agent_id)
        body = self._agent_thinking_body.get(agent_id)
        if thinking is None or body is None:
            body = ThinkingBody("")
            thinking = ThinkingBlock(body, title="Thinking…", collapsed=False)
            section.mount(thinking)
            self._agent_thinking[agent_id] = thinking
            self._agent_thinking_body[agent_id] = body
        body.append(text)

    def _handle_token(self, agent_id: str, text: str) -> None:
        section = self._open_section(agent_id)
        self._collapse_thinking(agent_id)
        response = self._agent_response.get(agent_id)
        if response is None:
            response = ResponseText("")
            section.mount(response)
            self._agent_response[agent_id] = response
        response.append(text)

    def _handle_tool_call(self, agent_id: str, command: str) -> None:
        section = self._open_section(agent_id)
        self._collapse_thinking(agent_id)
        self._agent_response[agent_id] = None
        section.mount(ToolCallBlock(f"$ {command}"))

    def _handle_tool_result(self, agent_id: str, result: CommandResult) -> None:
        section = self._open_section(agent_id)
        content = _format_tool_result(result)
        section.mount(ToolResultBlock(content))

    def _handle_error(self, agent_id: str, message: str) -> None:
        section = self._open_section(agent_id)
        self._collapse_thinking(agent_id)
        self._agent_response[agent_id] = None
        section.mount(ErrorBlock(f"Error: {message}"))

    def _handle_done(self, agent_id: str) -> None:
        self._collapse_thinking(agent_id)
        self._agent_response[agent_id] = None
        section = self._open_section(agent_id)
        section.mount(Rule(classes="turn-separator"))
        self._close_section(agent_id)

    def action_cancel_all(self) -> None:
        self.orchestrator.shutdown()
        self.exit()

    def _collapse_thinking(self, agent_id: str) -> None:
        thinking = self._agent_thinking.get(agent_id)
        if thinking is not None:
            thinking.collapsed = True
            self._agent_thinking[agent_id] = None
            self._agent_thinking_body[agent_id] = None


def _format_tool_result(result: CommandResult) -> Text:
    content = Text()
    stdout = result.stdout.rstrip("\n")
    stderr = result.stderr.rstrip("\n")
    if stdout:
        content.append(stdout, style="dim")
    if stderr:
        if stdout:
            content.append("\n")
        content.append(stderr, style="yellow")
    if not stdout and not stderr:
        content.append(f"(exit {result.exit_code})", style="dim italic")
    return content
