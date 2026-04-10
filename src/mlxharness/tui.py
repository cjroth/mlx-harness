from __future__ import annotations

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
    from mlxharness.agent import Agent

CHAT_LOG_ID = "chat-log"
PROMPT_INPUT_ID = "prompt-input"
AGENT_WORKER_GROUP = "agent"


class UserMessage(Static):
    """A single user input line shown with an accent border."""


class AssistantTurn(Vertical):
    """Container for all widgets produced by one agent.step() call."""


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
        Binding("ctrl+c", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
    ]

    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self.agent = agent
        self._chat_log: VerticalScroll | None = None
        self._prompt_input: Input | None = None
        self._current_turn: AssistantTurn | None = None
        self._current_response: ResponseText | None = None
        self._current_thinking: ThinkingBlock | None = None
        self._generating: bool = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id=CHAT_LOG_ID)
        yield Input(placeholder="Ask me anything…", id=PROMPT_INPUT_ID)

    def on_mount(self) -> None:
        self._chat_log = self.query_one(f"#{CHAT_LOG_ID}", VerticalScroll)
        self._prompt_input = self.query_one(f"#{PROMPT_INPUT_ID}", Input)
        self._chat_log.anchor()
        self._prompt_input.focus()

    @property
    def is_generating(self) -> bool:
        return self._generating

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text == "/quit":
            self.exit()
            return

        if self._generating:
            return

        assert self._chat_log is not None
        self._chat_log.mount(UserMessage(f"> {text}"))

        turn = AssistantTurn()
        self._chat_log.mount(turn)
        self._current_turn = turn
        self._current_response = None
        self._current_thinking = None
        self._generating = True
        event.input.disabled = True

        self.run_agent(text)

    @work(thread=True, exclusive=True, group=AGENT_WORKER_GROUP)
    def run_agent(self, user_input: str) -> None:
        worker = get_current_worker()
        try:
            for ev in self.agent.step(user_input):
                if worker.is_cancelled:
                    return
                self.call_from_thread(self._dispatch_event, ev)
        except Exception as e:
            if not worker.is_cancelled:
                self.call_from_thread(self._dispatch_event, ErrorEvent(message=str(e)))
                self.call_from_thread(self._dispatch_event, DoneEvent())

    def _dispatch_event(self, event: Event) -> None:
        match event:
            case ThinkingEvent(text=text):
                self._handle_thinking(text)
            case TokenEvent(text=text):
                self._handle_token(text)
            case ToolCallEvent(command=command):
                self._handle_tool_call(command)
            case ToolResultEvent(result=result):
                self._handle_tool_result(result)
            case ErrorEvent(message=message):
                self._handle_error(message)
            case DoneEvent():
                self._handle_done()

    def _handle_thinking(self, text: str) -> None:
        assert self._current_turn is not None
        if self._current_response is not None:
            self._current_response = None
        if self._current_thinking is None:
            body = ThinkingBody("")
            block = ThinkingBlock(body, title="Thinking…", collapsed=False)
            self._current_turn.mount(block)
            self._current_thinking = block
        body = self._current_thinking.query_one(ThinkingBody)
        body.append(text)

    def _handle_token(self, text: str) -> None:
        assert self._current_turn is not None
        self._collapse_thinking()
        if self._current_response is None:
            self._current_response = ResponseText("")
            self._current_turn.mount(self._current_response)
        self._current_response.append(text)

    def _handle_tool_call(self, command: str) -> None:
        assert self._current_turn is not None
        self._collapse_thinking()
        self._current_response = None
        self._current_turn.mount(ToolCallBlock(f"$ {command}"))

    def _handle_tool_result(self, result: CommandResult) -> None:
        assert self._current_turn is not None
        content = _format_tool_result(result)
        self._current_turn.mount(ToolResultBlock(content))

    def _handle_error(self, message: str) -> None:
        assert self._current_turn is not None
        self._collapse_thinking()
        self._current_response = None
        self._current_turn.mount(ErrorBlock(f"Error: {message}"))

    def _handle_done(self) -> None:
        self._collapse_thinking()
        self._current_response = None
        if self._current_turn is not None:
            self._current_turn.mount(Rule(classes="turn-separator"))
        self._finish_generation()

    def _collapse_thinking(self) -> None:
        if self._current_thinking is not None:
            self._current_thinking.collapsed = True
            self._current_thinking = None

    def _finish_generation(self) -> None:
        self._generating = False
        assert self._prompt_input is not None
        self._prompt_input.disabled = False
        self._prompt_input.focus()

    def action_interrupt(self) -> None:
        if not self._generating:
            return
        self.workers.cancel_group(self, AGENT_WORKER_GROUP)
        if self._current_turn is not None:
            self._current_turn.mount(Static("(interrupted)", classes="interrupted"))
        self._collapse_thinking()
        self._current_response = None
        self._finish_generation()


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
