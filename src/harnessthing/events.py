from __future__ import annotations

from dataclasses import dataclass

from harnessthing.executor import CommandResult


@dataclass
class TokenEvent:
    text: str


@dataclass
class ToolCallEvent:
    command: str


@dataclass
class ToolResultEvent:
    result: CommandResult


@dataclass
class ThinkingEvent:
    text: str


@dataclass
class ErrorEvent:
    message: str


@dataclass
class DoneEvent:
    pass


Event = TokenEvent | ToolCallEvent | ToolResultEvent | ThinkingEvent | ErrorEvent | DoneEvent
