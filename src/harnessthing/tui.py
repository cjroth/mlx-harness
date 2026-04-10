from __future__ import annotations

import readline  # noqa: F401 — enables readline support for input()
import sys

from harnessthing.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# ANSI escape codes
DIM = "\033[2m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def render_event(event: Event) -> None:
    match event:
        case TokenEvent(text=text):
            sys.stdout.write(text)
            sys.stdout.flush()

        case ThinkingEvent(text=text):
            sys.stdout.write(f"{DIM}{text}{RESET}")
            sys.stdout.flush()

        case ToolCallEvent(command=command):
            sys.stdout.write(f"\n{DIM}  $ {command}{RESET}\n")
            sys.stdout.flush()

        case ToolResultEvent(result=result):
            if result.stdout:
                for line in result.stdout.splitlines():
                    sys.stdout.write(f"{DIM}  {line}{RESET}\n")
            if result.stderr:
                for line in result.stderr.splitlines():
                    sys.stdout.write(f"{DIM}{YELLOW}  {line}{RESET}\n")
            sys.stdout.flush()

        case ErrorEvent(message=message):
            sys.stdout.write(f"{RED}{message}{RESET}\n")
            sys.stdout.flush()

        case DoneEvent():
            sys.stdout.write("\n")
            sys.stdout.flush()


def prompt() -> str | None:
    try:
        return input("> ")
    except (EOFError, KeyboardInterrupt):
        return None


def run_loop(agent) -> None:
    while True:
        user_input = prompt()

        if user_input is None:
            print()
            break

        if user_input.strip() == "/quit":
            break

        if not user_input.strip():
            continue

        try:
            for event in agent.step(user_input):
                render_event(event)
        except KeyboardInterrupt:
            sys.stdout.write(f"\n{DIM}(interrupted){RESET}\n")
        except Exception as e:
            render_event(ErrorEvent(message=str(e)))
