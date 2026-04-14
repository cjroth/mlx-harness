from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from mlxharness.a2a import A2AMessage, AgentRegistry
from mlxharness.executor import CommandResult, Executor


SpawnAgentFn = Callable[..., str]


@dataclass
class ToolContext:
    executor: Executor | None = None
    agent_registry: AgentRegistry | None = None
    caller_id: str = ""
    spawn_agent: SpawnAgentFn | None = None
    parent_id: str | None = None
    allowed_targets: set[str] | None = None  # for subagent send_message restriction


class Tool(Protocol):
    name: str

    def parse_text(self, text: str) -> dict | None: ...
    def execute(self, args: dict, ctx: ToolContext) -> Any: ...
    def schema(self) -> dict: ...


def _find_json_object(text: str, tool_name: str) -> dict | None:
    """Scan `text` for a JSON object whose "tool" field equals `tool_name`."""
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace = text.find("{", idx)
        if brace < 0:
            return None
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict) and obj.get("tool") == tool_name:
            args = {k: v for k, v in obj.items() if k != "tool"}
            return args
        idx = brace + end
    return None


class ShellTool:
    name = "shell"

    _native_re = re.compile(
        r'<\|tool_call>call:shell\{command:<\|"\|>(.*?)<\|"\|>\}<tool_call\|>',
        re.DOTALL,
    )

    def parse_text(self, text: str) -> dict | None:
        m = self._native_re.search(text)
        if m:
            return {"command": m.group(1)}
        return _find_json_object(text, self.name)

    def execute(self, args: dict, ctx: ToolContext) -> CommandResult:
        if ctx.executor is None:
            raise RuntimeError("shell tool requires an executor")
        return ctx.executor.run(args["command"])

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        }
                    },
                    "required": ["command"],
                },
            },
        }


class SpawnAgentTool:
    name = "spawn_agent"

    def parse_text(self, text: str) -> dict | None:
        return _find_json_object(text, self.name)

    def execute(self, args: dict, ctx: ToolContext) -> str:
        if ctx.spawn_agent is None:
            return "error: spawn_agent not available to this agent"
        role = args.get("role", "worker")
        system_prompt = args.get("system_prompt", "")
        initial_task = args.get("initial_task")
        model = args.get("model")
        agent_id = ctx.spawn_agent(
            role=role,
            system_prompt=system_prompt,
            initial_task=initial_task,
            model=model,
        )
        return f"spawned {agent_id}"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Spawn a child subagent that runs concurrently.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "description": "Short role name"},
                        "system_prompt": {"type": "string", "description": "System prompt for the child"},
                        "initial_task": {"type": "string", "description": "First user message to send the child"},
                        "model": {"type": "string", "description": "Model name (optional; defaults to shared engine)"},
                    },
                    "required": ["role", "system_prompt"],
                },
            },
        }


class SendMessageTool:
    name = "send_message"

    def parse_text(self, text: str) -> dict | None:
        return _find_json_object(text, self.name)

    def execute(self, args: dict, ctx: ToolContext) -> str:
        if ctx.agent_registry is None:
            return "error: registry unavailable"
        target = args["agent_id"]
        if ctx.allowed_targets is not None and target not in ctx.allowed_targets:
            return f"error: not permitted to message {target}"
        handle = ctx.agent_registry.get(target)
        if handle is None:
            return f"error: agent {target} not found"
        handle.inbox.put(A2AMessage(from_id=ctx.caller_id, to_id=target, content=args["content"]))
        return f"message sent to {target}"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a text message to another agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["agent_id", "content"],
                },
            },
        }


class CheckAgentTool:
    name = "check_agent"

    def parse_text(self, text: str) -> dict | None:
        return _find_json_object(text, self.name)

    def execute(self, args: dict, ctx: ToolContext) -> str:
        if ctx.agent_registry is None:
            return "error: registry unavailable"
        handle = ctx.agent_registry.get(args["agent_id"])
        if handle is None:
            return f"error: agent {args['agent_id']} not found"
        msgs = handle.outbox.drain()
        payload = {
            "id": handle.id,
            "role": handle.role,
            "status": handle.status.value,
            "messages": [{"from": m.from_id, "content": m.content} for m in msgs],
        }
        return json.dumps(payload)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "check_agent",
                "description": "Check status of a subagent and drain any pending messages from it.",
                "parameters": {
                    "type": "object",
                    "properties": {"agent_id": {"type": "string"}},
                    "required": ["agent_id"],
                },
            },
        }


class ListAgentsTool:
    name = "list_agents"

    def parse_text(self, text: str) -> dict | None:
        return _find_json_object(text, self.name)

    def execute(self, args: dict, ctx: ToolContext) -> str:
        if ctx.agent_registry is None:
            return "error: registry unavailable"
        payload = [
            {"id": h.id, "role": h.role, "status": h.status.value}
            for h in ctx.agent_registry.list()
        ]
        return json.dumps(payload)

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_agents",
                "description": "List all agents and their statuses.",
                "parameters": {"type": "object", "properties": {}},
            },
        }


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def parse(self, text: str) -> tuple[Tool, dict] | None:
        for tool in self._tools.values():
            args = tool.parse_text(text)
            if args is not None:
                return tool, args
        return None

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
