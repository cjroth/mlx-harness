from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlxharness.a2a import A2AMessage, AgentHandle, AgentRegistry, AgentStatus, Inbox
from mlxharness.executor import CommandResult
from mlxharness.tools import (
    CheckAgentTool,
    ListAgentsTool,
    SendMessageTool,
    ShellTool,
    SpawnAgentTool,
    ToolContext,
    ToolRegistry,
)


class TestShellToolParse:
    def test_native_gemma_format(self):
        tool = ShellTool()
        args = tool.parse_text('<|tool_call>call:shell{command:<|"|>ls -la<|"|>}<tool_call|>')
        assert args == {"command": "ls -la"}

    def test_native_format_with_thought(self):
        tool = ShellTool()
        text = '<|channel>thought\nListing.<channel|><|tool_call>call:shell{command:<|"|>ls<|"|>}<tool_call|>'
        args = tool.parse_text(text)
        assert args == {"command": "ls"}

    def test_json_fallback(self):
        tool = ShellTool()
        assert tool.parse_text('{"tool": "shell", "command": "ls"}') == {"command": "ls"}

    def test_no_match_returns_none(self):
        tool = ShellTool()
        assert tool.parse_text("no tool call here") is None

    def test_wrong_tool_name_returns_none(self):
        tool = ShellTool()
        assert tool.parse_text('{"tool": "spawn_agent", "role": "x"}') is None


class TestShellToolExecute:
    def test_runs_command_via_executor(self):
        executor = MagicMock()
        executor.run = MagicMock(return_value=CommandResult(0, "out", ""))
        ctx = ToolContext(executor=executor)
        tool = ShellTool()
        result = tool.execute({"command": "ls"}, ctx)
        executor.run.assert_called_once_with("ls")
        assert isinstance(result, CommandResult)

    def test_requires_executor(self):
        ctx = ToolContext(executor=None)
        tool = ShellTool()
        with pytest.raises(RuntimeError):
            tool.execute({"command": "ls"}, ctx)


class TestToolRegistry:
    def test_parse_picks_first_matching_tool(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        text = '<|tool_call>call:shell{command:<|"|>ls<|"|>}<tool_call|>'
        match = reg.parse(text)
        assert match is not None
        tool, args = match
        assert tool.name == "shell"
        assert args == {"command": "ls"}

    def test_parse_unknown_returns_none(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        assert reg.parse("plain text no call") is None

    def test_duplicate_name_raises(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        with pytest.raises(ValueError):
            reg.register(ShellTool())

    def test_schemas_returns_openai_function_list(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        schemas = reg.schemas()
        assert isinstance(schemas, list)
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "shell"


class TestA2ATools:
    def _ctx_with_registry(self) -> tuple[ToolContext, AgentRegistry]:
        reg = AgentRegistry()
        ctx = ToolContext(
            executor=None,
            agent_registry=reg,
            caller_id="orchestrator",
            spawn_agent=None,
        )
        return ctx, reg

    def test_spawn_agent_calls_spawn_callback(self):
        reg = AgentRegistry()
        spawn_mock = MagicMock(return_value="agent-1")
        ctx = ToolContext(
            executor=None,
            agent_registry=reg,
            caller_id="orchestrator",
            spawn_agent=spawn_mock,
        )
        tool = SpawnAgentTool()
        args = {"role": "worker", "system_prompt": "be helpful", "initial_task": "ls"}
        out = tool.execute(args, ctx)
        spawn_mock.assert_called_once_with(
            role="worker",
            system_prompt="be helpful",
            initial_task="ls",
            model=None,
        )
        assert "agent-1" in out

    def test_send_message_enqueues_to_target_inbox(self):
        ctx, reg = self._ctx_with_registry()
        handle = AgentHandle(
            id="agent-1", role="worker", status=AgentStatus.IDLE,
            inbox=Inbox(),
        )
        reg.register(handle)
        tool = SendMessageTool()
        tool.execute({"agent_id": "agent-1", "content": "do the thing"}, ctx)
        msg = handle.inbox.get(timeout=0.1)
        assert msg is not None
        assert msg.from_id == "orchestrator"
        assert msg.content == "do the thing"

    def test_send_message_unknown_agent_returns_error(self):
        ctx, reg = self._ctx_with_registry()
        tool = SendMessageTool()
        out = tool.execute({"agent_id": "missing", "content": "x"}, ctx)
        assert "not found" in out.lower() or "error" in out.lower()

    def test_check_agent_drains_outbox(self):
        ctx, reg = self._ctx_with_registry()
        handle = AgentHandle(
            id="agent-1", role="worker", status=AgentStatus.RUNNING,
            inbox=Inbox(),
        )
        handle.outbox.put(A2AMessage("agent-1", "orchestrator", "progress"))
        handle.outbox.put(A2AMessage("agent-1", "orchestrator", "done"))
        reg.register(handle)
        tool = CheckAgentTool()
        out = tool.execute({"agent_id": "agent-1"}, ctx)
        assert "running" in out
        assert "progress" in out
        assert "done" in out
        # Second check should find outbox empty
        out2 = tool.execute({"agent_id": "agent-1"}, ctx)
        assert "progress" not in out2

    def test_list_agents_includes_all(self):
        ctx, reg = self._ctx_with_registry()
        reg.register(AgentHandle(id="agent-1", role="worker", status=AgentStatus.RUNNING, inbox=Inbox()))
        reg.register(AgentHandle(id="agent-2", role="reviewer", status=AgentStatus.IDLE, inbox=Inbox()))
        tool = ListAgentsTool()
        out = tool.execute({}, ctx)
        assert "agent-1" in out
        assert "agent-2" in out
        assert "running" in out
        assert "idle" in out


class TestRegistryWithA2A:
    def test_parse_json_for_spawn_agent(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.register(SpawnAgentTool())
        text = '{"tool": "spawn_agent", "role": "worker", "system_prompt": "p", "initial_task": "t"}'
        match = reg.parse(text)
        assert match is not None
        tool, args = match
        assert tool.name == "spawn_agent"
        assert args == {"role": "worker", "system_prompt": "p", "initial_task": "t"}

    def test_parse_json_for_send_message(self):
        reg = ToolRegistry()
        reg.register(SendMessageTool())
        text = '{"tool": "send_message", "agent_id": "agent-1", "content": "hi"}'
        match = reg.parse(text)
        assert match is not None
        assert match[0].name == "send_message"
        assert match[1] == {"agent_id": "agent-1", "content": "hi"}
