from __future__ import annotations

import threading
from typing import Any

from mlxharness.a2a import (
    A2AMessage,
    AgentHandle,
    AgentRegistry,
    AgentStatus,
    Inbox,
)
from mlxharness.agent import Agent, EngineProtocol
from mlxharness.bus import EventBus
from mlxharness.events import Event
from mlxharness.executor import Executor
from mlxharness.tools import (
    CheckAgentTool,
    ListAgentsTool,
    SendMessageTool,
    ShellTool,
    SpawnAgentTool,
    ToolRegistry,
)


ORCHESTRATOR_ID = "orchestrator"

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the ORCHESTRATOR. You coordinate child subagents to accomplish tasks. "
    "You can spawn subagents with `spawn_agent`, message them with `send_message`, "
    "and poll their status with `check_agent`. You may also run shell commands directly. "
    "Subagents run concurrently — spawn them and continue; do not block waiting. "
    "Use `list_agents` to see all active agents. When you are done, respond with "
    "plain text (no tool call)."
)

SUBAGENT_SYSTEM_PROMPT_SUFFIX = (
    "\n\nYou are a subagent. You can run shell commands. To reply to your parent, "
    "respond with plain text — the parent receives your final message automatically."
)


def _orchestrator_registry(default_registry: ToolRegistry | None = None) -> ToolRegistry:
    reg = default_registry if default_registry is not None else ToolRegistry()
    reg.register(ShellTool())
    reg.register(SpawnAgentTool())
    reg.register(SendMessageTool())
    reg.register(CheckAgentTool())
    reg.register(ListAgentsTool())
    return reg


def _subagent_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ShellTool())
    reg.register(SendMessageTool())
    return reg


class Orchestrator:
    def __init__(
        self,
        engine: EngineProtocol,
        executor: Executor | None,
        bus: EventBus,
        gen_lock: threading.Lock | None = None,
    ) -> None:
        self.engine = engine
        self.executor = executor
        self.bus = bus
        self.gen_lock = gen_lock if gen_lock is not None else threading.Lock()
        self.registry = AgentRegistry()

        inbox = Inbox()
        self._orchestrator_handle = AgentHandle(
            id=ORCHESTRATOR_ID,
            role="orchestrator",
            status=AgentStatus.IDLE,
            inbox=inbox,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        )
        self.registry.register(self._orchestrator_handle)

        self._orchestrator = Agent(
            engine=engine,
            executor=executor,
            agent_id=ORCHESTRATOR_ID,
            tool_registry=_orchestrator_registry(),
            inbox=inbox,
            gen_lock=self.gen_lock,
            event_sink=lambda ev: self.bus.publish(ORCHESTRATOR_ID, ev),
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            agent_registry=self.registry,
            spawn_agent_fn=self._spawn_subagent,
        )
        self.thread: threading.Thread = threading.Thread(
            target=self._orchestrator.run_forever, daemon=True, name="orchestrator"
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.thread.start()
        self._started = True

    def submit_user(self, text: str) -> None:
        """Non-blocking: enqueues a user message to the orchestrator's inbox."""
        self._orchestrator_handle.inbox.put(
            A2AMessage(from_id="user", to_id=ORCHESTRATOR_ID, content=text)
        )

    def _spawn_subagent(
        self,
        role: str,
        system_prompt: str,
        initial_task: str | None = None,
        model: str | None = None,
    ) -> str:
        agent_id = self.registry.next_id()
        inbox = Inbox()
        outbox = Inbox()
        handle = AgentHandle(
            id=agent_id,
            role=role,
            status=AgentStatus.IDLE,
            inbox=inbox,
            outbox=outbox,
            system_prompt=system_prompt,
            model=model,
        )
        self.registry.register(handle)

        full_prompt = system_prompt + SUBAGENT_SYSTEM_PROMPT_SUFFIX
        agent = Agent(
            engine=self.engine,
            executor=self.executor,
            agent_id=agent_id,
            tool_registry=_subagent_registry(),
            inbox=inbox,
            outbox=outbox,
            gen_lock=self.gen_lock,
            event_sink=lambda ev, aid=agent_id: self.bus.publish(aid, ev),
            system_prompt=full_prompt,
            parent_id=ORCHESTRATOR_ID,
            agent_registry=self.registry,
            allowed_message_targets={ORCHESTRATOR_ID},
        )

        thread = threading.Thread(target=agent.run_forever, daemon=True, name=agent_id)
        handle.thread = thread
        # Bridge the subagent's outbox into the orchestrator's inbox so that
        # completion / replies are auto-delivered to the parent.
        bridge = threading.Thread(
            target=self._bridge_outbox, args=(handle,), daemon=True, name=f"{agent_id}-bridge"
        )
        thread.start()
        bridge.start()

        if initial_task:
            inbox.put(A2AMessage(from_id=ORCHESTRATOR_ID, to_id=agent_id, content=initial_task))

        # Store for shutdown
        self._agents_to_stop(agent)
        return agent_id

    def _agents_to_stop(self, agent: Agent) -> None:
        self._children: list[Agent]
        if not hasattr(self, "_children"):
            self._children = []
        self._children.append(agent)

    def _bridge_outbox(self, handle: AgentHandle) -> None:
        while not self._orchestrator._stop.is_set():
            msg = handle.outbox.get(timeout=0.1)
            if msg is None:
                if handle.thread is not None and not handle.thread.is_alive() and handle.outbox.empty():
                    return
                continue
            # Deliver subagent's reply to the orchestrator.
            self._orchestrator_handle.inbox.put(msg)

    def shutdown(self) -> None:
        self._orchestrator.stop()
        for child in getattr(self, "_children", []):
            child.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
