from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class A2AMessage:
    from_id: str
    to_id: str
    content: str


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


class Inbox:
    """Thread-safe FIFO of A2AMessages with blocking get and bulk drain."""

    def __init__(self) -> None:
        self._q: queue.Queue[A2AMessage] = queue.Queue()

    def put(self, msg: A2AMessage) -> None:
        self._q.put(msg)

    def get(self, timeout: float | None = None) -> A2AMessage | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[A2AMessage]:
        out: list[A2AMessage] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def empty(self) -> bool:
        return self._q.empty()


@dataclass
class AgentHandle:
    id: str
    role: str
    status: AgentStatus
    inbox: Inbox
    outbox: Inbox = field(default_factory=Inbox)
    thread: threading.Thread | None = None
    system_prompt: str | None = None
    model: str | None = None
    stop_flag: threading.Event = field(default_factory=threading.Event)


class AgentRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, AgentHandle] = {}
        self._next_seq = 0

    def register(self, handle: AgentHandle) -> None:
        with self._lock:
            if handle.id in self._agents:
                raise ValueError(f"Agent id already registered: {handle.id}")
            self._agents[handle.id] = handle

    def get(self, agent_id: str) -> AgentHandle | None:
        with self._lock:
            return self._agents.get(agent_id)

    def list(self) -> list[AgentHandle]:
        with self._lock:
            return list(self._agents.values())

    def next_id(self) -> str:
        with self._lock:
            self._next_seq += 1
            return f"agent-{self._next_seq}"
