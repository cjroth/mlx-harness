from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlxharness.events import Event


class EventBus:
    """Thread-safe fan-out of (agent_id, Event) envelopes to multiple subscribers.

    Subscribers receive only events published after they call subscribe().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[tuple[str, Event]]] = []

    def subscribe(self) -> queue.Queue[tuple[str, "Event"]]:
        q: queue.Queue[tuple[str, Event]] = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def publish(self, agent_id: str, event: "Event") -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put((agent_id, event))
