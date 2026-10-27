"""The PhAP event log: numbered events, a bounded replay buffer, and live subscribers."""

import asyncio
import collections
from datetime import UTC, datetime
from typing import Any

Event = dict[str, Any]


class EventLog:
    def __init__(self, buffer: int = 1000) -> None:
        self._events: collections.deque[Event] = collections.deque(maxlen=buffer)
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self.last_seq = 0

    def emit(self, type: str, **fields: Any) -> Event:
        self.last_seq += 1
        ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        event = {"seq": self.last_seq, "ts": ts, "type": type} | {k: v for k, v in fields.items() if v is not None}
        self._events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def since(self, seq: int) -> tuple[list[Event], bool]:
        """Buffered events after seq, and whether some of them were already evicted (a gap)."""
        oldest = self._events[0]["seq"] if self._events else self.last_seq + 1
        gap = seq > self.last_seq or seq + 1 < oldest
        return [e for e in self._events if e["seq"] > seq], gap

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(queue)
