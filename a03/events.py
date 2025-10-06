from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar
import heapq
import time


# Base Event and typed subevents
@dataclass(frozen=True)
class Event:
    timestamp_ns: int
    type: str
    data: Dict[str, Any]


@dataclass(frozen=True)
class BarEvent(Event):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OrderEvent(Event):
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class FillEvent(Event):
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float


# Event Bus (synchronous, deterministic order)
T = TypeVar("T", bound=Event)

class EventHandler(Protocol[T]):
    def __call__(self, event: T) -> None: ...


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler[Any]]] = {}
        self._queue: List[tuple[int, int, Event]] = []
        self._counter: int = 0

    def publish(self, event: Event) -> None:
        # Ensure stable ordering: sort by timestamp, then by insertion counter
        heapq.heappush(self._queue, (event.timestamp_ns, self._counter, event))
        self._counter += 1

    def subscribe(self, event_type: str, handler: EventHandler[Any]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def run(self) -> None:
        while self._queue:
            _, _, event = heapq.heappop(self._queue)
            for handler in self._subscribers.get(event.type, []):
                handler(event)


# Helper to create timestamps
def now_timestamp_ns() -> int:
    return time.time_ns()
