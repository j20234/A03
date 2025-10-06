from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .events import EventBus, BarEvent, OrderEvent


@dataclass
class SmaCrossoverConfig:
    symbol: str
    fast_window: int = 5
    slow_window: int = 20
    trade_qty: float = 1.0


class SmaCrossoverStrategy:
    def __init__(self, bus: EventBus, cfg: SmaCrossoverConfig) -> None:
        assert cfg.fast_window > 0 and cfg.slow_window > cfg.fast_window
        self.bus = bus
        self.cfg = cfg
        self.fast: Deque[float] = deque(maxlen=cfg.fast_window)
        self.slow: Deque[float] = deque(maxlen=cfg.slow_window)
        self.prev_fast_above: Optional[bool] = None
        self.last_bar_ts: Optional[int] = None
        self.last_close: Optional[float] = None
        bus.subscribe("BAR", self.on_bar)

    def on_bar(self, bar: BarEvent) -> None:
        if bar.symbol != self.cfg.symbol:
            return
        self.fast.append(bar.close)
        self.slow.append(bar.close)
        self.last_close = bar.close
        self.last_bar_ts = bar.timestamp_ns
        if len(self.slow) < self.slow.maxlen:
            return
        fast_ma = sum(self.fast) / len(self.fast)
        slow_ma = sum(self.slow) / len(self.slow)
        fast_above = fast_ma > slow_ma
        if self.prev_fast_above is None:
            self.prev_fast_above = fast_above
            return
        if fast_above and not self.prev_fast_above:
            self._emit_order("BUY")
        elif (not fast_above) and self.prev_fast_above:
            self._emit_order("SELL")
        self.prev_fast_above = fast_above

    def _emit_order(self, side: str) -> None:
        assert self.last_bar_ts is not None
        order = OrderEvent(
            timestamp_ns=self.last_bar_ts,
            type="ORDER",
            data={"symbol": self.cfg.symbol},
            symbol=self.cfg.symbol,
            side=side,
            quantity=self.cfg.trade_qty,
            order_type="MARKET",
        )
        self.bus.publish(order)
