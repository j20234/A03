from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from .events import EventBus, OrderEvent, FillEvent, BarEvent


@dataclass
class BrokerConfig:
    commission_rate_bps: float = 1.0  # 1 bp = 0.01%
    min_commission: float = 0.0
    slippage_bps: float = 0.0


class Broker:
    def __init__(self, bus: EventBus, cfg: BrokerConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self.last_price: Dict[str, float] = {}
        self.bus.subscribe("ORDER", self._on_order)
        self.bus.subscribe("BAR", self._on_bar)

    def _on_bar(self, bar: BarEvent) -> None:
        self.last_price[bar.symbol] = bar.close

    def _on_order(self, order: OrderEvent) -> None:
        last_price = self.last_price.get(order.symbol)
        if last_price is None:
            return  # cannot fill without a price
        self._submit(order, last_price)

    def _submit(self, order: OrderEvent, last_price: float) -> None:
        if order.order_type == "MARKET":
            exec_price = self._apply_slippage(last_price, order.side)
        elif order.order_type == "LIMIT":
            assert order.limit_price is not None
            can_fill = (
                (order.side == "BUY" and order.limit_price >= last_price) or
                (order.side == "SELL" and order.limit_price <= last_price)
            )
            if not can_fill:
                return
            exec_price = order.limit_price
        else:
            raise ValueError(f"Unknown order type: {order.order_type}")

        fee = max(self.cfg.min_commission, exec_price * order.quantity * (self.cfg.commission_rate_bps / 10_000))
        fill = FillEvent(
            timestamp_ns=order.timestamp_ns,
            type="FILL",
            data={"symbol": order.symbol},
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            fee=fee,
        )
        self.bus.publish(fill)

    def _apply_slippage(self, price: float, side: str) -> float:
        if self.cfg.slippage_bps == 0:
            return price
        factor = 1 + (self.cfg.slippage_bps / 10_000)
        return price * (factor if side == "BUY" else 1 / factor)
