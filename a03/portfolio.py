from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from .events import EventBus, FillEvent


@dataclass
class Position:
    quantity: float = 0.0
    avg_price: float = 0.0

    def apply_fill(self, side: str, quantity: float, price: float) -> None:
        if side == "BUY":
            new_qty = self.quantity + quantity
            if new_qty == 0:
                self.avg_price = 0.0
                self.quantity = 0.0
                return
            self.avg_price = (self.avg_price * self.quantity + price * quantity) / new_qty
            self.quantity = new_qty
        else:  # SELL
            self.quantity -= quantity
            if self.quantity <= 0:
                self.quantity = 0.0
                self.avg_price = 0.0


@dataclass
class Portfolio:
    bus: EventBus
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    last_price: Dict[str, float] = field(default_factory=dict)
    fees_paid: float = 0.0

    def __post_init__(self) -> None:
        self.bus.subscribe("FILL", self._on_fill)
        self.bus.subscribe("BAR", self._on_bar)

    def _on_fill(self, fill: FillEvent) -> None:
        pos = self.positions.setdefault(fill.symbol, Position())
        gross = fill.price * fill.quantity
        if fill.side == "BUY":
            self.cash -= gross + fill.fee
        else:
            self.cash += gross - fill.fee
        self.fees_paid += fill.fee
        pos.apply_fill(fill.side, fill.quantity, fill.price)

    def _on_bar(self, bar) -> None:
        self.last_price[bar.symbol] = bar.close

    def equity(self) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            price = self.last_price.get(symbol, pos.avg_price)
            value += pos.quantity * price
        return value
