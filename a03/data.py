from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import math

from .events import BarEvent, Event
from .time import MarketClockConfig, iter_trading_days, trading_day_timestamp_ns


@dataclass
class SyntheticBarConfig:
    symbol: str
    start_price: float = 100.0
    daily_drift: float = 0.0005  # ~0.05%
    daily_vol: float = 0.01      # 1% vol


def generate_sine_series(n: int, amplitude: float = 1.0, period: int = 20) -> List[float]:
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def generate_daily_bars(symbol: str, days: List[int], start_price: float, drift: float, vol: float) -> List[tuple[float, float, float, float, float]]:
    prices: List[float] = [start_price]
    for _ in range(1, len(days)):
        prev = prices[-1]
        shock = vol * (0.5 - (_ % 2) * 1.0) * 0.02  # simple alternating noise
        next_p = prev * (1.0 + drift + shock)
        prices.append(max(0.01, next_p))
    bars: List[tuple[float, float, float, float, float]] = []
    for p in prices:
        high = p * (1 + 0.005)
        low = p * (1 - 0.005)
        open_p = (high + low) / 2
        close = p
        volume = 1_000.0
        bars.append((open_p, high, low, close, volume))
    return bars


def generate_synthetic_bar_events(cfg: SyntheticBarConfig, start_day, end_day, clock: MarketClockConfig):
    days = list(iter_trading_days(start_day, end_day))
    timestamps = [trading_day_timestamp_ns(d, clock) for d in days]
    bars = generate_daily_bars(cfg.symbol, timestamps, cfg.start_price, cfg.daily_drift, cfg.daily_vol)
    for ts, (o, h, l, c, v) in zip(timestamps, bars):
        yield BarEvent(
            timestamp_ns=ts,
            type="BAR",
            data={"symbol": cfg.symbol},
            symbol=cfg.symbol,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
        )
