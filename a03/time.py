from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Iterable, Iterator, List

# Simple weekday market calendar (Mon-Fri), daily bars at a fixed time

@dataclass(frozen=True)
class MarketClockConfig:
    bar_time: dtime = dtime(16, 0)  # 16:00 local
    tz: timezone = timezone.utc


def iter_trading_days(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri
            yield cur
        cur = cur + timedelta(days=1)


def trading_day_timestamp_ns(day: date, cfg: MarketClockConfig) -> int:
    dt = datetime.combine(day, cfg.bar_time).replace(tzinfo=cfg.tz)
    return int(dt.timestamp() * 1e9)
