from datetime import date

from a03.events import EventBus
from a03.time import MarketClockConfig
from a03.data import SyntheticBarConfig, generate_synthetic_bar_events
from a03.strategy import SmaCrossoverConfig, SmaCrossoverStrategy
from a03.broker import Broker, BrokerConfig
from a03.portfolio import Portfolio


def main() -> None:
    bus = EventBus()

    # Components
    clock_cfg = MarketClockConfig()
    data_cfg = SyntheticBarConfig(symbol="AAPL", start_price=100.0)
    strat_cfg = SmaCrossoverConfig(symbol="AAPL", fast_window=5, slow_window=20, trade_qty=10)
    broker = Broker(bus, BrokerConfig(commission_rate_bps=1.0, slippage_bps=0.5))
    portfolio = Portfolio(bus=bus, cash=100_000.0)
    strategy = SmaCrossoverStrategy(bus=bus, cfg=strat_cfg)

    # Feed data
    for ev in generate_synthetic_bar_events(data_cfg, date(2024, 1, 1), date(2024, 6, 30), clock_cfg):
        bus.publish(ev)

    # Run event loop
    bus.run()

    # Results
    print("Final cash:", round(portfolio.cash, 2))
    print("Final equity:", round(portfolio.equity(), 2))
    print("Fees paid:", round(portfolio.fees_paid, 2))


if __name__ == "__main__":
    main()
