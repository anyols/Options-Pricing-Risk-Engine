"""Synthetic demo data: a small multi-leg, multi-underlying options book.

Everything here is illustrative (a fictitious equity "TECH" and a fictitious
commodity future "CRUDE") but internally consistent -- realistic vols,
moneyness and maturities -- so the dashboard, notebook and README charts
have a non-trivial book to show without needing a market data subscription.
"""

from __future__ import annotations

from optrisk.instruments.option import MarketEnv, OptionSpec, Stock
from optrisk.instruments.portfolio import Portfolio, Position

__all__ = ["build_demo_portfolio", "sample_equity_market", "sample_future_market"]


def sample_equity_market(spot: float = 185.0, vol: float = 0.28) -> MarketEnv:
    return MarketEnv(spot=spot, rate=0.045, vol=vol, dividend_yield=0.008)


def sample_future_market(forward: float = 78.0, vol: float = 0.35) -> MarketEnv:
    return MarketEnv(spot=forward, rate=0.045, vol=vol, dividend_yield=0.0)


def build_demo_portfolio() -> Portfolio:
    """A book with genuine convexity: long upside calls, a short OTM call spread
    leg, a long-dated protective put tail hedge (American), a short ATM put
    financing leg, a small futures-options position on a second underlying,
    and a partial delta hedge in the primary stock.
    """
    eq = sample_equity_market()
    fut = sample_future_market()

    portfolio = Portfolio(name="Demo Multi-Underlying Book")
    portfolio.add(
        Position(
            instrument=OptionSpec(
                option_type="call", strike=185.0, expiry=0.25, style="european", underlying_type="equity"
            ),
            quantity=10,
            market=eq,
            label="TECH 185C 3M",
            multiplier=100,
        )
    )
    portfolio.add(
        Position(
            instrument=OptionSpec(
                option_type="call", strike=210.0, expiry=0.25, style="european", underlying_type="equity"
            ),
            quantity=-15,
            market=eq,
            label="TECH 210C 3M",
            multiplier=100,
        )
    )
    portfolio.add(
        Position(
            instrument=OptionSpec(
                option_type="put", strike=160.0, expiry=0.5, style="american", underlying_type="equity"
            ),
            quantity=20,
            market=eq,
            label="TECH 160P 6M (Am)",
            multiplier=100,
        )
    )
    portfolio.add(
        Position(
            instrument=OptionSpec(
                option_type="put", strike=185.0, expiry=0.5, style="european", underlying_type="equity"
            ),
            quantity=-5,
            market=eq,
            label="TECH 185P 6M",
            multiplier=100,
        )
    )
    portfolio.add(
        Position(
            instrument=OptionSpec(
                option_type="call", strike=82.0, expiry=0.33, style="european", underlying_type="future"
            ),
            quantity=2,
            market=fut,
            label="CRUDE 82C 4M (Fut)",
            multiplier=1000,
        )
    )
    portfolio.add(
        Position(
            instrument=Stock(),
            quantity=-75,
            market=eq,
            label="TECH shares (partial hedge)",
            multiplier=1,
        )
    )
    return portfolio
