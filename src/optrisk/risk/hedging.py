"""Delta-hedging simulation: replicate an option position by dynamically
trading the underlying, and measure how discrete rebalancing tracks the
continuous-time Black-Scholes replication argument.

The underlying is simulated under a *realized* volatility that can differ
from the *implied* volatility used to price and hedge the option throughout.
That gap drives the classic delta-hedged P&L-explain relationship: for a
position of `option_quantity` options (positive = long),

    dPnL ~= option_quantity * 0.5 * Gamma * S^2 * (sigma_realized^2 - sigma_implied^2) * dt

i.e. a long-gamma, delta-hedged position profits when the world moves more
than it was hedged for, and loses when it moves less -- independent of
*which direction* the market moves. This module demonstrates that
relationship empirically via simulation rather than asserting it as a
formula: :func:`run_hedge_monte_carlo` gives the distribution of realized
hedging P&L, and :func:`run_hedge_frequency_comparison` shows how that
distribution tightens as rebalancing gets more frequent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optrisk.greeks.analytical import bsm_greeks
from optrisk.models.black_scholes import bsm_price

__all__ = [
    "HedgeSimulationResult",
    "run_hedge_frequency_comparison",
    "run_hedge_monte_carlo",
    "simulate_delta_hedge",
]


@dataclass
class HedgeSimulationResult:
    """One simulated hedging path, mark-to-model at every rebalancing date."""

    times: np.ndarray
    spot_path: np.ndarray
    delta_path: np.ndarray
    option_value_path: np.ndarray
    stock_position_path: np.ndarray
    cash_path: np.ndarray
    portfolio_value_path: np.ndarray
    final_pnl: float
    n_rebalances: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": self.times,
                "spot": self.spot_path,
                "delta": self.delta_path,
                "option_value": self.option_value_path,
                "stock_position": self.stock_position_path,
                "cash": self.cash_path,
                "portfolio_value": self.portfolio_value_path,
            }
        )


def simulate_delta_hedge(
    spot0: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    implied_vol: float,
    realized_vol: float,
    expiry: float,
    option_type: str = "call",
    option_quantity: float = 1.0,
    n_steps: int = 252,
    seed: int | None = None,
    spot_path: np.ndarray | None = None,
) -> HedgeSimulationResult:
    """Simulate delta-hedging `option_quantity` options from t=0 to expiry.

    `option_quantity` is signed: positive = long the option (hedged with a
    short stock position, the textbook Pi = V - Delta*S), negative = short
    the option (the dealer/seller perspective). The option is priced and
    hedged throughout using `implied_vol`; the underlying's *actual* path is
    drawn under `realized_vol` (or pass a pre-generated `spot_path`, e.g. to
    compare rebalancing frequencies on one shared realization).
    """
    if spot_path is not None and len(spot_path) != n_steps + 1:
        raise ValueError(f"spot_path must have length n_steps+1={n_steps + 1}, got {len(spot_path)}")

    dt = expiry / n_steps
    times = np.linspace(0.0, expiry, n_steps + 1)

    if spot_path is None:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_steps)
        log_returns = (rate - dividend_yield - 0.5 * realized_vol**2) * dt + realized_vol * np.sqrt(dt) * z
        spot_path = spot0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    else:
        spot_path = np.asarray(spot_path, dtype=float)

    remaining = np.maximum(expiry - times, 1e-8)
    option_value_path = np.asarray(
        bsm_price(spot_path, strike, rate, dividend_yield, implied_vol, remaining, option_type)
    )
    delta_path = np.asarray(
        bsm_greeks(spot_path, strike, rate, dividend_yield, implied_vol, remaining, option_type).delta
    )

    # hedge stock position that offsets the option book's delta to zero
    stock_position_path = -option_quantity * delta_path

    cash_path = np.empty(n_steps + 1)
    # t=0: receive/pay the option premium, buy/sell the initial hedge, rest is cash
    cash_path[0] = -option_quantity * option_value_path[0] - stock_position_path[0] * spot_path[0]
    growth = np.exp(rate * dt)
    for i in range(1, n_steps + 1):
        trade = stock_position_path[i] - stock_position_path[i - 1]
        cash_path[i] = cash_path[i - 1] * growth - trade * spot_path[i]

    portfolio_value_path = option_quantity * option_value_path + stock_position_path * spot_path + cash_path

    return HedgeSimulationResult(
        times=times,
        spot_path=spot_path,
        delta_path=delta_path,
        option_value_path=option_value_path,
        stock_position_path=stock_position_path,
        cash_path=cash_path,
        portfolio_value_path=portfolio_value_path,
        final_pnl=float(portfolio_value_path[-1]),
        n_rebalances=n_steps,
    )


def run_hedge_monte_carlo(
    spot0: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    implied_vol: float,
    realized_vol: float,
    expiry: float,
    option_type: str = "call",
    option_quantity: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """Final hedging P&L across `n_paths` independent simulated underlying paths."""
    rng = np.random.default_rng(seed)
    pnls = np.empty(n_paths)
    for i in range(n_paths):
        result = simulate_delta_hedge(
            spot0,
            strike,
            rate,
            dividend_yield,
            implied_vol,
            realized_vol,
            expiry,
            option_type,
            option_quantity,
            n_steps=n_steps,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        pnls[i] = result.final_pnl
    return pnls


def run_hedge_frequency_comparison(
    spot0: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    implied_vol: float,
    realized_vol: float,
    expiry: float,
    option_type: str = "call",
    option_quantity: float = 1.0,
    frequencies: dict[str, int] | None = None,
    n_paths: int = 300,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compare final hedging P&L across rebalancing frequencies.

    Every frequency is simulated against the *same* underlying realization
    per trial (subsampled from the finest grid), so differences across
    frequencies are attributable purely to rebalancing frequency, not to
    different random draws.
    """
    frequencies = frequencies or {"Daily": 252, "Weekly": 52, "Monthly": 12}
    finest = max(frequencies.values())
    for label, steps in frequencies.items():
        if finest % steps != 0:
            raise ValueError(f"frequency {label!r} ({steps} steps) must evenly divide the finest frequency ({finest})")

    rng = np.random.default_rng(seed)
    records = []
    for trial in range(n_paths):
        z = rng.standard_normal(finest)
        fine_dt = expiry / finest
        log_returns = (rate - dividend_yield - 0.5 * realized_vol**2) * fine_dt + realized_vol * np.sqrt(fine_dt) * z
        fine_path = spot0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

        for label, steps in frequencies.items():
            stride = finest // steps
            sub_path = fine_path[::stride]
            result = simulate_delta_hedge(
                spot0,
                strike,
                rate,
                dividend_yield,
                implied_vol,
                realized_vol,
                expiry,
                option_type,
                option_quantity,
                n_steps=steps,
                spot_path=sub_path,
            )
            records.append({"trial": trial, "frequency": label, "n_rebalances": steps, "final_pnl": result.final_pnl})

    return pd.DataFrame(records)
