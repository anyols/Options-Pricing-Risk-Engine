"""P&L attribution: decompose a realized portfolio P&L move into Delta,
Gamma, Vega, Theta and cross-term (Vanna/Volga) contributions, plus an
unexplained residual -- the "P&L explain" a derivatives risk desk produces
every day to reconcile today's marks against yesterday's Greeks.

Where `risk.scenarios` asks "how wrong would a Greeks-only view be under a
hypothetical shock", this module asks the complementary, backward-looking
question: "given what the market actually did (spot moved, vol moved, a day
passed), which Greek explains how much of the realized P&L, and how much is
left over that the model can't explain?" A persistently large unexplained
residual is itself a risk signal -- it means the book's convexity/vol-of-vol
exposure (or a stale/wrong Greek) is bigger than a second-order Taylor
expansion can capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from optrisk.instruments.option import Model
from optrisk.instruments.portfolio import Portfolio

__all__ = ["PnLAttribution", "attribute_pnl"]


@dataclass
class PnLAttribution:
    """A full-reprice P&L, decomposed into Greek-bucket contributions.

    Each `*_pnl` bucket uses the *base-case* Greeks (computed once, before
    the move) applied to the actual observed shock -- exactly what a desk's
    end-of-day P&L explain does with the prior day's closing Greeks.
    """

    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    cross_pnl: float
    full_pnl: float

    @property
    def explained_pnl(self) -> float:
        """Sum of every Greek bucket -- the Taylor-series estimate of the full P&L."""
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.theta_pnl + self.cross_pnl

    @property
    def unexplained_pnl(self) -> float:
        """Full reprice minus everything the Greeks account for."""
        return self.full_pnl - self.explained_pnl

    @property
    def unexplained_pct(self) -> float:
        """|unexplained| as a percentage of |full_pnl| (0 when full_pnl is ~0)."""
        denom = abs(self.full_pnl)
        return 100.0 * abs(self.unexplained_pnl) / denom if denom > 1e-12 else 0.0

    def to_waterfall_frame(self) -> pd.DataFrame:
        """Ordered bucket -> value rows, ready for a waterfall chart."""
        return pd.DataFrame(
            {
                "bucket": ["Delta", "Gamma", "Vega", "Theta", "Vanna/Volga", "Unexplained"],
                "pnl": [
                    self.delta_pnl,
                    self.gamma_pnl,
                    self.vega_pnl,
                    self.theta_pnl,
                    self.cross_pnl,
                    self.unexplained_pnl,
                ],
            }
        )


def attribute_pnl(
    portfolio: Portfolio,
    *,
    spot_shock_pct: float = 0.0,
    vol_shock: float = 0.0,
    time_elapsed: float = 0.0,
    model: Model = "auto",
) -> PnLAttribution:
    """Attribute the P&L of moving `portfolio` by the given spot/vol shock and
    time elapsed (in years -- e.g. 1/252 for one trading day) to its Greeks.

    Evaluated per position (each against its own spot level and Greeks) and
    summed, for the same reason as `risk.scenarios`: a single pre-aggregated
    portfolio Greek and one spot level are only equivalent to the per-position
    sum when every position shares the same underlying.
    """
    base_value = portfolio.value(model=model)

    delta_pnl = gamma_pnl = vega_pnl = theta_pnl = cross_pnl = 0.0
    for position in portfolio.positions:
        greeks = position.position_greeks(model=model)
        d_spot = position.market.spot * spot_shock_pct
        d_vol = vol_shock
        delta_pnl += greeks.delta * d_spot
        gamma_pnl += 0.5 * greeks.gamma * d_spot**2
        vega_pnl += greeks.vega * d_vol
        theta_pnl += greeks.theta * time_elapsed
        cross_pnl += greeks.vanna * d_spot * d_vol + 0.5 * greeks.volga * d_vol**2

    moved_portfolio = portfolio.shocked(spot_shock_pct=spot_shock_pct, vol_shock=vol_shock).aged(time_elapsed)
    full_pnl = moved_portfolio.value(model=model) - base_value

    return PnLAttribution(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        cross_pnl=cross_pnl,
        full_pnl=full_pnl,
    )
