"""Scenario analysis: full repricing vs Taylor-series Greek approximations
under spot and volatility shocks.

For every point on a (spot shock %, vol shock) grid this computes (a) the
exact full-reprice portfolio P&L, and (b) increasingly rich Taylor-series
approximations built only from the base-case Greeks. The *gap* between them
is the direct, quantitative measure of nonlinear portfolio risk: it is
exactly what a Greeks-only risk view misses as the market moves away from
the base case.

Shocks are applied uniformly (same relative spot %, same absolute vol
points) across every position in the book -- a single systematic-scenario
convention. For a multi-underlying book, the Taylor expansion is therefore
evaluated *per position* (each against its own spot level and Greeks) and
summed, rather than against one pre-aggregated portfolio Greek and a single
spot level, since those are only equivalent when every position shares the
same underlying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from optrisk.greeks.types import Greeks
from optrisk.instruments.portfolio import Portfolio

__all__ = ["TAYLOR_ORDERS", "ScenarioResult", "default_shock_grid", "run_scenario_analysis"]

# increasing order of Taylor-series richness; each includes every term before it
TAYLOR_ORDERS: Tuple[str, ...] = ("delta", "delta_gamma", "delta_gamma_vega", "full_2nd_order")

TAYLOR_ORDER_LABELS: Dict[str, str] = {
    "delta": "Delta only",
    "delta_gamma": "Delta + Gamma",
    "delta_gamma_vega": "Delta + Gamma + Vega",
    "full_2nd_order": "Full 2nd order (+ Vanna/Volga)",
}


def default_shock_grid(
    spot_range_pct: float = 0.20,
    n_spot: int = 21,
    vol_range: float = 0.15,
    n_vol: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """A symmetric (spot_shocks, vol_shocks) grid centered on the base case."""
    spot_shocks = np.linspace(-spot_range_pct, spot_range_pct, n_spot)
    vol_shocks = np.linspace(-vol_range, vol_range, n_vol)
    return spot_shocks, vol_shocks


def _position_taylor_pnl(greeks: Greeks, spot0: float, spot_shock_pct: float, vol_shock: float, order: str) -> float:
    """Taylor-series P&L for one position's already qty/multiplier-scaled Greeks."""
    d_spot = spot0 * spot_shock_pct
    pnl = greeks.delta * d_spot
    if order == "delta":
        return pnl
    pnl += 0.5 * greeks.gamma * d_spot**2
    if order == "delta_gamma":
        return pnl
    pnl += greeks.vega * vol_shock
    if order == "delta_gamma_vega":
        return pnl
    pnl += greeks.vanna * d_spot * vol_shock + 0.5 * greeks.volga * vol_shock**2
    if order == "full_2nd_order":
        return pnl
    raise ValueError(f"unknown Taylor order {order!r}; expected one of {TAYLOR_ORDERS}")


@dataclass
class ScenarioResult:
    """Full-reprice and Taylor-approximation P&L surfaces over a shock grid.

    `full_pnl` and every array in `taylor_pnl` have shape (n_vol, n_spot),
    indexed [vol_shock_index, spot_shock_index] to match a natural heatmap
    orientation (rows = vol, columns = spot).
    """

    spot_shocks: np.ndarray
    vol_shocks: np.ndarray
    base_value: float
    base_greeks: Greeks
    full_pnl: np.ndarray
    taylor_pnl: Dict[str, np.ndarray]

    def error(self, order: str) -> np.ndarray:
        """Taylor-approx P&L minus full-reprice P&L (+ means the approx overstates P&L)."""
        return self.taylor_pnl[order] - self.full_pnl

    def max_abs_error(self, order: str) -> float:
        return float(np.max(np.abs(self.error(order))))

    def to_long_frame(self) -> pd.DataFrame:
        """Tidy long-format frame: one row per (spot_shock, vol_shock)."""
        rows = []
        for i, dvol in enumerate(self.vol_shocks):
            for j, ds in enumerate(self.spot_shocks):
                row = {"spot_shock": ds, "vol_shock": dvol, "full_pnl": self.full_pnl[i, j]}
                for order in self.taylor_pnl:
                    row[f"{order}_pnl"] = self.taylor_pnl[order][i, j]
                    row[f"{order}_error"] = self.error(order)[i, j]
                rows.append(row)
        return pd.DataFrame(rows)


def run_scenario_analysis(
    portfolio: Portfolio,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
    model: str = "auto",
) -> ScenarioResult:
    """Full reprice + Taylor-approximation P&L surfaces for `portfolio` over the grid."""
    base_value = portfolio.value(model=model)
    base_greeks = portfolio.greeks(model=model)
    position_greeks = [p.position_greeks(model=model) for p in portfolio.positions]
    position_spots = [p.market.spot for p in portfolio.positions]

    n_vol, n_spot = len(vol_shocks), len(spot_shocks)
    full_pnl = np.zeros((n_vol, n_spot))
    taylor_pnl = {order: np.zeros((n_vol, n_spot)) for order in TAYLOR_ORDERS}

    for i, dvol in enumerate(vol_shocks):
        for j, ds in enumerate(spot_shocks):
            shocked_value = portfolio.shocked(spot_shock_pct=float(ds), vol_shock=float(dvol)).value(model=model)
            full_pnl[i, j] = shocked_value - base_value

            for order in TAYLOR_ORDERS:
                taylor_pnl[order][i, j] = sum(
                    _position_taylor_pnl(g, s0, float(ds), float(dvol), order)
                    for g, s0 in zip(position_greeks, position_spots)
                )

    return ScenarioResult(
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        base_value=base_value,
        base_greeks=base_greeks,
        full_pnl=full_pnl,
        taylor_pnl=taylor_pnl,
    )
