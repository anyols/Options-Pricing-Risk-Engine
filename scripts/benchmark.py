"""Performance benchmarks for the pricing/risk engine.

Run with: python scripts/benchmark.py

Every model in this engine is vectorized over numpy arrays wherever the
underlying math allows it (everything except the binomial tree and Monte
Carlo, which are inherently iterative/stochastic), so a full portfolio
scenario grid is priced as a handful of large array operations rather than
a Python loop over every (spot, vol) pair times every position.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optrisk.greeks.analytical import bsm_greeks
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.models.binomial import crr_price
from optrisk.models.black_scholes import bsm_price
from optrisk.models.heston import heston_mc_price, heston_price
from optrisk.models.monte_carlo import mc_price
from optrisk.risk.hedging import run_hedge_monte_carlo
from optrisk.risk.pnl_attribution import attribute_pnl
from optrisk.risk.scenarios import default_shock_grid, run_scenario_analysis


@dataclass
class Timing:
    label: str
    seconds: float
    units: int
    unit_name: str

    @property
    def rate(self) -> float:
        return self.units / self.seconds if self.seconds > 0 else float("inf")

    def report(self) -> str:
        time_str = f"{self.seconds * 1000:,.2f} ms"
        rate_str = f"{self.rate:,.0f} {self.unit_name}/s"
        return f"  {self.label:<58s} {time_str:>14s}  {rate_str:>22s}"


def timed(label: str, units: int, unit_name: str, fn: Callable[[], object], repeats: int = 3) -> Timing:
    best = min(_time_once(fn) for _ in range(repeats))
    return Timing(label=label, seconds=best, units=units, unit_name=unit_name)


def _time_once(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    print(f"  {'benchmark':<58s} {'time':>14s}  {'throughput':>22s}\n" + "-" * 100)

    # 1. Vectorized BSM pricing: one call over a large array vs. an equivalent Python loop.
    n = 1_000_000
    spots = np.random.default_rng(0).uniform(50, 150, n)
    t_vec = timed(
        f"BSM price, vectorized ({n:,} options, 1 call)",
        n,
        "options",
        lambda: bsm_price(spots, 100.0, 0.04, 0.01, 0.25, 0.5, "call"),
    )
    print(t_vec.report())

    n_loop = 20_000
    spots_loop = spots[:n_loop]
    t_loop = timed(
        f"BSM price, Python loop ({n_loop:,} options)",
        n_loop,
        "options",
        lambda: [bsm_price(float(s), 100.0, 0.04, 0.01, 0.25, 0.5, "call") for s in spots_loop],
    )
    print(t_loop.report())
    print(f"  -> vectorization speedup: {(t_vec.rate / t_loop.rate):,.0f}x\n")

    # 2. Closed-form Greeks (vectorized) vs the finite-difference engine (scalar, ~9 reprices/Greek-set).
    t_analytical_greeks = timed(
        f"Closed-form Greeks, vectorized ({n:,} sets, 1 call)",
        n,
        "Greek-sets",
        lambda: bsm_greeks(spots, 100.0, 0.04, 0.01, 0.25, 0.5, "call"),
    )
    print(t_analytical_greeks.report())

    # 3. Binomial tree (American exercise) -- inherently O(steps^2), not vectorizable across spots.
    t_binomial = timed(
        "Binomial tree, single price (500 steps, American put)",
        1,
        "prices",
        lambda: crr_price(100.0, 105.0, 0.04, 0.01, 0.25, 0.5, "put", style="american", steps=500),
    )
    print(t_binomial.report())

    # 4. Monte Carlo throughput.
    n_paths = 500_000
    t_mc = timed(
        f"Monte Carlo, GBM ({n_paths:,} paths, antithetic + control variate)",
        n_paths,
        "paths",
        lambda: mc_price(100.0, 100.0, 0.04, 0.01, 0.25, 0.5, "call", n_paths=n_paths, seed=0),
    )
    print(t_mc.report())

    # 5. Heston: COS (semi-analytical) vs Monte Carlo, same price target.
    # (params inlined into both calls rather than shared via a dict, since
    # **dict-unpacking against a mixed-type signature like n_terms: int
    # defeats mypy's keyword-argument checking)
    t_heston_cos = timed(
        "Heston price via COS method (160 series terms)",
        1,
        "prices",
        lambda: heston_price(100.0, 100.0, 0.03, 0.0, 0.5, 0.045, 1.8, 0.045, 0.55, -0.75, "call"),
    )
    print(t_heston_cos.report())
    t_heston_mc = timed(
        "Heston price via Monte Carlo (100k paths, 100 steps)",
        1,
        "prices",
        lambda: heston_mc_price(
            100.0, 100.0, 0.03, 0.0, 0.5, 0.045, 1.8, 0.045, 0.55, -0.75, "call", n_paths=100_000, n_steps=100, seed=0
        ),
    )
    print(t_heston_mc.report())
    speedup = t_heston_mc.seconds / t_heston_cos.seconds
    print(f"  -> COS is {speedup:,.0f}x faster than Monte Carlo for the same price\n")

    # 6. A realistic multi-leg portfolio: full scenario grid reprice and P&L attribution.
    portfolio = build_demo_portfolio()
    spot_shocks, vol_shocks = default_shock_grid(spot_range_pct=0.25, n_spot=41, vol_range=0.15, n_vol=31)
    grid_points = len(spot_shocks) * len(vol_shocks)
    t_scenario = timed(
        f"Full scenario grid reprice, {len(portfolio)}-leg book ({grid_points:,} grid points)",
        grid_points,
        "reprices",
        lambda: run_scenario_analysis(portfolio, spot_shocks, vol_shocks),
    )
    print(t_scenario.report())

    t_attribution = timed(
        f"P&L attribution, {len(portfolio)}-leg book (1 full reprice + Greeks)",
        1,
        "reports",
        lambda: attribute_pnl(portfolio, spot_shock_pct=-0.01, vol_shock=0.005, time_elapsed=1 / 252),
    )
    print(t_attribution.report())

    # 7. Delta-hedging Monte Carlo throughput.
    n_hedge_paths = 2_000
    t_hedge = timed(
        f"Delta-hedge simulation ({n_hedge_paths:,} paths x 252 rebalances)",
        n_hedge_paths,
        "paths",
        lambda: run_hedge_monte_carlo(
            spot0=100.0,
            strike=100.0,
            rate=0.03,
            dividend_yield=0.0,
            implied_vol=0.22,
            realized_vol=0.30,
            expiry=0.5,
            n_steps=252,
            n_paths=n_hedge_paths,
            seed=0,
        ),
    )
    print(t_hedge.report())


if __name__ == "__main__":
    main()
