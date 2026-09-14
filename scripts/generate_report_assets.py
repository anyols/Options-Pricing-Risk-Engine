"""Generate every chart used in the README/report into assets/ as PNGs.

Run with: python scripts/generate_report_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

from optrisk.market.sample_data import build_demo_portfolio  # noqa: E402
from optrisk.models.heston import heston_implied_vol_smile  # noqa: E402
from optrisk.risk.hedging import run_hedge_frequency_comparison, simulate_delta_hedge  # noqa: E402
from optrisk.risk.scenarios import default_shock_grid, run_scenario_analysis  # noqa: E402
from optrisk.viz.plots import (  # noqa: E402
    plot_greek_curves,
    plot_greeks_bar,
    plot_hedge_path,
    plot_hedge_pnl_distribution,
    plot_payoff_diagram,
    plot_pnl_heatmap,
    plot_taylor_error_heatmaps,
    plot_taylor_slice,
    plot_vol_smile,
)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    saved = []

    def save(fig: Figure, name: str) -> None:
        path = ASSETS / f"{name}.png"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path.name)
        print(f"  wrote {path.relative_to(ROOT)}")

    print("Building demo portfolio...")
    portfolio = build_demo_portfolio()
    print(
        portfolio.to_frame()[["label", "quantity", "price", "value", "delta", "gamma", "vega"]].to_string(index=False)
    )

    print("\n[1/7] Payoff diagram")
    save(plot_payoff_diagram(portfolio), "payoff_diagram")

    print("[2/7] Greek curves")
    save(plot_greek_curves(portfolio), "greek_curves")

    print("[3/7] Portfolio Greeks summary")
    save(plot_greeks_bar(portfolio.greeks()), "greeks_summary")

    print("[4/7] Scenario analysis: full reprice vs Taylor approximation")
    spot_shocks, vol_shocks = default_shock_grid(spot_range_pct=0.25, n_spot=41, vol_range=0.15, n_vol=31)
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)
    save(plot_pnl_heatmap(result), "pnl_heatmap_full_reprice")
    save(plot_taylor_error_heatmaps(result), "taylor_error_heatmaps")
    # slice at a meaningfully nonzero vol shock so the Vega/Vanna/Volga terms
    # actually differentiate the curves (at vol_shock=0 they contribute nothing)
    hero_vol_idx = int(0.83 * (len(vol_shocks) - 1))
    print(f"    hero slice at vol shock = {vol_shocks[hero_vol_idx] * 100:+.1f} pts")
    save(plot_taylor_slice(result, vol_shock_index=hero_vol_idx), "taylor_slice")
    for order in result.taylor_pnl:
        print(f"    max |error| ({order}): {result.max_abs_error(order):,.2f}")

    print("[5/7] Delta-hedging simulation")
    hedge = simulate_delta_hedge(
        spot0=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        implied_vol=0.22,
        realized_vol=0.34,
        expiry=0.5,
        option_type="call",
        option_quantity=1.0,
        n_steps=252,
        seed=7,
    )
    save(plot_hedge_path(hedge), "hedge_path")

    print("[6/7] Hedging P&L distribution vs rebalancing frequency")
    freq_frame = run_hedge_frequency_comparison(
        spot0=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        implied_vol=0.22,
        realized_vol=0.34,
        expiry=0.5,
        option_type="call",
        frequencies={"Daily": 252, "Weekly": 36, "Monthly": 12, "Quarterly": 4},
        n_paths=1000,
        seed=7,
    )
    save(plot_hedge_pnl_distribution(freq_frame), "hedge_pnl_distribution")
    print(freq_frame.groupby("frequency")["final_pnl"].agg(["mean", "std"]).to_string())

    print("[7/7] Heston-implied volatility smile vs flat BSM")
    strikes = np.linspace(70, 130, 25)
    heston_ivs = heston_implied_vol_smile(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        expiry=0.5,
        v0=0.045,
        kappa=1.8,
        theta=0.045,
        xi=0.55,
        rho=-0.75,
        strikes=strikes,
        option_type="call",
    )
    flat = np.full_like(strikes, np.sqrt(0.045))
    save(
        plot_vol_smile(strikes, {"Heston (stochastic vol)": heston_ivs, "Flat BSM assumption": flat}, spot=100.0),
        "vol_smile",
    )

    print(f"\nDone: {len(saved)} charts written to {ASSETS}")


if __name__ == "__main__":
    main()
