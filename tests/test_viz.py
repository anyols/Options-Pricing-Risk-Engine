"""Smoke tests for the chart library: every function must run end-to-end on
real engine output and return a figure. Visual correctness isn't something a
unit test can check, but "does it run without raising" catches the typo
class of bug that only ever surfaces when a chart is actually rendered.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from optrisk.greeks.types import Greeks
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.models.heston import heston_implied_vol_smile
from optrisk.risk.hedging import run_hedge_frequency_comparison, simulate_delta_hedge
from optrisk.risk.scenarios import default_shock_grid, run_scenario_analysis
from optrisk.viz.plots import (
    plot_greek_curves,
    plot_greeks_bar,
    plot_hedge_path,
    plot_hedge_pnl_distribution,
    plot_payoff_diagram,
    plot_pnl_heatmap,
    plot_taylor_error_heatmaps,
    plot_taylor_slice,
    plot_vol_smile,
    plotly_hedge_path,
    plotly_pnl_distribution,
    plotly_pnl_surface,
    plotly_vol_smile,
)


@pytest.fixture(scope="module")
def portfolio():
    return build_demo_portfolio()


@pytest.fixture(scope="module")
def scenario_result(portfolio):
    spot_shocks, vol_shocks = default_shock_grid(n_spot=9, n_vol=7)
    return run_scenario_analysis(portfolio, spot_shocks, vol_shocks)


@pytest.fixture(scope="module")
def hedge_result():
    return simulate_delta_hedge(
        spot0=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        implied_vol=0.25,
        realized_vol=0.3,
        expiry=0.5,
        option_type="call",
        n_steps=60,
        seed=1,
    )


@pytest.fixture(scope="module")
def frequency_frame():
    return run_hedge_frequency_comparison(
        spot0=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.0,
        implied_vol=0.25,
        realized_vol=0.3,
        expiry=0.5,
        frequencies={"Daily": 24, "Weekly": 6, "Monthly": 2},
        n_paths=30,
        seed=1,
    )


def _assert_is_figure(fig):
    assert fig is not None
    assert hasattr(fig, "savefig") or hasattr(fig, "to_dict")
    plt.close("all")


def test_plot_payoff_diagram(portfolio):
    _assert_is_figure(plot_payoff_diagram(portfolio))


def test_plot_greek_curves(portfolio):
    _assert_is_figure(plot_greek_curves(portfolio, spot_shocks_pct=np.linspace(-0.2, 0.2, 21)))


def test_plot_pnl_heatmap(scenario_result):
    _assert_is_figure(plot_pnl_heatmap(scenario_result))
    _assert_is_figure(plot_pnl_heatmap(scenario_result, order="delta_gamma"))


def test_plot_taylor_error_heatmaps(scenario_result):
    _assert_is_figure(plot_taylor_error_heatmaps(scenario_result))


def test_plot_taylor_slice(scenario_result):
    _assert_is_figure(plot_taylor_slice(scenario_result))


def test_plot_hedge_path(hedge_result):
    _assert_is_figure(plot_hedge_path(hedge_result))


def test_plot_hedge_pnl_distribution(frequency_frame):
    _assert_is_figure(plot_hedge_pnl_distribution(frequency_frame))


def test_plot_vol_smile():
    strikes = np.linspace(80, 120, 9)
    ivs = heston_implied_vol_smile(
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        expiry=1.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        strikes=strikes,
        option_type="call",
    )
    _assert_is_figure(plot_vol_smile(strikes, {"Heston": ivs, "Flat BSM": np.full_like(ivs, 0.2)}, spot=100.0))


def test_plot_greeks_bar():
    g = Greeks(delta=120.0, gamma=15.0, vega=-40.0, theta=-8.0, rho=25.0)
    _assert_is_figure(plot_greeks_bar(g))


def test_plotly_pnl_surface(scenario_result):
    fig = plotly_pnl_surface(scenario_result)
    assert fig.data


def test_plotly_hedge_path(hedge_result):
    fig = plotly_hedge_path(hedge_result)
    assert fig.data


def test_plotly_vol_smile():
    strikes = np.linspace(80, 120, 5)
    fig = plotly_vol_smile(strikes, {"Heston": np.full(5, 0.25)}, spot=100.0)
    assert fig.data


def test_plotly_pnl_distribution(frequency_frame):
    fig = plotly_pnl_distribution(frequency_frame)
    assert fig.data
