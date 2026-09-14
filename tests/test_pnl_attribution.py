"""Tests for the Greek-bucket P&L attribution ("P&L explain") engine."""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.instruments.option import MarketEnv, OptionSpec, Stock
from optrisk.instruments.portfolio import Portfolio, Position
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.risk.pnl_attribution import attribute_pnl

EQ = MarketEnv(spot=100.0, rate=0.04, vol=0.25, dividend_yield=0.01)
ONE_TRADING_DAY = 1.0 / 252.0


def _single_call_portfolio() -> Portfolio:
    spec = OptionSpec(option_type="call", strike=100.0, expiry=0.5, style="european", underlying_type="equity")
    return Portfolio(positions=[Position(instrument=spec, quantity=10, market=EQ, multiplier=100)])


def _delta_hedged_straddle_portfolio() -> Portfolio:
    call = OptionSpec(option_type="call", strike=100.0, expiry=0.5, style="european", underlying_type="equity")
    put = OptionSpec(option_type="put", strike=100.0, expiry=0.5, style="european", underlying_type="equity")
    portfolio = Portfolio(
        positions=[
            Position(instrument=call, quantity=10, market=EQ, multiplier=100, label="call"),
            Position(instrument=put, quantity=10, market=EQ, multiplier=100, label="put"),
        ]
    )
    net_delta = portfolio.greeks().delta
    portfolio.add(Position(instrument=Stock(), quantity=-net_delta, market=EQ, multiplier=1, label="hedge"))
    return portfolio


def test_no_move_gives_zero_everywhere():
    attribution = attribute_pnl(_single_call_portfolio())
    assert attribution.full_pnl == pytest.approx(0.0, abs=1e-9)
    assert attribution.explained_pnl == pytest.approx(0.0, abs=1e-9)
    assert attribution.unexplained_pnl == pytest.approx(0.0, abs=1e-9)
    assert attribution.unexplained_pct == 0.0


def test_explained_plus_unexplained_equals_full():
    attribution = attribute_pnl(_single_call_portfolio(), spot_shock_pct=0.08, vol_shock=0.03, time_elapsed=0.05)
    assert attribution.explained_pnl + attribution.unexplained_pnl == pytest.approx(attribution.full_pnl, rel=1e-9)


def test_pure_time_decay_is_almost_entirely_explained_by_theta():
    # only time moves: with spot/vol unchanged, theta alone should account
    # for nearly all of the realized P&L over a single trading day.
    attribution = attribute_pnl(_single_call_portfolio(), time_elapsed=ONE_TRADING_DAY)
    assert attribution.delta_pnl == 0.0
    assert attribution.gamma_pnl == 0.0
    assert attribution.vega_pnl == 0.0
    assert attribution.cross_pnl == 0.0
    assert attribution.theta_pnl != 0.0
    assert attribution.unexplained_pct < 1.0  # well under 1% of the realized move


def test_small_combined_move_has_small_unexplained_fraction():
    attribution = attribute_pnl(
        _single_call_portfolio(), spot_shock_pct=0.01, vol_shock=0.005, time_elapsed=ONE_TRADING_DAY
    )
    assert attribution.unexplained_pct < 5.0


def test_large_move_has_larger_unexplained_fraction_than_small_move():
    small = attribute_pnl(_single_call_portfolio(), spot_shock_pct=0.01, vol_shock=0.005)
    large = attribute_pnl(_single_call_portfolio(), spot_shock_pct=0.25, vol_shock=0.10)
    assert large.unexplained_pct > small.unexplained_pct


def test_delta_hedged_book_has_near_zero_delta_bucket():
    attribution = attribute_pnl(_delta_hedged_straddle_portfolio(), spot_shock_pct=0.05)
    assert attribution.delta_pnl == pytest.approx(0.0, abs=1e-6)
    # long gamma, delta-hedged: the realized P&L should still be positive (gamma bucket dominates)
    assert attribution.gamma_pnl > 0
    assert attribution.full_pnl > 0


def test_demo_portfolio_runs_and_is_finite():
    attribution = attribute_pnl(
        build_demo_portfolio(), spot_shock_pct=-0.03, vol_shock=0.02, time_elapsed=ONE_TRADING_DAY
    )
    for value in (
        attribution.delta_pnl,
        attribution.gamma_pnl,
        attribution.vega_pnl,
        attribution.theta_pnl,
        attribution.cross_pnl,
        attribution.full_pnl,
        attribution.unexplained_pnl,
    ):
        assert np.isfinite(value)


def test_waterfall_frame_sums_to_full_pnl():
    attribution = attribute_pnl(
        _single_call_portfolio(), spot_shock_pct=0.04, vol_shock=-0.02, time_elapsed=ONE_TRADING_DAY
    )
    frame = attribution.to_waterfall_frame()
    assert list(frame["bucket"]) == ["Delta", "Gamma", "Vega", "Theta", "Vanna/Volga", "Unexplained"]
    assert frame["pnl"].sum() == pytest.approx(attribution.full_pnl, rel=1e-9)
