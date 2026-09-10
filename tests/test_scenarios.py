"""Tests for the full-reprice-vs-Taylor-approximation scenario engine."""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.instruments.option import MarketEnv, OptionSpec, Stock
from optrisk.instruments.portfolio import Portfolio, Position
from optrisk.market.sample_data import build_demo_portfolio
from optrisk.risk.scenarios import TAYLOR_ORDERS, default_shock_grid, run_scenario_analysis

EQ = MarketEnv(spot=100.0, rate=0.04, vol=0.25, dividend_yield=0.01)


def _single_call_portfolio() -> Portfolio:
    spec = OptionSpec(option_type="call", strike=100.0, expiry=0.5, style="european", underlying_type="equity")
    return Portfolio(positions=[Position(instrument=spec, quantity=10, market=EQ, multiplier=100)])


def _delta_hedged_straddle_portfolio() -> Portfolio:
    # a raw ATM straddle is *not* delta-neutral in general (lognormal
    # skew + any (r - q) drift gives it a small net delta), so hedge it
    # explicitly with stock to isolate the pure long-gamma exposure --
    # exactly the "delta-hedging" construction the rest of this project is about.
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


def test_default_shock_grid_shape_and_symmetry():
    spot_shocks, vol_shocks = default_shock_grid(spot_range_pct=0.2, n_spot=21, vol_range=0.1, n_vol=11)
    assert len(spot_shocks) == 21
    assert len(vol_shocks) == 11
    assert spot_shocks[0] == pytest.approx(-0.2)
    assert spot_shocks[-1] == pytest.approx(0.2)
    assert 0.0 in spot_shocks  # odd length -> midpoint is exactly zero
    assert 0.0 in vol_shocks


def test_zero_shock_gives_zero_pnl_everywhere():
    portfolio = _single_call_portfolio()
    spot_shocks = np.array([-0.1, 0.0, 0.1])
    vol_shocks = np.array([-0.05, 0.0, 0.05])
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    assert result.full_pnl[1, 1] == pytest.approx(0.0, abs=1e-8)
    for order in TAYLOR_ORDERS:
        assert result.taylor_pnl[order][1, 1] == pytest.approx(0.0, abs=1e-8)


def test_richer_taylor_order_is_more_accurate_for_small_shocks():
    portfolio = _single_call_portfolio()
    spot_shocks = np.array([-0.03, 0.0, 0.03])
    vol_shocks = np.array([0.0])
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    err_delta = abs(result.error("delta")[0, 0])
    err_delta_gamma = abs(result.error("delta_gamma")[0, 0])
    # including convexity (Gamma) must not make things worse, and for a
    # genuinely curved payoff it should meaningfully improve accuracy
    assert err_delta_gamma <= err_delta
    assert err_delta_gamma < 0.5 * err_delta


def test_delta_only_error_grows_with_shock_size():
    portfolio = _single_call_portfolio()
    spot_shocks = np.array([-0.30, -0.02, 0.0, 0.02, 0.30])
    vol_shocks = np.array([0.0])
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    small_move_error = abs(result.error("delta")[0, 3])  # +2%
    large_move_error = abs(result.error("delta")[0, 4])  # +30%
    assert large_move_error > small_move_error


def test_delta_hedged_straddle_is_long_gamma_pnl_positive_both_ways_while_delta_predicts_flat():
    portfolio = _delta_hedged_straddle_portfolio()
    spot_shocks = np.array([-0.05, 0.0, 0.05])
    vol_shocks = np.array([0.0])
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    # delta-hedged, so the delta-only approx predicts ~zero P&L on both sides ...
    assert result.taylor_pnl["delta"][0, 0] == pytest.approx(0.0, abs=1e-6)
    assert result.taylor_pnl["delta"][0, 2] == pytest.approx(0.0, abs=1e-6)
    # ... but full repricing shows the real, positive long-gamma P&L on both sides
    assert result.full_pnl[0, 0] > 0
    assert result.full_pnl[0, 2] > 0


def test_taylor_matches_aggregated_greeks_formula_for_single_underlying_book():
    portfolio = _single_call_portfolio()
    spot_shocks = np.array([0.04])
    vol_shocks = np.array([0.02])
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    g = portfolio.greeks()
    spot0 = portfolio.positions[0].market.spot
    d_spot = spot0 * 0.04
    d_vol = 0.02
    expected_full_2nd_order = (
        g.delta * d_spot + 0.5 * g.gamma * d_spot**2 + g.vega * d_vol + g.vanna * d_spot * d_vol + 0.5 * g.volga * d_vol**2
    )
    assert result.taylor_pnl["full_2nd_order"][0, 0] == pytest.approx(expected_full_2nd_order, rel=1e-9)


def test_scenario_engine_runs_on_multi_underlying_demo_portfolio():
    portfolio = build_demo_portfolio()
    spot_shocks, vol_shocks = default_shock_grid(n_spot=5, n_vol=5)
    result = run_scenario_analysis(portfolio, spot_shocks, vol_shocks)

    assert result.full_pnl.shape == (5, 5)
    assert np.all(np.isfinite(result.full_pnl))
    for order in TAYLOR_ORDERS:
        assert np.all(np.isfinite(result.taylor_pnl[order]))

    frame = result.to_long_frame()
    assert len(frame) == 25
    assert "full_pnl" in frame.columns
    assert "delta_gamma_error" in frame.columns


def test_unknown_taylor_order_raises():
    from optrisk.risk.scenarios import _position_taylor_pnl
    from optrisk.greeks.types import Greeks

    with pytest.raises(ValueError):
        _position_taylor_pnl(Greeks(), 100.0, 0.01, 0.0, "cubic")
