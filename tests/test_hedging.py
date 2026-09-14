"""Tests for the delta-hedging simulator.

Statistical assertions use fixed seeds and comparative (not absolute)
thresholds where possible, since these are inherently Monte Carlo
quantities; the deterministic convergence test avoids randomness entirely
by refining a *fixed* Brownian realization rather than drawing new paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.risk.hedging import (
    run_hedge_frequency_comparison,
    run_hedge_monte_carlo,
    simulate_delta_hedge,
)

CASE = {"spot0": 100.0, "strike": 100.0, "rate": 0.03, "dividend_yield": 0.0, "expiry": 0.5}


def test_hedge_is_zero_cost_at_inception():
    result = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.25, option_type="call", seed=1)
    assert result.portfolio_value_path[0] == pytest.approx(0.0, abs=1e-8)


def test_hedging_error_shrinks_as_the_same_path_is_refined():
    # build one fine Brownian path and subsample it at coarser resolutions,
    # so refining the grid is the *same* realization observed more often --
    # under matched realized/implied vol, the BS replication argument says
    # hedging error should shrink toward zero as rebalancing frequency grows.
    rng = np.random.default_rng(123)
    finest = 2048
    vol = 0.3
    dt = CASE["expiry"] / finest
    z = rng.standard_normal(finest)
    log_returns = (CASE["rate"] - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    fine_path = CASE["spot0"] * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

    errors = []
    for steps in (8, 64, 512, 2048):
        stride = finest // steps
        sub_path = fine_path[::stride]
        result = simulate_delta_hedge(
            **CASE, implied_vol=vol, realized_vol=vol, option_type="call", n_steps=steps, spot_path=sub_path
        )
        errors.append(abs(result.final_pnl))

    assert errors[-1] < errors[0]
    assert errors[-1] < 0.25 * errors[0]  # substantial reduction, not just any reduction
    assert errors[-1] < 0.5  # small relative to the ~$100 underlying / several-dollar option value


def test_long_gamma_profits_when_realized_exceeds_implied_on_average():
    high_realized = run_hedge_monte_carlo(
        **CASE, implied_vol=0.20, realized_vol=0.40, option_type="call", option_quantity=1.0, n_paths=400, seed=42
    )
    low_realized = run_hedge_monte_carlo(
        **CASE, implied_vol=0.20, realized_vol=0.10, option_type="call", option_quantity=1.0, n_paths=400, seed=43
    )
    # more realized vol than you hedged for -> better P&L for a long-gamma (long option) position
    assert high_realized.mean() > low_realized.mean()
    assert high_realized.mean() > 0


def test_short_option_loses_when_realized_exceeds_implied_on_average():
    pnls = run_hedge_monte_carlo(
        **CASE, implied_vol=0.20, realized_vol=0.45, option_type="call", option_quantity=-1.0, n_paths=400, seed=42
    )
    assert pnls.mean() < 0


def test_long_and_short_are_mirror_images_on_the_same_path():
    long_result = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.35, option_quantity=1.0, seed=7)
    short_result = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.35, option_quantity=-1.0, seed=7)
    assert short_result.final_pnl == pytest.approx(-long_result.final_pnl, abs=1e-8)


def test_rebalancing_more_frequently_reduces_pnl_variance():
    frame = run_hedge_frequency_comparison(
        **CASE,
        implied_vol=0.20,
        realized_vol=0.35,
        option_type="put",
        frequencies={"Daily": 126, "Weekly": 18, "Monthly": 6},
        n_paths=250,
        seed=11,
    )
    variance_by_freq = frame.groupby("frequency")["final_pnl"].var()
    assert variance_by_freq["Daily"] < variance_by_freq["Monthly"]


def test_reproducible_with_fixed_seed():
    a = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.25, seed=99)
    b = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.25, seed=99)
    assert a.final_pnl == b.final_pnl


def test_mismatched_spot_path_length_raises():
    with pytest.raises(ValueError):
        simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.25, n_steps=50, spot_path=np.full(10, 100.0))


def test_frequencies_must_evenly_divide_finest():
    with pytest.raises(ValueError):
        run_hedge_frequency_comparison(
            **CASE, implied_vol=0.2, realized_vol=0.2, frequencies={"A": 10, "B": 7}, n_paths=2
        )


def test_result_to_frame_has_expected_columns():
    result = simulate_delta_hedge(**CASE, implied_vol=0.25, realized_vol=0.25, seed=1, n_steps=10)
    frame = result.to_frame()
    assert len(frame) == 11
    assert {"time", "spot", "delta", "portfolio_value"}.issubset(frame.columns)
