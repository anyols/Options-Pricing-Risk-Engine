"""Statistical tests for the GBM Monte Carlo pricer (fixed seeds for reproducibility).

Variance-reduction tests deliberately compare the *empirical variance of the
price estimator across many independent runs*, not a single run's internal
std_error -- a single-seed std_error comparison is itself a noisy estimate
and can occasionally flip order by chance even when the technique reliably
reduces variance in expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.models.black_scholes import bsm_price
from optrisk.models.monte_carlo import mc_price

SEED = 7
CASE = {"spot": 100.0, "strike": 100.0, "rate": 0.04, "dividend_yield": 0.01, "vol": 0.3, "expiry": 1.0}
N_REPS = 25
REP_PATHS = 2_000


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_matches_bsm_within_statistical_error(option_type):
    result = mc_price(**CASE, option_type=option_type, n_paths=200_000, seed=SEED)
    bsm = bsm_price(**CASE, option_type=option_type)
    assert abs(result.price - bsm) < 4 * result.std_error


def test_mc_put_call_parity_within_statistical_error():
    call = mc_price(**CASE, option_type="call", n_paths=200_000, seed=SEED)
    put = mc_price(**CASE, option_type="put", n_paths=200_000, seed=SEED + 1)
    spot, strike, rate, q, expiry = CASE["spot"], CASE["strike"], CASE["rate"], CASE["dividend_yield"], CASE["expiry"]

    parity_rhs = spot * np.exp(-q * expiry) - strike * np.exp(-rate * expiry)
    combined_se = (call.std_error**2 + put.std_error**2) ** 0.5
    assert abs((call.price - put.price) - parity_rhs) < 4 * combined_se


def test_control_variate_reduces_estimator_variance():
    plain = [
        mc_price(**CASE, option_type="call", n_paths=REP_PATHS, antithetic=False, control_variate=False, seed=s).price
        for s in range(N_REPS)
    ]
    controlled = [
        mc_price(**CASE, option_type="call", n_paths=REP_PATHS, antithetic=False, control_variate=True, seed=s).price
        for s in range(N_REPS)
    ]
    assert np.var(controlled) < np.var(plain)


def test_antithetic_reduces_estimator_variance():
    plain = [
        mc_price(**CASE, option_type="call", n_paths=REP_PATHS, antithetic=False, control_variate=False, seed=s).price
        for s in range(N_REPS)
    ]
    antithetic = [
        mc_price(**CASE, option_type="call", n_paths=REP_PATHS, antithetic=True, control_variate=False, seed=s).price
        for s in range(N_REPS)
    ]
    assert np.var(antithetic) < np.var(plain)


def test_reproducible_with_fixed_seed():
    a = mc_price(**CASE, option_type="call", n_paths=10_000, seed=42)
    b = mc_price(**CASE, option_type="call", n_paths=10_000, seed=42)
    assert a.price == b.price


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_price(**CASE, option_type="straddle")
