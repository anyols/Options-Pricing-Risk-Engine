"""Correctness tests for the Heston COS pricer.

Two independent validation routes, neither of which trusts a memorized
"reference price" from a textbook: (1) the analytic degenerate-case limit,
where zero vol-of-vol collapses Heston to BSM exactly, and (2) agreement
with an independent Heston Monte Carlo simulation across strikes and option
types.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.models.black_scholes import bsm_price
from optrisk.models.heston import _heston_char_func, heston_implied_vol_smile, heston_mc_price, heston_price

BASE = dict(spot=100.0, rate=0.03, dividend_yield=0.0, expiry=1.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)


def test_characteristic_function_at_zero_is_one():
    # phi(0) = E[e^0] = 1 always; a basic normalization sanity check
    cf = _heston_char_func(
        np.array([0.0]), x=0.3, v0=0.05, kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, rate=0.03, dividend_yield=0.01, tau=1.0
    )
    assert cf[0] == pytest.approx(1.0 + 0.0j, abs=1e-10)


def test_put_call_parity():
    params = dict(spot=100.0, strike=105.0, rate=0.03, dividend_yield=0.01, expiry=0.75, v0=0.05, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6)
    call = heston_price(**params, option_type="call")
    put = heston_price(**params, option_type="put")
    rhs = params["spot"] * np.exp(-params["dividend_yield"] * params["expiry"]) - params["strike"] * np.exp(
        -params["rate"] * params["expiry"]
    )
    assert (call - put) == pytest.approx(rhs, abs=1e-4)


def test_degenerate_case_matches_bsm():
    # xi -> 0 with v0 = theta collapses the Heston SDE to GBM with constant
    # vol sqrt(theta); xi=0.05 is small enough to demonstrate the limit
    # without hitting the 1/xi^2 floating-point cancellation that appears
    # for xi below ~0.01 (a known characteristic of this parameterization,
    # not a sign of an incorrect formula -- see scratch exploration).
    theta = 0.04
    heston = heston_price(spot=100.0, strike=100.0, rate=0.03, dividend_yield=0.0, expiry=1.0, v0=theta, kappa=1.5, theta=theta, xi=0.05, rho=-0.5, option_type="call")
    bsm = bsm_price(100.0, 100.0, 0.03, 0.0, np.sqrt(theta), 1.0, "call")
    assert heston == pytest.approx(bsm, abs=1e-2)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_cos_matches_independent_monte_carlo(strike, option_type):
    cos_price = heston_price(**BASE, strike=strike, option_type=option_type)
    mc = heston_mc_price(**BASE, strike=strike, option_type=option_type, n_paths=150_000, n_steps=150, seed=1)
    assert abs(cos_price - mc.price) < 5 * mc.std_error


def test_price_is_positive_and_finite():
    for strike in (50.0, 100.0, 200.0):
        price = heston_price(**BASE, strike=strike, option_type="call")
        assert np.isfinite(price)
        assert price >= 0.0


def test_converges_with_more_series_terms():
    prices = [heston_price(**BASE, strike=100.0, n_terms=n) for n in (48, 96, 160, 256)]
    # later refinements should agree closely with each other
    assert abs(prices[-1] - prices[-2]) < 1e-6
    assert abs(prices[-1] - prices[0]) < 1e-2


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        heston_price(**BASE, strike=100.0, option_type="straddle")


def test_implied_vol_smile_is_not_flat_and_reflects_negative_skew():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    ivs = heston_implied_vol_smile(
        spot=100.0, rate=0.03, dividend_yield=0.0, expiry=1.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        strikes=strikes, option_type="call",
    )
    assert np.all(np.isfinite(ivs))
    assert np.ptp(ivs) > 0.01  # a real smile, not (numerically) flat
    # rho < 0 is the classic equity case: low strikes carry higher implied vol than high strikes
    assert ivs[0] > ivs[-1]
