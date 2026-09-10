"""Correctness tests for analytical and numerical Greeks.

Strategy: cross-validate the closed-form Greeks against the independent
finite-difference engine (not against memorized numbers), plus a handful of
exact algebraic identities that follow from put-call parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from optrisk.greeks.analytical import (
    black76_full_greeks,
    black76_greeks,
    bsm_full_greeks,
    bsm_greeks,
)
from optrisk.greeks.numerical import numerical_greeks
from optrisk.models.black76 import black76_price
from optrisk.models.black_scholes import bsm_price

BSM_CASES = [
    # spot, strike, rate, div_yield, vol, expiry
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.0),
    (100.0, 120.0, 0.03, 0.01, 0.35, 0.25),
    (80.0, 70.0, 0.01, 0.00, 0.15, 2.0),
    (50.0, 55.0, 0.07, 0.03, 0.60, 0.75),
]

FIRST_SECOND_ORDER = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]


@pytest.mark.parametrize("spot,strike,rate,q,vol,expiry", BSM_CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bsm_analytical_matches_finite_difference(spot, strike, rate, q, vol, expiry, option_type):
    analytical = bsm_greeks(spot, strike, rate, q, vol, expiry, option_type)

    def pricer(spot, vol, expiry, rate):
        return bsm_price(spot, strike, rate, q, vol, expiry, option_type)

    numerical = numerical_greeks(pricer, spot=spot, vol=vol, expiry=expiry, rate=rate)
    for name in FIRST_SECOND_ORDER:
        a, n_ = getattr(analytical, name), getattr(numerical, name)
        assert a == pytest.approx(n_, abs=5e-3, rel=5e-3), f"{name}: analytical={a} fd={n_}"


@pytest.mark.parametrize("forward,strike,rate,vol,expiry", [(f, k, r, v, t) for f, k, r, _, v, t in BSM_CASES])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_analytical_matches_finite_difference(forward, strike, rate, vol, expiry, option_type):
    analytical = black76_greeks(forward, strike, rate, vol, expiry, option_type)

    def pricer(spot, vol, expiry, rate):
        return black76_price(spot, strike, rate, vol, expiry, option_type)

    numerical = numerical_greeks(pricer, spot=forward, vol=vol, expiry=expiry, rate=rate)
    for name in FIRST_SECOND_ORDER:
        a, n_ = getattr(analytical, name), getattr(numerical, name)
        assert a == pytest.approx(n_, abs=5e-3, rel=5e-3), f"{name}: analytical={a} fd={n_}"


@pytest.mark.parametrize("spot,strike,rate,q,vol,expiry", BSM_CASES)
def test_gamma_vanna_volga_identical_for_call_and_put(spot, strike, rate, q, vol, expiry):
    # C - P = S e^{-qT} - K e^{-rT} is linear in S and independent of sigma,
    # so every S/sigma cross-derivative must be identical for call and put.
    call = bsm_greeks(spot, strike, rate, q, vol, expiry, "call")
    put = bsm_greeks(spot, strike, rate, q, vol, expiry, "put")
    assert call.gamma == pytest.approx(put.gamma, rel=1e-9)
    assert call.vanna == pytest.approx(put.vanna, rel=1e-9)
    assert call.volga == pytest.approx(put.volga, rel=1e-9)


@pytest.mark.parametrize("spot,strike,rate,q,vol,expiry", BSM_CASES)
def test_call_delta_minus_put_delta_equals_disc_dividend(spot, strike, rate, q, vol, expiry):
    call = bsm_greeks(spot, strike, rate, q, vol, expiry, "call")
    put = bsm_greeks(spot, strike, rate, q, vol, expiry, "put")
    assert (call.delta - put.delta) == pytest.approx(np.exp(-q * expiry), abs=1e-10)


def test_call_delta_bounded_by_disc_dividend():
    for spot, strike, rate, q, vol, expiry in BSM_CASES:
        g = bsm_greeks(spot, strike, rate, q, vol, expiry, "call")
        assert 0.0 <= g.delta <= np.exp(-q * expiry) + 1e-12


def test_gamma_is_positive():
    for spot, strike, rate, q, vol, expiry in BSM_CASES:
        for opt in ("call", "put"):
            g = bsm_greeks(spot, strike, rate, q, vol, expiry, opt)
            assert g.gamma > 0


def test_atm_no_dividend_call_theta_is_negative():
    g = bsm_greeks(spot=100.0, strike=100.0, rate=0.03, dividend_yield=0.0, vol=0.25, expiry=1.0, option_type="call")
    assert g.theta < 0


# --- third-order Greeks: cross-check the engine's mixed-partial stencils
# against a central difference of the *analytical* lower-order Greek taken
# through an independent code path. ---


def test_charm_matches_finite_difference_of_analytical_delta():
    spot, strike, rate, q, vol, expiry = 100.0, 105.0, 0.04, 0.01, 0.3, 1.0
    h = 1e-4
    delta_up = bsm_greeks(spot, strike, rate, q, vol, expiry + h, "call").delta
    delta_dn = bsm_greeks(spot, strike, rate, q, vol, expiry - h, "call").delta
    charm_from_delta = -(delta_up - delta_dn) / (2 * h)
    charm_from_engine = bsm_full_greeks(spot, strike, rate, q, vol, expiry, "call").charm
    assert charm_from_engine == pytest.approx(charm_from_delta, abs=5e-3, rel=5e-3)


def test_speed_matches_finite_difference_of_analytical_gamma():
    spot, strike, rate, q, vol, expiry = 100.0, 105.0, 0.04, 0.01, 0.3, 1.0
    h = max(spot * 1e-3, 1e-3)
    gamma_up = bsm_greeks(spot + h, strike, rate, q, vol, expiry, "call").gamma
    gamma_dn = bsm_greeks(spot - h, strike, rate, q, vol, expiry, "call").gamma
    speed_from_gamma = (gamma_up - gamma_dn) / (2 * h)
    speed_from_engine = bsm_full_greeks(spot, strike, rate, q, vol, expiry, "call").speed
    assert speed_from_engine == pytest.approx(speed_from_gamma, abs=5e-5, rel=5e-2)


def test_zomma_matches_finite_difference_of_analytical_gamma():
    spot, strike, rate, q, vol, expiry = 100.0, 105.0, 0.04, 0.01, 0.3, 1.0
    h = 1e-4
    gamma_up = bsm_greeks(spot, strike, rate, q, vol + h, expiry, "call").gamma
    gamma_dn = bsm_greeks(spot, strike, rate, q, vol - h, expiry, "call").gamma
    zomma_from_gamma = (gamma_up - gamma_dn) / (2 * h)
    zomma_from_engine = bsm_full_greeks(spot, strike, rate, q, vol, expiry, "call").zomma
    assert zomma_from_engine == pytest.approx(zomma_from_gamma, abs=5e-3, rel=5e-2)


def test_color_matches_finite_difference_of_analytical_gamma():
    spot, strike, rate, q, vol, expiry = 100.0, 105.0, 0.04, 0.01, 0.3, 1.0
    h = 1e-4
    gamma_up = bsm_greeks(spot, strike, rate, q, vol, expiry + h, "call").gamma
    gamma_dn = bsm_greeks(spot, strike, rate, q, vol, expiry - h, "call").gamma
    color_from_gamma = -(gamma_up - gamma_dn) / (2 * h)
    color_from_engine = bsm_full_greeks(spot, strike, rate, q, vol, expiry, "call").color
    assert color_from_engine == pytest.approx(color_from_gamma, abs=5e-3, rel=5e-2)


def test_black76_full_greeks_runs_and_is_finite():
    g = black76_full_greeks(100.0, 95.0, 0.03, 0.3, 0.5, "put")
    for value in g.as_dict().values():
        assert np.isfinite(value)


def test_greeks_arithmetic_and_sum():
    from optrisk.greeks.types import Greeks

    a = Greeks(delta=0.5, gamma=0.1)
    b = Greeks(delta=-0.2, gamma=0.05)
    assert (a + b).delta == pytest.approx(0.3)
    assert (2.0 * a).delta == pytest.approx(1.0)
    assert (a - b).delta == pytest.approx(0.7)
    total = sum([a, b, a])
    assert total.delta == pytest.approx(0.8)
